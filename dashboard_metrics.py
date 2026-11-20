"""Metrics for the Lead Generation dashboard, derived from the four CSV exports."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd

from leadgen_supporting_analysis import DAY, ID, KEY, account_ids, after_launch, load_data

EXPORTS = {
    "accounts": "Accounts",
    "builder_events": "Builder events",
    "publishing": "Publishing",
    "lead_generation": "Lead generation",
}
DIMENSIONS = {
    "Agency size": "agency_segment",
    "Builder device": "primary_builder_device",
    "Plan": "plan",
    "Industry": "industry",
    "First builder product": "builder_product",
}
OPENING_TEMPLATE = "Opening-day template selection"
TEMPLATE_GROUPS = ["Selected a template", "No template selected", "No builder open"]
ONBOARDING_DIMENSIONS = {**DIMENSIONS, OPENING_TEMPLATE: "opening_template"}
DEFAULT_EXPORT_NAMES = {stem: f"{stem} - {stem}.csv" for stem in EXPORTS}


@dataclass
class Snapshot:
    accounts: pd.DataFrame
    activity: pd.DataFrame
    additional_assets: pd.DataFrame
    lead_days: pd.DataFrame
    cutoff: pd.Timestamp

    @property
    def fixed_panel(self):
        a = self.accounts
        complete_month = (self.cutoff - 84 * DAY).to_period("M").start_time
        return a.index[a.initially_active & a.launch.lt(complete_month)]


def snapshot_from_exports(files: dict[str, Path | bytes]) -> Snapshot:
    """Apply the assignment's data definitions before grouping or filtering."""
    sources = {k: BytesIO(v) if isinstance(v, bytes) else v for k, v in files.items()}
    ends = pd.read_csv(sources["accounts"], usecols=["observation_end_date"], parse_dates=["observation_end_date"])
    cutoff = ends.observation_end_date.max()
    if pd.isna(cutoff):
        raise ValueError("The accounts export needs at least one valid observation_end_date.")
    if isinstance(sources["accounts"], BytesIO):
        sources["accounts"].seek(0)
    a, builder, publishes, leads = load_data(None, cutoff=cutoff, sources=sources)
    if a.empty:
        raise ValueError("The accounts export has no rows.")

    first = publishes.drop_duplicates(KEY).rename(columns={"published_date": "launch"})
    if first[ID].duplicated().any():
        raise ValueError("The assignment attributes each sub-account to one published product. The uploaded exports contain accounts that published multiple products.")
    more, actions, leads = after_launch(publishes, first, builder, leads)
    a = a.join(first.set_index(ID)[["launch", "used_template"]])
    a["ttp"] = (a.launch - a.signup_date).dt.days
    age = (a.subscription_churn_date - a.signup_date).dt.days
    a["canceled"] = a.subscription_churn_date.le(a.end)
    a["canceled_90d"] = age.between(0, 90) & a.canceled
    a["eligible_90d"] = a.signup_date.add(90 * DAY).le(a.end)
    a["subscribed_day_14"] = ~age.le(14)

    def events(mask):
        return builder.loc[mask].sort_values(["event_timestamp", "product_type"]).drop_duplicates(ID).set_index(ID)

    first_open = events(builder.event_name.eq("builder_opened"))
    first_failure = events(builder.validation_failure)
    a["open"] = first_open.day
    a["builder_product"] = first_open.product_type.fillna("Unknown")
    a["builder_product"] = a.builder_product.fillna("No builder recorded")
    a["first_error"] = first_failure.day
    same_product_first = first.set_index(KEY).launch
    for frame, label in [(first_open, "open_to_publish"), (first_failure, "error_to_publish")]:
        launch = same_product_first.reindex(frame.set_index("product_type", append=True).index)
        a[label] = (launch.set_axis(frame.index) - frame.day).dt.days

    template = builder.loc[builder.event_name.eq("template_selected"), KEY + ["day"]].drop_duplicates()
    same_day = first_open.reset_index().merge(template, on=KEY + ["day"])
    a["opening_template"] = TEMPLATE_GROUPS[2]
    a.loc[a.open.notna(), "opening_template"] = TEMPLATE_GROUPS[1]
    a.loc[a.index.isin(same_day[ID]), "opening_template"] = TEMPLATE_GROUPS[0]
    opened = builder.loc[builder.event_name.eq("builder_opened")].groupby(KEY).day.min()
    created = builder.loc[builder.event_name.eq("asset_created")].groupby(KEY).day.min()
    milestones = pd.concat([opened.rename("open"), created.rename("create"), same_product_first], axis=1)
    creators = milestones.loc[milestones.create.ge(milestones.open)]
    funnel_published = creators.loc[creators.launch.ge(creators.create)].index
    lead_pairs = leads.loc[leads.leads.gt(0)].set_index(KEY).index
    a["created_asset"] = a.index.isin(creators.index.get_level_values(ID))
    a["funnel_published"] = a.index.isin(funnel_published.get_level_values(ID))
    a["funnel_lead"] = a.index.isin(funnel_published.intersection(lead_pairs).get_level_values(ID))

    initial = account_ids(actions.loc[actions.age.between(1, 28)])
    subscribed_through_28 = a.subscription_churn_date.isna() | a.subscription_churn_date.gt(a.launch + 28 * DAY)
    a["initially_active"] = a.index.isin(initial) & a.launch.add(84 * DAY).le(a.end) & subscribed_through_28

    # Preserve the separate edit-only series used to investigate the April dip.
    keys = more.rename(columns={"published_date": "day"})[KEY + ["day"]].drop_duplicates()
    activity = actions.merge(keys.assign(published_again=True), on=KEY + ["day"], how="left")
    activity["published_again"] = activity.published_again.eq(True)
    return Snapshot(
        a,
        activity[[ID, "day", "published_again"]],
        more[[ID, "published_date"]],
        leads[[ID, "activity_date", "leads", "visitors"]],
        cutoff,
    )


def select_accounts(a, filters, dimensions=DIMENSIONS):
    for name, choices in filters.items():
        if choices:
            a = a.loc[a[dimensions[name]].isin(choices)]
    return a


def rate(n, d):
    return n / d if d else float("nan")


def by_group(a, split):
    if split is None:
        return [("All selected", a)]
    return a.groupby(split, dropna=False, sort=True, observed=True)


def rate_table(a, numerator, split=None):
    rows = []
    for label, group in by_group(a, split):
        n, d = int(numerator.reindex(group.index).sum()), len(group)
        rows.append((str(label), n, d, rate(n, d)))
    return pd.DataFrame(rows, columns=["Group", "Count", "Eligible", "Rate"])


def weekly_activation(a, week_or_month):
    eligible = a.loc[a.signup_date.add(7 * DAY).le(a.end)]
    period = eligible.signup_date.dt.to_period(week_or_month).dt.start_time
    table = rate_table(eligible.assign(period=period), eligible.ttp.between(0, 7), "period")
    return table.rename(columns={"Group": "Period"})


def template_publish_curve(a):
    eligible = a.loc[a.open.add(7 * DAY).le(a.end)]
    rows = []
    for day in range(8):
        frame = rate_table(eligible, eligible.open_to_publish.between(0, day), "opening_template")
        rows.append(frame.assign(Day=day))
    return pd.concat(rows, ignore_index=True)


def activation_funnel(a):
    counts = [len(a), int(a.created_asset.sum()), int(a.funnel_published.sum()), int(a.funnel_lead.sum())]
    stages = ["Signed up", "Opened builder and created asset", "First successful publish", "Received a first lead"]
    previous = [None, *counts[:-1]]
    return pd.DataFrame({
        "Stage": stages,
        "Count": counts,
        "Rate": [rate(n, counts[0]) for n in counts],
        "From previous stage": [rate(n, p) for n, p in zip(counts, previous)],
        "Drop-off": pd.array([p - n if p is not None else None for n, p in zip(counts, previous)], dtype="Int64"),
    })


def time_to_publish(a, split):
    rows = []
    for label, group in by_group(a, split):
        published = group.ttp.dropna()
        rows.append((str(label), len(published), *published.quantile([.25, .50, .75, .90]), published.mean()))
    return pd.DataFrame(rows, columns=["Group", "Publishers", "P25 (days)", "Median (days)", "P75 (days)", "P90 (days)", "Mean (days)"])


def _active_ids(snapshot, lo, hi):
    return account_ids(snapshot.activity.loc[snapshot.activity.day.between(lo, hi)])


def complete_periods(lo, hi, *, monthly):
    for period in pd.period_range(lo, hi, freq="M" if monthly else "W-SUN"):
        start, end = period.start_time.normalize(), period.end_time.normalize()
        if lo <= start and end <= hi:
            yield start, end


def activity_trend(snapshot, a, lo, hi, *, fixed, monthly):
    activity = snapshot.activity.loc[snapshot.activity[ID].isin(a.index) & snapshot.activity.day.between(lo, hi)]
    rows = []
    for start, end in complete_periods(lo, hi, monthly=monthly):
        cohort = a if fixed else a.loc[a.launch.le(end) & a.last_event.ge(start)]
        events = activity.loc[activity.day.between(start, end) & activity[ID].isin(cohort.index)]
        active = events[ID].nunique()
        edit_only = active - events.loc[events.published_again, ID].nunique()
        rows.append((start, active, len(cohort), rate(active, len(cohort)), edit_only))
    return pd.DataFrame(rows, columns=["Period", "Count", "Eligible", "Rate", "Live edits only"])


def established_trend(snapshot, a, lo, hi, *, monthly):
    leads = snapshot.lead_days.loc[snapshot.lead_days[ID].isin(a.index)]
    assets = snapshot.additional_assets.loc[snapshot.additional_assets[ID].isin(a.index)]
    rows = []
    for start, end in complete_periods(lo, hi, monthly=monthly):
        eligible = a.index[a.launch.lt(start) & a.last_event.ge(start) & a.end.ge(end)]
        volume = leads.loc[leads.activity_date.between(start, end) & leads[ID].isin(eligible)]
        extra = assets.loc[assets.published_date.between(start, end) & assets[ID].isin(eligible)]
        coverage = volume.groupby(ID).leads.sum().ge(5).sum()
        rows.append((start, len(eligible), coverage, volume.leads.sum(), volume.visitors.sum(),
                     extra[ID].nunique(), len(extra)))
    frame = pd.DataFrame(rows, columns=["Period", "Eligible accounts", "Accounts with 5+ leads", "Leads",
                                        "Recorded visitors", "Repeat publishers", "Additional assets"])
    eligible = frame["Eligible accounts"].replace(0, float("nan"))
    visitors = frame["Recorded visitors"].replace(0, float("nan"))
    return frame.assign(**{
        "Received 5+ leads": frame["Accounts with 5+ leads"] / eligible,
        "Leads per 100 visitors": 100 * frame["Leads"] / visitors,
        "Published again": frame["Repeat publishers"] / eligible,
        "Additional assets per account": frame["Additional assets"] / eligible,
    })


def calendar_churn(snapshot, a, lo, hi, *, monthly):
    """Count churn events in the period among accounts at risk when it begins."""
    inactive_days = 56 * DAY
    activity = snapshot.activity.loc[snapshot.activity[ID].isin(a.index)
                                     & snapshot.activity.day.between(lo - inactive_days, hi)]
    product_rows, subscription_rows = [], []
    for start, end in complete_periods(lo, hi, monthly=monthly):
        live = a.loc[a.signup_date.lt(start) & a.last_event.ge(start) & a.end.ge(end)]
        canceled = int(live.subscription_churn_date.between(start, end).sum())
        subscription_rows.append((start, canceled, len(live), rate(canceled, len(live))))

        recent = activity.loc[activity.day.between(start - inactive_days, start - DAY)]
        last_action = recent.groupby(ID).day.max()
        last_action = last_action.loc[last_action.index.isin(live.index)]
        threshold = last_action + inactive_days
        first_return = activity.loc[activity.day.between(start, end)].groupby(ID).day.min().reindex(threshold.index)
        observable = live.last_event.reindex(threshold.index)
        churned = int((threshold.le(end) & threshold.le(observable) & ~first_return.le(threshold)).sum())
        product_rows.append((start, churned, len(threshold), rate(churned, len(threshold))))
    columns = ["Period", "Count", "Eligible", "Rate"]
    return pd.DataFrame(product_rows, columns=columns), pd.DataFrame(subscription_rows, columns=columns)


def quiet_accounts(snapshot, a, end):
    start = end - 27 * DAY
    base = a.index[a.launch.lt(start - 28 * DAY) & a.last_event.ge(end) & ~a.subscription_churn_date.le(end)]
    quiet = base.intersection(_active_ids(snapshot, start - 28 * DAY, start - DAY)).difference(
        _active_ids(snapshot, start, end)
    )
    volume = snapshot.lead_days.loc[snapshot.lead_days.activity_date.between(start, end)]
    v = volume.groupby(ID)[["visitors", "leads"]].sum().reindex(quiet, fill_value=0)
    low = v.leads.lt(5)
    counts = [(low & v.visitors.lt(100)).sum(), (low & v.visitors.ge(100)).sum(), (~low).sum()]
    return pd.DataFrame({
        "Group": ["Under 100 visitors and under 5 leads", "100+ visitors and under 5 leads", "5+ leads"],
        "Count": counts,
        "Eligible": len(quiet),
        "Rate": [rate(int(n), len(quiet)) for n in counts],
    })


def churn_90d(a, split):
    eligible = a.loc[a.eligible_90d].copy()
    if split == "publish_timing":
        eligible = eligible.loc[eligible.subscribed_day_14].copy()
        eligible["publish_timing"] = "Unpublished by day 14"
        eligible.loc[eligible.ttp.between(0, 7), "publish_timing"] = "Signup days 0–7"
        eligible.loc[eligible.ttp.between(8, 14), "publish_timing"] = "Signup days 8–14"
    return rate_table(eligible, eligible.canceled_90d, split)


def validation_opportunity(a, start, end):
    eligible = a.loc[a.first_error.between(start, end) & a.first_error.add(7 * DAY).le(a.end)]
    succeeded = eligible.error_to_publish.between(0, 7)
    return len(eligible), int(succeeded.sum())
