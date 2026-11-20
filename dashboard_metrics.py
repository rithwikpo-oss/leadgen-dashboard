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
    a["canceled_90d"] = age.between(15, 90) & a.canceled
    a["eligible_90d"] = a.signup_date.add(90 * DAY).le(a.end) & ~age.le(14)

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
    a["template_selected"] = a.index.isin(same_day[ID])
    opened = builder.loc[builder.event_name.eq("builder_opened")].groupby(KEY).day.min()
    created = builder.loc[builder.event_name.eq("asset_created")].groupby(KEY).day.min()
    milestones = pd.concat([opened.rename("open"), created.rename("create"), same_product_first], axis=1)
    creators = milestones.loc[milestones.create.ge(milestones.open)]
    funnel_published = creators.loc[creators.launch.ge(creators.create)].index
    lead_pairs = leads.loc[leads.leads.gt(0)].set_index(KEY).index
    a["created_asset"] = a.index.isin(creators.index.get_level_values(ID))
    a["funnel_published"] = a.index.isin(funnel_published.get_level_values(ID))
    a["funnel_lead"] = a.index.isin(funnel_published.intersection(lead_pairs).get_level_values(ID))

    a["repeat_28d"] = a.index.isin(account_ids(more.loc[more.age.between(1, 28)]))
    initial = account_ids(actions.loc[actions.age.between(1, 28)])
    subscribed_through_28 = a.subscription_churn_date.isna() | a.subscription_churn_date.gt(a.launch + 28 * DAY)
    a["initially_active"] = a.index.isin(initial) & a.launch.add(84 * DAY).le(a.end) & subscribed_through_28
    a["active_weeks_5_8"] = a.index.isin(account_ids(actions.loc[actions.age.between(29, 56)]))
    a["active_weeks_9_12"] = a.index.isin(account_ids(actions.loc[actions.age.between(57, 84)]))

    # Preserve the separate edit-only series used to investigate the April dip.
    keys = more.rename(columns={"published_date": "day"})[KEY + ["day"]].drop_duplicates()
    activity = actions.merge(keys.assign(published_again=True), on=KEY + ["day"], how="left")
    activity["published_again"] = activity.published_again.eq(True)
    return Snapshot(a, activity, more, leads, cutoff)


def select_accounts(a, filters):
    for name, choices in filters.items():
        if choices:
            a = a.loc[a[DIMENSIONS[name]].isin(choices)]
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


def activation_funnel(a):
    counts = [len(a), int(a.created_asset.sum()), int(a.funnel_published.sum()), int(a.funnel_lead.sum())]
    stages = ["Signed up", "Opened builder and created asset", "First successful publish", "Received a first lead"]
    return pd.DataFrame({"Stage": stages, "Count": counts, "Rate": [rate(n, len(a)) for n in counts]})


def time_to_publish(a, split):
    rows = []
    for label, group in by_group(a, split):
        published = group.ttp.dropna()
        rows.append((str(label), len(published), *published.quantile([.25, .50, .75, .90]), published.mean()))
    return pd.DataFrame(rows, columns=["Group", "Publishers", "P25 (days)", "Median (days)", "P75 (days)", "P90 (days)", "Mean (days)"])


def _active_ids(snapshot, lo, hi):
    return account_ids(snapshot.activity.loc[snapshot.activity.day.between(lo, hi)])


def activity_trend(snapshot, a, lo, hi, *, fixed, monthly):
    freq = "M" if monthly else "W-SUN"
    periods = pd.period_range(lo, hi, freq=freq)
    activity = snapshot.activity.loc[snapshot.activity[ID].isin(a.index) & snapshot.activity.day.between(lo, hi)].copy()
    activity["period"] = activity.day.dt.to_period(freq)
    by_period = {p: rows for p, rows in activity.groupby("period")}
    empty = activity.iloc[:0]
    rows = []
    for p in periods:
        start, end = p.start_time.normalize(), p.end_time.normalize()
        if start < lo or end > hi:
            continue
        cohort = a if fixed else a.loc[a.launch.le(end) & a.last_event.ge(start)]
        events = by_period.get(p, empty)
        events = events.loc[events[ID].isin(cohort.index)]
        active = events[ID].nunique()
        edit_only = active - events.loc[events.published_again, ID].nunique()
        rows.append((start, active, len(cohort), rate(active, len(cohort)), edit_only))
    return pd.DataFrame(rows, columns=["Period", "Count", "Eligible", "Rate", "Live edits only"])


def calendar_return(snapshot, a, end):
    previous = _active_ids(snapshot, end - 55 * DAY, end - 28 * DAY).intersection(a.index)
    current = _active_ids(snapshot, end - 27 * DAY, end).intersection(a.index)
    return len(previous.intersection(current)), len(previous)


def established_28d(snapshot, a, end):
    start = end - 27 * DAY
    active_sub = a.subscription_churn_date.isna() | a.subscription_churn_date.ge(start)
    eligible = a.index[a.launch.lt(start) & active_sub & a.end.ge(end)]
    lead = snapshot.lead_days.loc[snapshot.lead_days.activity_date.between(start, end)]
    volume = lead.groupby(ID)[["leads", "visitors"]].sum().reindex(eligible, fill_value=0)
    extra = snapshot.additional_assets.loc[
        snapshot.additional_assets[ID].isin(eligible) & snapshot.additional_assets.published_date.between(start, end)
    ]
    return volume, extra


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


def launch_cohorts(a, full_month_only=True):
    a = a.loc[a.initially_active].copy()
    if full_month_only:
        end = a.launch.dt.to_period("M").dt.end_time.dt.normalize()
        a = a.loc[end.add(84 * DAY).le(a.end)]
    a["month"] = a.launch.dt.strftime("%Y-%m")
    rows = []
    for label, g in by_group(a, "month"):
        for col, window in [("active_weeks_5_8", "Weeks 5-8"), ("active_weeks_9_12", "Weeks 9-12")]:
            n = int(g[col].sum())
            rows.append((label, window, n, len(g), rate(n, len(g))))
    return pd.DataFrame(rows, columns=["First publish month", "Window", "Count", "Eligible", "Rate"])


def churn_90d(a, split):
    eligible = a.loc[a.eligible_90d].copy()
    eligible["publish_timing"] = "Unpublished by day 14"
    eligible.loc[eligible.ttp.between(0, 7), "publish_timing"] = "Signup days 0-7"
    eligible.loc[eligible.ttp.between(8, 14), "publish_timing"] = "Signup days 8-14"
    return rate_table(eligible, eligible.canceled_90d, split)


def validation_opportunity(a, start, end):
    eligible = a.loc[a.first_error.between(start, end) & a.first_error.add(7 * DAY).le(a.end)]
    succeeded = eligible.error_to_publish.between(0, 7)
    return len(eligible), int(succeeded.sum())
