"""Reproduce the ten-slide Lead Generation case study from the four supplied CSVs.

Run: python leadgen_supporting_analysis.py --data-dir path/to/csvs
Requires Python 3.10+ and pandas. Counts represent distinct sub-accounts unless
a row explicitly measures assets or drafts. All date windows are inclusive.
The report prints to standard output. No source data is modified.
"""
import argparse
from pathlib import Path

import pandas as pd

ID = "account_id"
KEY = [ID, "product_type"]
ASSET = KEY + ["asset_id"]
DAY = pd.Timedelta(days=1)
END = pd.Timestamp("2026-05-31")
MAY = pd.Timestamp("2026-05-04")
QSTART = pd.Timestamp("2026-02-01")
QEND = pd.Timestamp("2026-04-30")


def load_data(folder, *, cutoff=END, sources=None):
    def read(stem, dates):
        source = sources[stem] if sources is not None else folder / f"{stem} - {stem}.csv"
        return pd.read_csv(source, parse_dates=dates)

    accounts = read("accounts", ["signup_date", "subscription_churn_date", "observation_end_date"])
    accounts["end"] = accounts["observation_end_date"].clip(upper=cutoff)
    accounts["last_event"] = accounts[["end", "subscription_churn_date"]].min(axis=1)
    accounts = accounts.set_index(ID)
    if not accounts.index.is_unique:
        raise ValueError("Duplicate sub-account IDs in accounts.csv")

    def in_scope(frame, col):
        joined = frame.join(accounts[["signup_date", "last_event"]], on=ID, validate="many_to_one")
        return joined.loc[joined[col].between(joined.signup_date, joined.last_event)].drop(
            columns=["signup_date", "last_event"]
        )

    builder = read("builder_events", ["event_timestamp"])
    builder["day"] = builder.event_timestamp.dt.normalize()
    builder = in_scope(builder, "day")
    builder["validation_failure"] = (builder.event_name.eq("publish_attempt_failed") &
                                     builder.error_code.eq("validation_error"))
    publishes = read("publishing", ["published_date"])
    publishes = in_scope(publishes.loc[publishes.publish_status.eq("published")], "published_date")
    publishes = publishes.sort_values(["published_date", "asset_id"]).drop_duplicates(ASSET)
    lead = in_scope(read("lead_generation", ["activity_date"]), "activity_date")
    return accounts, builder, publishes, lead


def measure(label, numerator, denominator):
    rate = 100 * numerator / denominator if denominator else float("nan")
    print(f"{label}: {numerator:,} / {denominator:,} = {rate:.2f}%")


def tag_launch(frame, first):
    return frame.merge(first[KEY + ["launch"]], on=KEY, validate="many_to_one")


def after_launch(publishes, first, builder, lead):
    pubs = tag_launch(publishes, first)
    pubs["age"] = (pubs.published_date - pubs.launch).dt.days
    more = pubs.loc[pubs.age.gt(0)].copy()
    # Match edits to the same asset and require an edit after its publication day.
    edits = builder.loc[builder.event_name.eq("asset_edited")].merge(
        publishes[ASSET + ["published_date"]], on=ASSET, validate="many_to_one"
    )
    edits = edits.loc[edits.day.gt(edits.published_date)]
    actions = pd.concat([edits[KEY + ["day"]], more[KEY + ["published_date"]].rename(
        columns={"published_date": "day"})]).drop_duplicates()
    actions = tag_launch(actions, first)
    actions["age"] = (actions.day - actions.launch).dt.days
    lead = tag_launch(lead, first)
    lead["age"] = (lead.activity_date - lead.launch).dt.days
    return more, actions, lead.loc[lead.age.ge(0)].copy()


def account_ids(frame):
    return pd.Index(frame[ID].unique())


def account_summary(accounts, builder, first, lead):
    opened = builder.loc[builder.event_name.eq("builder_opened")].groupby(KEY).day.min()
    created = builder.loc[builder.event_name.eq("asset_created")].groupby(KEY).day.min()
    stages = pd.concat([opened.rename("open"), created.rename("create"),
                        first.set_index(KEY).launch], axis=1)
    creators = stages.loc[stages.create.ge(stages.open)]
    launched = creators.loc[creators.launch.ge(creators.create)]
    lead_accounts = account_ids(lead.loc[lead.leads.gt(0)]).intersection(launched.index.get_level_values(ID))
    failed = account_ids(builder.loc[builder.validation_failure])
    unpublished = creators.index.get_level_values(ID).unique().difference(launched.index.get_level_values(ID))
    print("\nSlides 1, 2 and 10 | Onboarding through May 31 and through cancellation")
    print("Funnel:", len(accounts), creators.index.get_level_values(ID).nunique(),
          launched.index.get_level_values(ID).nunique(), len(lead_accounts))
    measure("Creators without a publish", len(unpublished), creators.index.get_level_values(ID).nunique())
    measure("Of unpublished creators, logged failure", len(unpublished.intersection(failed)), len(unpublished))
    a = accounts.join(first.set_index(ID)[["launch", "used_template"]])
    a["ttp"] = (a.launch - a.signup_date).dt.days
    measure("Published by signup day 7", a.ttp.between(0, 7).sum(), len(a))
    print("Publisher TTP (days): median", a.ttp.median(), "mean", round(a.ttp.mean(), 2))
    for label, g in a.groupby("primary_builder_device"):
        print(label, "TTP quantiles (25/50/75/90%)", g.ttp.quantile([.25,.5,.75,.9]).tolist())
        measure("  Published by day 7", g.ttp.le(7).sum(), len(g))
        measure("  No publish through cutoff", g.ttp.isna().sum(), len(g))
    for label, g in a.dropna(subset=["launch"]).groupby("used_template"):
        print("First asset template", label, "TTP quantiles", g.ttp.quantile([.25,.5,.75,.9]).tolist())
    return a


def product_metrics(a, more, actions, lead):
    print("\nSlides 3-7 | Adoption, activity and leads")
    eligible = a.index[a.launch.add(28*DAY).le(a.end)]
    repeat = account_ids(more.loc[more.age.between(1,28)])
    measure("New repeat publishing within 28 days", len(eligible.intersection(repeat)), len(eligible))
    for column in ("agency_segment", "primary_builder_device", "industry", "plan", "used_template"):
        for label, g in a.loc[eligible].groupby(column):
            measure(f"  {column} {label}", len(g.index.intersection(repeat)), len(g))
    firstlead = lead.loc[lead.leads.gt(0)].groupby(ID).activity_date.min()
    since_lead = actions.day.sub(actions[ID].map(firstlead)).dt.days
    returned = account_ids(actions.loc[since_lead.between(1,14)])
    lead_eligible = firstlead.index[firstlead.add(14*DAY).le(a.end.reindex(firstlead.index))]
    for field, values in (("Published by day 7", a.ttp.le(7)), ("Used template", a.used_template)):
        for value in (False, True):
            group = a.index[values.eq(value)]
            for label, base, result in (("repeat", eligible, repeat), ("return after lead", lead_eligible, returned)):
                group_base = base.intersection(group)
                measure(f"{field} = {value}, {label}", len(group_base.intersection(result)), len(group_base))
    for used,g in a.loc[eligible].groupby("used_template"):
        measure(f"Early publish among template={used}", g.ttp.le(7).sum(), len(g))
    may_base = a.index[a.launch.lt(MAY) & (a.subscription_churn_date.isna() | a.subscription_churn_date.ge(MAY))]
    may_pubs = more.loc[more[ID].isin(may_base) & more.published_date.between(MAY, END)]
    may_value = lead.loc[lead.activity_date.between(MAY,END)].groupby(ID)[["visitors","leads"]].sum()
    value = may_value.reindex(may_base,fill_value=0)
    measure("Existing repeat publishing in May", may_pubs[ID].nunique(), len(may_base))
    print(f"Adoption depth: {len(may_pubs):,} / {len(may_base):,} = {len(may_pubs)/len(may_base):.2f}")
    measure("Accounts with 5+ May leads", value.leads.ge(5).sum(), len(value))
    measure("Template share of May additional assets", may_pubs.used_template.sum(), len(may_pubs))

    def active_between(start, end):
        return account_ids(actions.loc[actions.day.between(start,end)])

    def active_by_age(lo, hi):
        return account_ids(actions.loc[actions.age.between(lo,hi)])

    for start in (MAY-56*DAY, MAY-28*DAY, MAY):
        finish = start+27*DAY
        quiet_base = a.index[a.launch.lt(start-28*DAY) &
                             (a.subscription_churn_date.isna() | a.subscription_churn_date.gt(finish))]
        quiet = quiet_base.intersection(active_between(start-28*DAY,start-DAY)).difference(active_between(start,finish))
        values = lead.loc[lead.activity_date.between(start,finish)].groupby(ID)[["visitors","leads"]].sum()
        v = values.reindex(quiet,fill_value=0)
        print("Quiet starting", start.date(), "low traffic / conversion / 5+ leads:",
              (v.leads.lt(5) & v.visitors.lt(100)).sum(), (v.leads.lt(5) & v.visitors.ge(100)).sum(),
              v.leads.ge(5).sum())
    mature = a.index[a.launch.add(84*DAY).le(a.end) &
                     (a.subscription_churn_date.isna() | a.subscription_churn_date.gt(a.launch+28*DAY))]
    active28 = mature.intersection(active_by_age(1,28))
    quiet_later = active28.difference(active_by_age(29,84))
    last_complete_month = (END - 84*DAY).to_period("M").start_time
    fixed = active28.intersection(a.index[a.launch.lt(last_complete_month)])
    first_period = fixed.intersection(active_between(MAY-28*DAY,MAY-DAY))
    second_period = fixed.intersection(active_between(MAY,END))
    measure("28-day calendar return",len(first_period.intersection(second_period)),len(first_period))
    measure("Inactive in weeks 5-12",len(quiet_later),len(active28))
    for lo,hi in ((29,56),(57,84)):
        measure(f"Four complete cohorts, days {lo}-{hi}",len(fixed.intersection(active_by_age(lo,hi))),len(fixed))
    for start in pd.date_range("2026-03-30",periods=9,freq="7D"):
        weekly_active = fixed.intersection(active_between(start,start+6*DAY))
        week_pubs = account_ids(more.loc[more.published_date.between(start,start+6*DAY)])
        measure(f"Calendar week {start.date()}",len(weekly_active),len(fixed))
        print("  Live-edit-only accounts:",len(weekly_active.difference(week_pubs)))
    early_value = lead.loc[lead.age.between(1,28)].groupby(ID).leads.sum()
    lead_age = (firstlead-a.launch.reindex(firstlead.index)).dt.days
    traffic = lead.loc[lead.age.between(1,28)].groupby(ID)[["visitors","leads"]].sum().reindex(mature,fill_value=0)
    for label,group in traffic.join(a.agency_segment).groupby("agency_segment"):
        print(label, "first-28-day mean visitors",round(group.visitors.mean()),
              "leads per 100 visitors",round(100*group.leads.sum()/group.visitors.sum(),2))
    for lo,hi in ((0,7),(8,14)):
        g = mature.intersection(a.index[a.ttp.between(lo,hi)])
        active_g = g.intersection(active28)
        early_lead_g = g.intersection(lead_age.index[lead_age.between(0,14)])
        for name, base, result in (("second asset",g,repeat),("post-lead return",early_lead_g,returned),
                    ("5+ leads",g,early_value.index[early_value.ge(5)]),
                    ("weeks 9-12 activity",active_g,active_by_age(57,84)),
                    ("observed churn through May 31",active_g,a.index[a.subscription_churn_date.notna()])):
            measure(f"Publish days {lo}-{hi}, {name}",len(base.intersection(result)),len(base))


def churn(a):
    print("\nSlides 6, 7 and 10 | Subscription cancellation")
    measure("All recorded cancellations", a.subscription_churn_date.notna().sum(), len(a))
    elapsed = (a.subscription_churn_date-a.signup_date).dt.days
    followup = (a.end-a.signup_date).dt.days
    mature = a.loc[followup.ge(90) & ~elapsed.le(14)].copy()
    mature["cancel90"] = elapsed.reindex(mature.index).between(15,90)
    mature["timing"] = pd.cut(mature.ttp.fillna(float("inf")),[-1,7,14,float("inf")],
                              labels=["signup days 0-7","signup days 8-14","unpublished at day 14"])
    for column in ("timing","agency_segment"):
        for label,g in mature.groupby(column,observed=True):
            measure(f"90-day cancellation, {label}",g.cancel90.sum(),len(g))
    for label,g in a.dropna(subset=["launch"]).groupby("used_template"):
        measure(f"Post-launch cancellation through cutoff, template={label}",g.subscription_churn_date.notna().sum(),len(g))


def opportunities(a, first, builder):
    print("\nSlides 2, 8 and 9 | First opportunities, measured from the event date")
    def first_event(mask):
        return (builder.loc[mask]
                .sort_values(["event_timestamp","product_type"]).drop_duplicates(ID)
                .merge(first[KEY+["launch"]],how="left",on=KEY,validate="one_to_one")
                .join(a[["end","monthly_recurring_revenue_usd","primary_builder_device","ttp"]],on=ID))

    opened = first_event(builder.event_name.eq("builder_opened"))
    selected = builder.loc[builder.event_name.eq("template_selected"),KEY+["day"]].drop_duplicates()
    opened = opened.merge(selected.assign(selected=True),how="left",on=KEY+["day"])
    opened["selected"] = opened.selected.eq(True)
    eligible = opened.loc[opened.day.add(7*DAY).le(opened.end)].copy()
    eligible["speed"] = (eligible.launch-eligible.day).dt.days
    for label,g in eligible.groupby("selected"):
        measure(f"Full opening-day template={label}, 7-day publish", g.speed.between(0,7).sum(),len(g))
    print("Mean builder-to-publish days for day-14 publishers:",
          eligible.loc[eligible.ttp.le(14)].groupby("primary_builder_device").speed.mean().round(1).to_dict())

    def quarter(frame):
        return frame.loc[frame.day.between(QSTART,QEND) & frame.day.add(7*DAY).le(frame.end)].copy()

    q = quarter(eligible)
    failures = quarter(first_event(builder.validation_failure))
    for label,g in q.groupby("selected"):
        measure(f"Quarter opening-day template={label}",g.speed.between(0,7).sum(),len(g))
    for label,g in q.groupby("primary_builder_device"):
        measure(f"Quarter device={label}",g.speed.between(0,7).sum(),len(g))
    for label,g,scenarios in (
        ("Validation (all-rule ceiling)",failures,(1.0,)),
        ("Starter templates",q.loc[~q.selected],(.05,.10)),
        ("Mobile publishing",q.loc[q.primary_builder_device.eq("mobile")],(.05,.10)),
    ):
        elapsed = (g.launch-g.day).dt.days
        missed = g.loc[~elapsed.between(0,7)]
        mean_mrr = missed.monthly_recurring_revenue_usd.mean()
        print(f"{label}: {len(g)} exposed; {len(missed)} unpublished at seven days; mean MRR \u0024{mean_mrr:.3f}")
        for fraction in scenarios:
            extra = len(missed)*fraction
            arr = extra*.01*mean_mrr*12
            print(f"  Recovery {fraction:.0%}: {extra:.2f} extra accounts; annualized retained ARR per 1pp lower churn \u0024{arr:,.2f}")
    print("Pilot sizing requires unique eligible sub-accounts for the selected rule and measured incremental account-level lift.")


def export_sheet_evidence(path, a, builder, first, more, actions, lead):
    """Export one auditable observation per sub-account for the workbook formulas."""
    import csv

    epoch = pd.Timestamp("1899-12-30")
    fields = [
        "primary_builder_device", "agency_segment", "plan", "industry",
        "monthly_recurring_revenue_usd", "signup_date", "end",
        "subscription_churn_date", "launch", "ttp", "used_template",
    ]
    ev = a[fields].copy()
    ev.columns = [
        "Device", "Agency_Size", "Account_Plan", "Industry_Group", "Monthly_Revenue",
        "Signup_Date", "Observation_End", "Churn_Date", "First_Publish_Date",
        "TTP_Days", "Template_Used",
    ]

    def mark(label, ids):
        ev[label] = ev.index.isin(ids).astype(int)

    def field(label, series, missing=0):
        ev[label] = series.reindex(ev.index).fillna(missing).astype(int)

    def action_age(lo, hi):
        return account_ids(actions.loc[actions.age.between(lo, hi)])

    def action_dates(lo, hi):
        return account_ids(actions.loc[actions.day.between(lo, hi)])

    def ids_from_index(frame):
        return frame.index.get_level_values(ID).unique()

    def event_rows(mask):
        return (builder.loc[mask].sort_values(["event_timestamp", "product_type"])
                .drop_duplicates(ID).merge(first[KEY + ["launch"]], on=KEY, how="left")
                .set_index(ID))

    opened = builder.loc[builder.event_name.eq("builder_opened")].groupby(KEY).day.min()
    created = builder.loc[builder.event_name.eq("asset_created")].groupby(KEY).day.min()
    stages = pd.concat([opened.rename("open"), created.rename("create"),
                        first.set_index(KEY).launch], axis=1)
    creators = stages.loc[stages.create.ge(stages.open)]
    in_funnel = creators.loc[creators.launch.ge(creators.create)]
    mark("Created_Asset", ids_from_index(creators))
    mark("Published_In_Funnel", ids_from_index(in_funnel))
    mark("Funnel_First_Lead", ids_from_index(in_funnel).intersection(account_ids(lead.loc[lead.leads.gt(0)])))
    mark("Validation_Failure", account_ids(builder.loc[builder.validation_failure]))

    first_open = event_rows(builder.event_name.eq("builder_opened"))
    selections = builder.loc[builder.event_name.eq("template_selected"), KEY + ["day"]].drop_duplicates()
    selected = first_open.reset_index().merge(selections, on=KEY + ["day"])
    mark("Selected_At_Open", account_ids(selected))
    field("Builder_Publish_Days", (first_open.launch - first_open.day).dt.days, -999)
    open_eligible = first_open.day.add(7 * DAY).le(a.end.reindex(first_open.index))
    mark("Builder_7d_Eligible", first_open.index[open_eligible])
    mark("Quarter_Builder", first_open.index[open_eligible & first_open.day.between(QSTART, QEND)])
    failures = event_rows(builder.validation_failure)
    error_eligible = failures.day.add(7 * DAY).le(a.end.reindex(failures.index))
    mark("Quarter_Validation", failures.index[error_eligible & failures.day.between(QSTART, QEND)])
    field("Error_Publish_Days", (failures.launch - failures.day).dt.days, -999)

    first_lead = lead.loc[lead.leads.gt(0)].groupby(ID).activity_date.min()
    lead_age = (first_lead - a.launch.reindex(first_lead.index)).dt.days
    field("First_Lead_Age", lead_age, -999)
    days_after_lead = (actions.day - actions[ID].map(first_lead)).dt.days
    mark("Returned_After_Lead", account_ids(actions.loc[days_after_lead.between(1, 14)]))
    mark("Lead_14d_Eligible", first_lead.index[first_lead.add(14 * DAY).le(a.end.reindex(first_lead.index))])
    mark("New_28d_Eligible", a.index[a.launch.add(28 * DAY).le(a.end)])
    mark("Second_Asset_28d", account_ids(more.loc[more.age.between(1, 28)]))

    may_base = a.index[a.launch.lt(MAY) &
                       (a.subscription_churn_date.isna() | a.subscription_churn_date.ge(MAY))]
    mark("May_Eligible", may_base)
    may_assets = more.loc[more[ID].isin(may_base) & more.published_date.between(MAY, END)]
    field("May_Additional_Assets", may_assets.groupby(ID).size())
    field("May_Template_Assets", may_assets.groupby(ID).used_template.sum())
    may_leads = lead.loc[lead.activity_date.between(MAY, END)].groupby(ID).leads.sum()
    field("May_Leads", may_leads)

    for month, start in (("Mar", MAY - 56 * DAY), ("Apr", MAY - 28 * DAY), ("May", MAY)):
        finish = start + 27 * DAY
        base = a.index[a.launch.lt(start - 28 * DAY) &
                       (a.subscription_churn_date.isna() | a.subscription_churn_date.gt(finish))]
        quiet = base.intersection(action_dates(start - 28 * DAY, start - DAY)).difference(
            action_dates(start, finish)
        )
        value = lead.loc[lead.activity_date.between(start, finish)].groupby(ID)[["visitors", "leads"]].sum()
        v = value.reindex(quiet, fill_value=0)
        category = 1 + v.visitors.ge(100).astype(int)
        field("Quiet_" + month, category.where(v.leads.lt(5), 3))

    mature = a.index[a.launch.add(84 * DAY).le(a.end) &
                     (a.subscription_churn_date.isna() | a.subscription_churn_date.gt(a.launch + 28 * DAY))]
    active28 = mature.intersection(action_age(1, 28))
    complete_month = (END - 84 * DAY).to_period("M").start_time
    fixed = active28.intersection(a.index[a.launch.lt(complete_month)])
    mark("Post_Publish_84d_Eligible", mature)
    mark("Active_Days_1_28", active28)
    mark("Active_Weeks_5_8", action_age(29, 56))
    mark("Active_Weeks_9_12", action_age(57, 84))
    mark("Fixed_Panel", fixed)
    mark("Active_Apr_Window", action_dates(MAY - 28 * DAY, MAY - DAY))
    mark("Active_May_Window", action_dates(MAY, END))
    for i, start in enumerate(pd.date_range("2026-03-30", periods=9, freq="7D"), 1):
        finish = start + 6 * DAY
        active = action_dates(start, finish)
        published = account_ids(more.loc[more.published_date.between(start, finish)])
        field(f"Week_{i}", pd.Series(1 + active.isin(published).astype(int), index=active))

    first_28 = lead.loc[lead.age.between(1, 28)].groupby(ID)[["visitors", "leads"]].sum()
    field("Post_Launch_28d_Visitors", first_28.visitors)
    field("Post_Launch_28d_Leads", first_28.leads)
    elapsed = (a.subscription_churn_date - a.signup_date).dt.days
    mark("Eligible_90d", a.index[(a.end - a.signup_date).dt.days.ge(90) & ~elapsed.le(14)])
    mark("Cancel_90d", a.index[elapsed.between(15, 90)])
    mark("Canceled", a.index[a.subscription_churn_date.notna()])

    for label in ["Signup_Date", "Observation_End", "Churn_Date", "First_Publish_Date"]:
        ev[label] = (ev[label] - epoch).dt.days.fillna(0).astype(int)
    ev["TTP_Days"] = ev.TTP_Days.fillna(-999).astype(int)
    ev["Template_Used"] = ev.Template_Used.fillna(-1).astype(int)
    ev.index.name = "Acct_ID"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["Acct_ID", *ev.columns])
        writer.writerows(ev.itertuples())
    print(f"Exported {len(ev):,} accounts and {len(ev.columns) + 1} evidence columns.")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--sheet-evidence', type=Path, help='Write a TSV with account-level evidence for spreadsheet formulas.')
    args = parser.parse_args()
    accounts,builder,publishes,raw_lead = load_data(args.data_dir)
    first = publishes.drop_duplicates(KEY).rename(columns={"published_date":"launch"})
    if first[ID].duplicated().any():
        raise ValueError("This case has one published product per sub-account. Define cross-product attribution before reuse.")
    more,actions,lead = after_launch(publishes,first,builder,raw_lead)
    a = account_summary(accounts,builder,first,lead)
    if args.sheet_evidence is not None:
        export_sheet_evidence(args.sheet_evidence, a, builder, first, more, actions, lead)
    product_metrics(a,more,actions,lead)
    churn(a)
    opportunities(a,first,builder)


if __name__ == "__main__":
    main()

