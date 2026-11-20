"""Streamlit entrypoint. Run with streamlit run app.py."""

from contextlib import nullcontext
from hashlib import sha256
from html import escape
from math import ceil, floor
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import dashboard_metrics as dm

ROOT = Path(__file__).resolve().parent
METRICS_SOURCES = (ROOT / "dashboard_metrics.py", ROOT / "leadgen_supporting_analysis.py")
COLORS = ["#1B8174", "#355D80", "#DB8254", "#8F77A4", "#CFAB5C"]
DATE_KEYS = ("health_dates", "activation_dates", "retention_dates", "quiet_end",
             "validation_dates", "churn_diagnostic_dates")
st.set_page_config(page_title="Lead Generation Dashboard", layout="wide")


@st.cache_data(show_spinner="Reading the four CSV exports…")
def load_bundled_snapshot(metrics_revision):
    # The revision also invalidates cached account data when its schema changes.
    files = {}
    for key, name in dm.DEFAULT_EXPORT_NAMES.items():
        csv = ROOT / "data" / name
        files[key] = csv if csv.is_file() else csv.with_suffix(".csv.gz")
    return dm.snapshot_from_exports(files)


def load_uploaded_snapshot(files, metrics_revision):
    signature = (metrics_revision, tuple(sha256(content).digest() for content in files.values()))
    cached = st.session_state.get("uploaded_snapshot")
    if cached is None or cached[0] != signature:
        cached = (signature, dm.snapshot_from_exports(files))
        st.session_state["uploaded_snapshot"] = cached
        reset_dates()
    return cached[1]


def reset_dates():
    for key in DATE_KEYS:
        st.session_state.pop(key, None)
        st.session_state.pop(f"saved_{key}", None)


def layout():
    st.markdown(
        """<style>
        .block-container {padding-top: 2.3rem; padding-bottom: 3rem; max-width: 1240px;}
        [data-testid=stMetric] {background: #f3f8f7; border-radius: .70rem; border-color: #dcebe7;}
        [data-testid=stMetricLabel] {font-weight: 550;}
        [data-testid=stSidebar] {border-right: 1px solid #e2ebe8;}
        .lg-funnel {list-style: none; max-width: 48rem; padding: 0 !important; margin: .75rem auto 1rem !important; display: grid; gap: 4px;}
        .lg-funnel li {position: relative; isolation: isolate; min-height: 66px; margin: 0; padding: .4rem .5rem; display: grid; align-content: center; text-align: center; color: #123331;}
        .lg-funnel-shape {position: absolute; inset: 0; z-index: -1;}
        .lg-funnel-stage {font-size: .83rem; font-weight: 660; line-height: 1.22;}
        .lg-funnel-primary {font-size: 1.34rem; font-weight: 720; font-variant-numeric: tabular-nums;}
        .lg-funnel-secondary {font-size: .79rem; color: #315859; margin-left: .4rem; font-variant-numeric: tabular-nums;}
        </style>""",
        unsafe_allow_html=True,
    )


def date_window(label, first, last, *, key, default=None):
    saved_key = f"saved_{key}"
    selected = st.date_input(label, value=st.session_state.get(saved_key, default or (first.date(), last.date())),
                             min_value=first.date(), max_value=last.date(), key=key)
    if not isinstance(selected, (tuple, list)) or len(selected) != 2:
        st.info("Select both the start and the end date.")
        st.stop()
    st.session_state[saved_key] = selected
    return tuple(pd.Timestamp(d) for d in selected)


def pct(n, d):
    return f"{n / d:.1%}" if d else "—"


def stat(container, name, value, detail):
    with container:
        st.metric(name, value, border=True)
        st.caption(detail)


def show_table(frame, key, *, visible=False, pct_columns=()):
    table = frame.copy()
    if "Period" in table:
        table["Period"] = pd.to_datetime(table.Period).dt.strftime("%Y-%m-%d")
    display = table.copy()
    for column in ("Rate", "From previous stage", "Activation rate", "Unpublished creator rate", *pct_columns):
        if column in display:
            display[column] = display[column].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    with (nullcontext() if visible else st.expander("View counts and download table")):
        st.dataframe(display, hide_index=True, width="stretch")
        st.download_button("Download CSV", table.to_csv(index=False), file_name=f"{key}.csv",
                           mime="text/csv", key=f"download_{key}")


def funnel_markup(frame, value):
    """Draw step widths proportional to their share of the signup cohort."""
    colors = ["#9CD6CA", "#ADE0D5", "#C2E8E0", "#D8EFEA"]
    widths = frame.Rate.tolist()
    segments = []
    for i, row in enumerate(frame.itertuples(index=False)):
        top = 50 * (1 - widths[i])
        bottom = 50 * (1 - widths[min(i + 1, len(widths) - 1)])
        polygon = f"polygon({top:.3f}% 0,{100-top:.3f}% 0,{100-bottom:.3f}% 100%,{bottom:.3f}% 100%)"
        primary, secondary = f"{row.Rate:.1%}", f"{row.Count:,}"
        if value == "Count":
            primary, secondary = secondary, primary
        segments.append(
            f'<li><div class="lg-funnel-shape" aria-hidden="true" '
            f'style="background:{colors[i % len(colors)]};clip-path:{polygon}"></div>'
            f'<span class="lg-funnel-stage">{escape(row.Stage)}</span>'
            f'<div><span class="lg-funnel-primary">{primary}</span>'
            f'<span class="lg-funnel-secondary">{secondary}</span></div></li>'
        )
    return '<ol class="lg-funnel" aria-label="Onboarding funnel">' + "".join(segments) + "</ol>"


def show_funnel(frame, value):
    if frame.empty or not frame.Rate.notna().any():
        st.info("No signups match these filters and dates.")
        return
    st.markdown(funnel_markup(frame, value), unsafe_allow_html=True)
    dropoffs = frame["Drop-off"].dropna()
    if not dropoffs.empty and dropoffs.max() > 0:
        largest = frame.loc[dropoffs.idxmax()]
        st.caption(f"Largest loss by count | {largest['Drop-off']:,} accounts before {largest.Stage.lower()} "
                   f"({1 - largest['From previous stage']:.1%} of the previous stage).")
    show_table(frame, "activation_funnel")


def plot_bars(frame, value, *, key, order=None):
    if frame.empty or not frame[value].notna().any():
        st.info("No eligible accounts match the selected filters.")
        return
    is_rate = value == "Rate"
    base = alt.Chart(frame).encode(
        x=alt.X(f"{value}:Q", axis=alt.Axis(format=".0%" if is_rate else ",~s"),
                scale=alt.Scale(range=[0, alt.ExprRef(expr="width - 58")])),
        y=alt.Y("Group:N", title=None, sort=order if order is not None else "-x",
                axis=alt.Axis(labelLimit=220)),
        tooltip=["Group:N", alt.Tooltip("Rate:Q", format=".1%"),
                 alt.Tooltip("Count:Q", format=","), alt.Tooltip("Eligible:Q", format=",")],
    )
    bars = base.mark_bar(color=COLORS[0], cornerRadiusEnd=4)
    labels = base.mark_text(align="left", dx=5, color="#163740").encode(
        text=alt.Text(f"{value}:Q", format=".1%" if is_rate else ","))
    chart = (bars + labels).properties(height=max(120, 42 * len(frame))).configure_view(stroke=None)
    st.altair_chart(chart, width="stretch", key=key)
    show_table(frame, key)


def activity_dates(snapshot, key):
    cutoff = snapshot.cutoff
    earliest = snapshot.accounts.signup_date.min()
    first_month = (cutoff.to_period("M") - 5).start_time
    default = (max(earliest, first_month).date(), cutoff.date())
    dates, group = st.columns((2, 1))
    with dates:
        start, end = date_window("Calendar dates", earliest, cutoff, key=f"{key}_dates", default=default)
    with group:
        saved_key = f"saved_{key}_cadence"
        choices = ["Week", "Month"]
        cadence = st.radio("Group by", choices, index=choices.index(st.session_state.get(saved_key, "Month")),
                           horizontal=True, key=f"{key}_cadence")
        st.session_state[saved_key] = cadence
    return start, end, cadence == "Month"


def period_chart(frame, monthly, metric, *, key, fmt, details, height=175, color=COLORS[0],
                 compact=False, y_domain=None):
    if frame.empty:
        st.info("Select a complete calendar week or month to see this trend.")
        return
    if not frame[metric].notna().any():
        st.info("No eligible accounts match this selection.")
        return
    dates = pd.to_datetime(frame.Period)
    labels = dates.dt.strftime("%b %Y" if monthly else "%d %b %Y")
    data = frame.assign(**{"Calendar period": labels})
    axis = alt.Axis(labelAngle=0, labelOverlap="greedy", labelPadding=7, labelFlush=True)
    if compact and monthly:
        axis.labelExpr = "slice(datum.label, 0, 3) + \" '\" + slice(datum.label, -2)"
    scale = alt.Scale(domain=y_domain, nice=False, zero=False) if y_domain else alt.Scale(zero=True)
    chart = alt.Chart(data).mark_line(
        point=alt.OverlayMarkDef(size=43, filled=True, color=color), strokeWidth=2.3, color=color
    ).encode(
        x=alt.X("Calendar period:O", sort=labels.tolist(), title=None, axis=axis),
        y=alt.Y(f"{metric}:Q", title=None, axis=alt.Axis(format=fmt, tickCount=4), scale=scale),
        tooltip=[alt.Tooltip("Calendar period:O", title="Month" if monthly else "Week starting"),
                 alt.Tooltip(f"{metric}:Q", format=fmt),
                 *[alt.Tooltip(f"{name}:Q", format=",") for name in details if name != metric]],
    ).properties(height=height).configure_view(stroke=None)
    st.altair_chart(chart, width="stretch", key=key)


def health(s, a, _value):
    start, end, monthly = activity_dates(s, "health")
    trend = dm.established_trend(s, a, start, end, monthly=monthly)
    st.subheader("Lead generation and repeat publishing")
    period = "month" if monthly else "week"
    if trend.empty:
        st.info("Select a complete calendar week or month to see these trends.")
        return
    latest = trend.iloc[-1]
    previous = trend.iloc[-2] if len(trend) > 1 else None
    latest_label = latest.Period.strftime("%B %Y" if monthly else "%d %b %Y")
    st.caption(f"Latest complete {period} shown is {latest_label}. Each point uses accounts that were already live and subscribed when the {period} began.")
    metrics = [
        ("Received 5+ leads", ".1%", "Accounts with 5+ leads", "Eligible accounts", COLORS[0]),
        ("Leads per 100 visitors", ".1f", "Leads", "Recorded visitors", COLORS[1]),
        ("Published again", ".1%", "Repeat publishers", "Eligible accounts", COLORS[0]),
        ("Additional assets per account", ".2f", "Additional assets", "Eligible accounts", COLORS[1]),
    ]
    for row in (metrics[:2], metrics[2:]):
        for column, (name, fmt, count, eligible, color) in zip(st.columns(2), row):
            with column.container(border=True):
                amount = format(latest[name], fmt) if pd.notna(latest[name]) else "-"
                delta = None
                if previous is not None and pd.notna(latest[name]) and pd.notna(previous[name]):
                    change = latest[name] - previous[name]
                    difference = f"{100 * change:+.1f} pp" if fmt.endswith("%") else format(change, f"+{fmt}")
                    previous_label = previous.Period.strftime("%b %Y" if monthly else "week of %d %b")
                    delta = f"{difference} vs {previous_label}"
                st.metric(name, amount, delta=delta, delta_color="off")
                period_chart(trend, monthly, name, key=f"health_{count}", fmt=fmt,
                             details=(count, eligible), height=132, color=color)
    show_table(trend, "product_health_trends", pct_columns=("Received 5+ leads", "Published again"))


def publish_percentiles(a, split_name):
    publishers = a.loc[a.ttp.notna()].copy()
    if publishers.empty:
        st.info("No accounts in this selection have a successful publish.")
        return
    split = dm.ONBOARDING_DIMENSIONS[split_name]
    order = dm.TEMPLATE_GROUPS if split_name == dm.OPENING_TEMPLATE else "ascending"
    if split_name == "Builder device":
        publishers[split] = publishers[split].fillna("Unknown").str.title()
    st.caption(f"Grouped by {split_name.lower()}.")

    table = dm.time_to_publish(publishers, split).round(1)
    upper = max(5, ceil(table["P90 (days)"].max() / 5) * 5)
    scale = alt.Scale(domain=[0, upper], nice=False)
    axis = alt.Axis(tickMinStep=1, tickCount=6, grid=True, domain=False)
    base = alt.Chart(table).encode(
        y=alt.Y("Group:N", sort=order, title=None,
                axis=alt.Axis(ticks=False, domain=False, labelPadding=10, labelLimit=175)),
        tooltip=["Group:N", alt.Tooltip("Publishers:Q", format=","),
                 *[alt.Tooltip(f"{metric}:Q", format=".1f")
                   for metric in ("P25 (days)", "Median (days)", "P75 (days)", "P90 (days)")]],
    )
    middle = base.mark_rule(strokeWidth=11, strokeCap="round", color=COLORS[0]).encode(
        x=alt.X("P25 (days):Q", scale=scale, axis=axis, title="Days since signup"),
        x2="P75 (days):Q",
    )
    tail = base.mark_rule(strokeWidth=3, strokeCap="round", color="#AAD8D0").encode(
        x=alt.X("P75 (days):Q", scale=scale, axis=axis, title="Days since signup"), x2="P90 (days):Q",
    )
    median = base.mark_tick(orient="vertical", size=19, thickness=2.5, color="#163740").encode(
        x=alt.X("Median (days):Q", scale=scale, axis=axis, title="Days since signup"),
    )
    chart = (tail + middle + median).properties(height=max(156, 38 * len(table) + 48)).configure_view(stroke=None)
    st.altair_chart(chart, width="stretch", key="publish_percentiles")
    st.caption("The thick bar spans P25 to P75, the thin line extends to P90, and the dark tick marks the median.")
    show_table(table[["Group", "Publishers", "Median (days)", "P25 (days)",
                      "P75 (days)", "P90 (days)", "Mean (days)"]], "time_to_first_publish")


def show_onboarding_trends(a, value):
    st.subheader("Onboarding trends by signup cohort")
    st.caption(
        "Activation and creator drop-off use signup day 7. Median publish time uses first publishes within "
        f"{dm.PUBLISH_TIME_FOLLOWUP_DAYS} days. The signup date filter can include part of a week or month."
    )
    cadence = st.radio("Cohort size", ["Week", "Month"], index=1, horizontal=True, key="activation_cohort_size")
    monthly = cadence == "Month"
    trend = dm.onboarding_trends(a, monthly=monthly)
    count = value == "Count"
    rate_format = "," if count else ".1%"
    cards = [
        ("7-day activation", "Activated by day 7" if count else "Activation rate", rate_format,
         ("Activated by day 7", "Signups with 7 days"), COLORS[0]),
        ("Median time to publish", "Median days", ".1f",
         ("Publishers in window", "Signups with follow-up"), COLORS[1]),
        ("Creators still unpublished", "Unpublished creators" if count else "Unpublished creator rate", rate_format,
         ("Unpublished creators", "Creators by day 7"), COLORS[2]),
    ]
    for column, (title, metric, fmt, fields, color) in zip(st.columns(3), cards):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            period_chart(trend, monthly, metric, key=f"onboarding_{metric}", fmt=fmt,
                         details=fields, height=140, color=color, compact=True)
    show_table(trend, "onboarding_trends")


def onboarding(s, a, value):
    min_date = s.accounts.signup_date.min()
    max_date = s.accounts.signup_date.max()
    start, end = date_window("Signup dates", min_date, max_date, key="activation_dates")
    a = a.loc[a.signup_date.between(start, end)]
    if a.empty:
        st.info("No accounts match the selected signup dates and filters.")
        return
    eligible = a.loc[a.signup_date.add(7 * dm.DAY).le(a.end)]
    summary = dm.onboarding_summary(a)
    activated = summary["Activated by day 7"]
    median = summary["Median days"]
    unpublished = summary["Unpublished creators"]
    creators = summary["Creators by day 7"]
    c = st.columns(3)
    stat(c[0], "7-day activation", pct(activated, len(eligible)), f"{activated:,} of {len(eligible):,} signups with 7 days of follow-up")
    stat(c[1], "Median time to publish", f"{median:g} days" if pd.notna(median) else "-",
         f"{summary['Publishers in window']:,} accounts published by signup day {dm.PUBLISH_TIME_FOLLOWUP_DAYS}")
    stat(c[2], "Creators unpublished on day 7", f"{unpublished:,}", f"{pct(unpublished, creators)} of {creators:,} day-7 creators")

    st.subheader("Onboarding funnel")
    st.caption("Share of selected signups reaching each stage by the data cutoff.")
    show_funnel(dm.activation_funnel(a), value)

    split_name = st.selectbox("Compare by", list(dm.ONBOARDING_DIMENSIONS), index=1, key="activation_split",
                              help="Groups both the 7-day activation and time-to-first-publish charts.")
    st.subheader("7-day activation breakdown")
    table = dm.rate_table(eligible, eligible.ttp.between(0, 7), dm.ONBOARDING_DIMENSIONS[split_name])
    plot_bars(table, value, key="activation_by_group",
              order=dm.TEMPLATE_GROUPS if split_name == dm.OPENING_TEMPLATE else None)

    st.subheader("Time to first publish")
    st.caption("Days from signup to first successful publish, among accounts that published. Use 7-day activation above to see how many got live.")
    publish_percentiles(a, split_name)
    show_onboarding_trends(a, value)


def focused_rate_bounds(rates):
    # Leave room around the data and keep a minimum 10-percentage-point range.
    lower = max(0, 5 * floor((100 * rates.min() - 1.5) / 5))
    upper = min(100, 5 * ceil((100 * rates.max() + 1.5) / 5))
    if upper - lower < 10:
        lower = max(0, upper - 10)
        upper = lower + 10
    return lower, upper


def show_active_return(returned, monthly):
    available = returned.dropna(subset=["Rate"])
    if available.empty:
        st.info("No 28-day return windows have previously active accounts for these filters and dates.")
        return

    lower, upper = focused_rate_bounds(available.Rate)
    latest = available.iloc[-1]
    delta = None
    if len(available) > 1:
        previous = available.iloc[-2]
        previous_label = previous.Period.strftime("%b %Y" if monthly else "week of %d %b")
        delta = f"{100 * (latest.Rate - previous.Rate):+.1f} pp vs {previous_label}"

    summary, trend = st.columns((1, 3))
    with summary:
        st.metric(f"Ending {latest['Window ends']:%d %b %Y}", f"{latest.Rate:.1%}",
                  delta=delta, delta_color="off", border=True)
        st.caption(f"{latest.Returned:,} of {latest['Previously active']:,} previously active accounts returned.")
    with trend:
        period_chart(returned, monthly, "Rate", key="active_return", fmt=".1%",
                     details=("Returned", "Previously active"), color=COLORS[1],
                     y_domain=[lower / 100, upper / 100])
        st.caption(f"Focused y-axis shows {lower}% to {upper}%.")
    show_table(returned, "active_return")


def retention(s, a, _value):
    start, end, monthly = activity_dates(s, "retention")
    activity = dm.activity_trend(s, a, start, end, fixed=False, monthly=monthly)
    product, subscription = dm.calendar_churn(s, a, start, end, monthly=monthly)
    st.subheader("Product activity after first publish")
    st.caption("Share of published accounts observed in the period that edited a live asset or published another one.")
    period_chart(activity, monthly, "Rate", key="active_share", fmt=".1%", details=("Count", "Eligible"))
    show_table(activity, "active_share")

    st.subheader("28-day active return")
    st.caption("Of the accounts that edited a live asset or published again in the previous 28 days, the share that did either in the next 28 days. Each point uses two adjacent 28-day windows ending on the last day of the displayed week or month.")
    returned = dm.active_return_trend(s, a, start, end, monthly=monthly)
    show_active_return(returned, monthly)

    st.subheader("Churn over time")
    left, right = st.columns(2)
    with left:
        st.markdown("**56-day builder inactivity**")
        st.caption("Previously active accounts that reached 56 days without a live edit or repeat publish, as a share of those at risk when the period began. A return restarts the inactivity clock.")
        period_chart(product, monthly, "Rate", key="calendar_product_churn", fmt=".1%",
                     details=("Count", "Eligible"), color=COLORS[2])
        show_table(product, "calendar_product_churn")
    with right:
        st.markdown("**Subscription churn rate**")
        st.caption("Counts cancellations in the week or month of their recorded date, among accounts subscribed at the start of that period.")
        subscription = subscription.rename(columns={"Count": "Cancellations", "Eligible": "Subscribed at start"})
        period_chart(subscription, monthly, "Rate", key="calendar_subscription_churn", fmt=".1%",
                     details=("Cancellations", "Subscribed at start"), color=COLORS[1])
        show_table(subscription, "calendar_subscription_churn")

    if not returned.empty:
        quiet_end = returned["Window ends"].iloc[-1]
        quiet = dm.quiet_accounts(s, a, quiet_end)
        with_leads = quiet.loc[quiet.Group.eq("5+ leads")].iloc[0]
        if with_leads.Eligible:
            st.info(
                f"**{with_leads.Rate:.1%} of quiet, still-subscribed accounts received 5+ leads** "
                f"({with_leads.Count:,} of {with_leads.Eligible:,}) in the 28 days ending {quiet_end:%d %b %Y}. "
                "These accounts were active in the prior 28 days, then recorded no live edits or repeat publishes. "
                "See Case-study diagnostics for the traffic and lead breakdown."
            )


def churn_by_segment(s, a, value):
    start, end = date_window("Signup dates", s.accounts.signup_date.min(), s.accounts.signup_date.max(),
                             key="churn_diagnostic_dates")
    cohort = a.loc[a.signup_date.between(start, end)]
    mature = cohort.loc[cohort.eligible_90d]
    c = st.columns(2)
    n = int(mature.canceled_90d.sum())
    stat(c[0], "90-day subscription churn", pct(n, len(mature)), f"{n:,} of {len(mature):,} signups with 90 days of follow-up")
    canceled = int(cohort.canceled.sum())
    stat(c[1], "All observed cancellations", pct(canceled, len(cohort)),
         f"{canceled:,} of {len(cohort):,} selected signups through the data cutoff")
    dimension = st.selectbox("Compare by", ["First-publish timing", *dm.DIMENSIONS], key="churn_split")
    split = "publish_timing" if dimension == "First-publish timing" else dm.DIMENSIONS[dimension]
    order = ["Signup days 0-7", "Signup days 8-14", "Unpublished by day 14"] if split == "publish_timing" else None
    if split == "publish_timing":
        st.caption("Publishing groups use accounts still subscribed after signup day 14. The chart counts cancellations in those groups through day 90.")
    else:
        st.caption("The chart counts all cancellations by signup day 90 among signups with 90 days of follow-up.")
    plot_bars(dm.churn_90d(cohort, split), value, key="churn_by_segment", order=order)


def april_activity_check(s, a, value, earliest):
    first = max(earliest, pd.Timestamp("2026-03-30"))
    last = min(s.cutoff, pd.Timestamp("2026-05-03"))
    panel = a.loc[a.index.isin(s.fixed_panel)]
    fixed = dm.activity_trend(s, panel, first, last, fixed=True, monthly=False)
    overall = dm.activity_trend(s, a, first, last, fixed=False, monthly=False)
    frame = pd.concat(
        [overall.assign(Population="Overall"), fixed.assign(Population="Fixed cohort")], ignore_index=True,
    )
    data = frame.loc[frame.Eligible.gt(0)].copy()
    if data.empty:
        st.info("No eligible accounts and complete weeks match these filters.")
        return
    st.caption("Overall includes published accounts observed in each week. The fixed cohort follows the same initially active accounts throughout. The dashed marker shows the week of April 6.")
    data["Week"] = data.Period.dt.strftime("%b %d")
    weeks = data.sort_values("Period").Week.unique().tolist()
    focused = value == "Rate" and st.toggle("Focus activity rate axis", value=True, key="april_focus")
    scale = alt.Scale(zero=True)
    if focused:
        lower, upper = focused_rate_bounds(data.Rate)
        scale = alt.Scale(domain=[lower / 100, upper / 100], nice=False, zero=False)

    x = alt.X("Week:O", sort=weeks, title="Week starting", axis=alt.Axis(labelAngle=0))
    curves = alt.Chart(data).mark_line(point=True, strokeWidth=2.3).encode(
        x=x, y=alt.Y(f"{value}:Q", title="Active share" if value == "Rate" else "Active accounts",
                     axis=alt.Axis(format=".0%" if value == "Rate" else ","), scale=scale),
        color=alt.Color("Population:N", title=None,
                        scale=alt.Scale(domain=["Overall", "Fixed cohort"], range=COLORS[:2]),
                        legend=alt.Legend(orient="top")),
        tooltip=["Week:O", "Population:N", alt.Tooltip("Rate:Q", format=".1%"),
                 alt.Tooltip("Count:Q", format=",", title="Active accounts"),
                 alt.Tooltip("Eligible:Q", format=","), alt.Tooltip("Live edits only:Q", format=",")],
    )
    marker = alt.Chart(data.loc[data.Week.eq("Apr 06"), ["Week"]].drop_duplicates()).mark_rule(
        color="#83949b", strokeDash=[4, 4],
    ).encode(x=x)
    st.altair_chart((marker + curves).properties(height=225).configure_view(stroke=None),
                    width="stretch", key="april_activity_check")
    if focused:
        st.caption(f"Focused y-axis shows {lower}% to {upper}%.")
    show_table(frame, "april_activity_check")


def diagnostics(s, a, value):
    st.subheader("Quiet accounts by traffic and leads")
    earliest = s.accounts.signup_date.min()
    earliest_quiet_end = earliest + 56 * dm.DAY
    if earliest_quiet_end <= s.cutoff:
        quiet_end = st.date_input("Quiet window ends", value=s.cutoff.date(), min_value=earliest_quiet_end.date(),
                                  max_value=s.cutoff.date(), key="quiet_end")
        quiet = dm.quiet_accounts(s, a, pd.Timestamp(quiet_end))
        st.caption(f"{quiet.Count.sum():,} accounts were active in the previous 28 days, went quiet in the selected 28 days, and were still subscribed.")
        plot_bars(quiet, value, key="quiet_account_breakdown", order=quiet.Group.tolist())
    else:
        st.info("The data needs two 28-day windows after the first possible publish to measure quiet accounts.")

    st.subheader("Validation recovery scenario")
    default_start = max(earliest, min(pd.Timestamp("2026-02-01"), s.cutoff))
    default_end = max(default_start, min(pd.Timestamp("2026-04-30"), s.cutoff))
    start, end = date_window("First validation error dates", earliest, s.cutoff, key="validation_dates",
                             default=(default_start.date(), default_end.date()))
    eligible, completed = dm.validation_opportunity(a, start, end)
    unresolved = eligible - completed
    c = st.columns(3)
    stat(c[0], "Accounts flagged", f"{eligible:,}", "All logged rules with 7 days of follow-up")
    stat(c[1], "First publish within 7 days", f"{completed:,}", f"{pct(completed, eligible)} of flagged accounts")
    stat(c[2], "Still unpublished at day 7", f"{unresolved:,}", f"{pct(unresolved, eligible)} of flagged accounts")
    if st.checkbox("Explore a recovery scenario"):
        assumed = st.slider("Assumed share of unresolved cases recovered", 0, 100, 25, step=5, format="%d%%")
        st.metric("Modeled additional first publishers", f"{unresolved * assumed / 100:,.0f}")
    st.caption("A single-rule pilot needs rule, draft, and attempt IDs plus observed incremental first-publish lift.")

    with st.expander("Subscription churn by signup segment"):
        churn_by_segment(s, a, value)
    if earliest <= pd.Timestamp("2026-04-06") <= s.cutoff:
        with st.expander("April activity check"):
            april_activity_check(s, a, value, earliest)


def tab_filters(accounts, page):
    dimensions = dm.ONBOARDING_DIMENSIONS if page == "Onboarding" else dm.DIMENSIONS
    state_key = f"saved_filters_{page}"
    saved = st.session_state.get(state_key, {})
    filters = {}
    with st.sidebar:
        st.caption(f"Applies to every chart in {page}.")
        for label, col in dimensions.items():
            options = dm.TEMPLATE_GROUPS if label == dm.OPENING_TEMPLATE else sorted(accounts[col].dropna().unique())
            selected = [choice for choice in saved.get(label, []) if choice in options]
            filters[label] = st.multiselect(label, options=options, default=selected, placeholder="All",
                                            key=f"filter_{page}_{col}")
        st.session_state[state_key] = filters
        value = "Rate"
        if page in ("Onboarding", "Case-study diagnostics"):
            mode_key = f"display_mode_{page}"
            modes = ["Rate", "Count"]
            value = st.radio("Show charts as", modes, horizontal=True,
                             index=modes.index(st.session_state.get(mode_key, "Rate")),
                             key=f"mode_{page}")
            st.session_state[mode_key] = value
    return dm.select_accounts(accounts, filters, dimensions), value, filters


def main():
    layout()
    st.title("Lead Generation Dashboard")
    uploads = None
    with st.sidebar:
        st.header("Filters")
        with st.expander("Data source"):
            mode = st.radio("Use", ["Bundled case-study CSVs", "Upload four CSVs"])
            if mode == "Upload four CSVs":
                data = {k: st.file_uploader(v, type=["csv"], key=k) for k, v in dm.EXPORTS.items()}
                if any(file is None for file in data.values()):
                    st.info("Upload all four exports to load the dashboard.")
                    st.stop()
                uploads = {k: file.getvalue() for k, file in data.items()}
                st.caption("Uploaded files are used for this session.")
    try:
        metrics_revision = tuple(sha256(path.read_bytes()).digest() for path in METRICS_SOURCES)
        if uploads is None:
            if st.session_state.pop("uploaded_snapshot", None) is not None:
                reset_dates()
            s = load_bundled_snapshot(metrics_revision)
        else:
            s = load_uploaded_snapshot(uploads, metrics_revision)
    except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load the CSVs. {exc}")
        st.stop()
    st.caption(f"Data through {s.cutoff:%d %b %Y}")
    pages = {"Product health": health, "Onboarding": onboarding,
             "Retention & churn": retention, "Case-study diagnostics": diagnostics}
    current = st.radio("Dashboard view", list(pages), horizontal=True, label_visibility="collapsed")
    a, value, filters = tab_filters(s.accounts, current)
    selection = f"{len(a):,} of {len(s.accounts):,} sub-accounts selected"
    filter_summary = [f"{label} = {', '.join(str(choice) for choice in choices)}"
                      for label, choices in filters.items() if choices]
    st.caption(" | ".join([selection, *filter_summary]))
    if a.empty:
        st.info("No accounts match these filters.")
        st.stop()
    st.divider()
    pages[current](s, a, value)


if __name__ == "__main__":
    main()
