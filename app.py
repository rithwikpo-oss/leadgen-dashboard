"""Streamlit entrypoint. Run with streamlit run app.py."""

from contextlib import nullcontext
from hashlib import sha256
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import dashboard_metrics as dm

ROOT = Path(__file__).resolve().parent
COLORS = ["#1B8174", "#355D80", "#DB8254", "#8F77A4", "#CFAB5C"]
st.set_page_config(page_title="Lead Generation Dashboard", layout="wide")


@st.cache_data(show_spinner="Reading the four CSV exports.")
def load_bundled_snapshot():
    files = {}
    for key, name in dm.DEFAULT_EXPORT_NAMES.items():
        csv = ROOT / "data" / name
        files[key] = csv if csv.is_file() else csv.with_suffix(".csv.gz")
    return dm.snapshot_from_exports(files)


def load_uploaded_snapshot(files):
    signature = tuple(sha256(content).digest() for content in files.values())
    cached = st.session_state.get("uploaded_snapshot")
    if cached is None or cached[0] != signature:
        cached = (signature, dm.snapshot_from_exports(files))
        st.session_state["uploaded_snapshot"] = cached
        for key in ("health_dates", "activation_dates", "retention_dates", "quiet_end", "validation_dates",
                    *dm.DIMENSIONS.values()):
            st.session_state.pop(key, None)
    return cached[1]


def layout():
    st.markdown(
        """<style>
        .block-container {padding-top: 2.3rem; padding-bottom: 3rem; max-width: 1240px;}
        [data-testid=stMetric] {background: #f3f8f7; border-radius: .70rem; border-color: #dcebe7;}
        [data-testid=stMetricLabel] {font-weight: 550;}
        [data-testid=stSidebar] {border-right: 1px solid #e2ebe8;}
        .lg-funnel {list-style: none; padding: 0 !important; margin: .75rem 0 1rem !important; display: grid; gap: 4px;}
        .lg-funnel li {position: relative; isolation: isolate; min-height: 66px; margin: 0; padding: .4rem .5rem; display: grid; align-content: center; text-align: center; color: #123331;}
        .lg-funnel-shape {position: absolute; inset: 0; z-index: -1;}
        .lg-funnel-stage {font-size: .83rem; font-weight: 660; line-height: 1.22;}
        .lg-funnel-primary {font-size: 1.34rem; font-weight: 720; font-variant-numeric: tabular-nums;}
        .lg-funnel-secondary {font-size: .79rem; color: #315859; margin-left: .4rem; font-variant-numeric: tabular-nums;}
        </style>""",
        unsafe_allow_html=True,
    )


def date_window(label, first, last, *, key, default=None):
    selected = st.date_input(label, value=default or (first.date(), last.date()),
                             min_value=first.date(), max_value=last.date(), key=key)
    if not isinstance(selected, (tuple, list)) or len(selected) != 2:
        st.info("Select both the start and the end date.")
        st.stop()
    return tuple(pd.Timestamp(d) for d in selected)


def pct(n, d):
    return f"{n / d:.1%}" if d else "-"


def stat(container, name, value, detail):
    with container:
        st.metric(name, value, border=True)
        st.caption(detail)


def show_table(frame, key, *, visible=False):
    table = frame.copy()
    if "Period" in table:
        table["Period"] = pd.to_datetime(table.Period).dt.strftime("%Y-%m-%d")
    display = table.copy()
    for column in ("Rate", "From previous stage"):
        if column in display:
            display[column] = display[column].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
    with (nullcontext() if visible else st.expander("View counts and download table")):
        st.dataframe(display, hide_index=True, use_container_width=True)
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
            f'style="background:{colors[i]};clip-path:{polygon}"></div>'
            f'<span class="lg-funnel-stage">{escape(row.Stage)}</span>'
            f'<div><span class="lg-funnel-primary">{primary}</span>'
            f'<span class="lg-funnel-secondary">{secondary}</span></div></li>'
        )
    return '<ol class="lg-funnel" aria-label="Onboarding funnel">' + "".join(segments) + "</ol>"


def show_funnel(frame, value):
    st.markdown(funnel_markup(frame, value), unsafe_allow_html=True)
    largest = frame.loc[frame["Drop-off"].idxmax()]
    if largest["Drop-off"]:
        st.caption(f"Largest loss by count � {largest['Drop-off']:,} accounts before {largest.Stage.lower()} "
                   f"({1 - largest['From previous stage']:.1%} of the previous stage).")
    show_table(frame, "activation_funnel")


def plot(frame, x, value, *, key, group=None, line=False, horizontal=False, select=None):
    if frame.empty:
        st.info("No complete periods or eligible accounts match the selected filters.")
        return
    x_type = "T" if x == "Period" else "N"
    if x_type == "T":
        frame = frame.assign(Period=pd.to_datetime(frame.Period))
    tip = [alt.Tooltip(f"{x}:{x_type}", **({"format": "%d %b %Y"} if x_type == "T" else {}))]
    for c in [group, "Count", "Eligible", "Rate", "Live edits only", "Publishers", value]:
        if not c or c == x or c not in frame or c in [t.shorthand.split(":")[0] for t in tip]:
            continue
        fmt = ".1%" if c == "Rate" else ",.1f" if c.endswith("(days)") else ",.0f"
        if c == group:
            tip.append(alt.Tooltip(f"{c}:N"))
        else:
            tip.append(alt.Tooltip(f"{c}:Q", format=fmt))
    value_enc = alt.Axis(format=".0%" if value == "Rate" else ",~s")
    encoding = {
        "x": alt.X(f"{value}:Q", axis=value_enc, title=value) if horizontal else
        alt.X(f"{x}:{x_type}", title=None, sort=select, axis=alt.Axis(labelAngle=0)),
        "y": alt.Y(f"{x}:{x_type}", title=None, sort=select, axis=alt.Axis(labelLimit=220)) if horizontal else
        alt.Y(f"{value}:Q", axis=value_enc, title=value),
        "tooltip": tip,
    }
    if group:
        encoding["color"] = alt.Color(f"{group}:N", title=None, scale=alt.Scale(range=COLORS))
        if not line and not horizontal:
            encoding["xOffset"] = alt.XOffset(f"{group}:N")
    chart = alt.Chart(frame)
    if line:
        chart = chart.mark_line(point=alt.OverlayMarkDef(size=55), color=COLORS[0], strokeWidth=2.8)
    else:
        chart = chart.mark_bar(color=COLORS[0], cornerRadiusEnd=4)
    chart = chart.encode(**encoding).properties(height=280).configure_view(stroke=None)
    st.altair_chart(chart, use_container_width=True, key=key)
    show_table(frame, key)


def health(s, a, value):
    cutoff = s.cutoff
    earliest = s.accounts.signup_date.min()
    default = (max(earliest, cutoff - 62 * dm.DAY).date(), cutoff.date())
    controls = st.columns((2, 1, 2))
    with controls[0]:
        start, end = date_window("Activity dates", earliest, cutoff, key="health_dates", default=default)
    with controls[1]:
        cadence = st.radio("Group by", ["Week", "Month"], horizontal=True, key="health_cadence")
    with controls[2]:
        base = st.radio("Activity cohort", ["All published accounts", "April case-study panel"],
                        horizontal=True, key="activity_cohort")
    fixed = base == "April case-study panel"
    panel = a.loc[a.index.isin(s.fixed_panel)] if fixed else a
    volume, extras = dm.established_28d(s, a, end)
    repeat = extras[dm.ID].nunique()
    n, d = dm.calendar_return(s, panel, end)
    window_start = end - 27 * dm.DAY
    st.subheader("Lead generation and repeat publishing")
    st.caption(f"{window_start:%d %b}-{end:%d %b %Y} � Accounts published before this 28-day window and subscribed at its start.")
    c = st.columns(4)
    leads = int(volume.leads.ge(5).sum())
    stat(c[0], "Received 5+ leads", pct(leads, len(volume)), f"{leads:,} of {len(volume):,} eligible accounts")
    visitors, captured = int(volume.visitors.sum()), int(volume.leads.sum())
    stat(c[1], "Leads per 100 visitors", f"{100 * captured / visitors:.1f}" if visitors else "-",
         f"{captured:,} leads from {visitors:,} recorded visitors")
    stat(c[2], "Published again", pct(repeat, len(volume)), f"{repeat:,} of {len(volume):,} eligible accounts")
    stat(c[3], "Additional assets per account", f"{len(extras) / len(volume):.2f}" if len(volume) else "-",
         f"{len(extras):,} assets across {len(volume):,} eligible accounts")
    st.subheader("Activity and return")
    monthly = cadence == "Month"
    frame = dm.activity_trend(s, panel, start, end, fixed=fixed, monthly=monthly)
    active, returned = st.columns(2)
    if frame.empty:
        stat(active, "Active accounts in the latest period", "-", "Select a complete week or month to see activity.")
    else:
        latest = frame.iloc[-1]
        period = latest.Period.to_period("M" if monthly else "W-SUN")
        stat(active, f"Active accounts in latest {cadence.lower()}", f"{latest.Count:,}",
             f"{period.start_time:%d %b}-{period.end_time:%d %b %Y} � {pct(latest.Count, latest.Eligible)} of {latest.Eligible:,} in scope")
    stat(returned, "28-day active-account return", pct(n, d),
         f"{n:,} of {d:,} active in the preceding 28 days � {window_start:%d %b}-{end:%d %b %Y}")
    detail = ("The April panel follows the same initially active accounts across periods." if fixed else
              "The activity rate uses published accounts with observation time during each displayed period.")
    st.caption(f"Activity means a live edit or a later distinct publish. {detail}")
    plot(frame, "Period", value, key="activity_over_time", line=True)


def onboarding(s, a, value):
    min_date = s.accounts.signup_date.min()
    max_date = s.accounts.signup_date.max()
    left, right = st.columns((2, 2))
    with left:
        start, end = date_window("Signup dates", min_date, max_date, key="activation_dates")
    with right:
        split_name = st.selectbox("Compare by", list(dm.DIMENSIONS), index=1, key="activation_split")
    a = a.loc[a.signup_date.between(start, end)]
    eligible = a.loc[a.signup_date.add(7 * dm.DAY).le(a.end)]
    activated = int(eligible.ttp.between(0, 7).sum())
    median = a.ttp.median() if a.ttp.notna().any() else float("nan")
    unpublished = int((a.created_asset & ~a.funnel_published).sum())
    c = st.columns(3)
    stat(c[0], "7-day activation", pct(activated, len(eligible)), f"{activated:,} of {len(eligible):,} signups with 7 days of follow-up")
    stat(c[1], "Median time to publish", f"{median:g} days" if pd.notna(median) else "-",
         f"{a.ttp.notna().sum():,} accounts with a first publish")
    stat(c[2], "Asset creators who did not publish", f"{unpublished:,}", f"{pct(unpublished, int(a.created_asset.sum()))} of creators")
    a1, a2 = st.columns(2)
    with a1:
        st.subheader("Onboarding funnel")
        st.caption("Share of selected signups reaching each stage by the data cutoff.")
        show_funnel(dm.activation_funnel(a), value)
    with a2:
        st.subheader(f"7-day activation by {split_name.lower()}")
        table = dm.rate_table(eligible, eligible.ttp.between(0, 7), dm.DIMENSIONS[split_name])
        plot(table, "Group", value, horizontal=True, key="activation_by_group")
    st.subheader("Time to first publish")
    st.caption("Days from signup to the first successful publish, among accounts that published.")
    ttp_split = st.radio("Break down publish time by", [split_name, "First asset template use"],
                         horizontal=True)
    if ttp_split == "First asset template use":
        publishers = a.loc[a.launch.notna()].assign(
            template=a.used_template.map({1: "Used template", 0: "Started from scratch"})
        )
        column = "template"
    else:
        publishers = a.loc[a.launch.notna()]
        column = dm.DIMENSIONS[split_name]
    ttp = dm.time_to_publish(publishers, column).round(1)
    if ttp.empty:
        st.info("No accounts in this selection have a successful publish.")
    else:
        st.caption("P90 is the time by which 90% of publishers had published.")
        show_table(ttp[["Group", "Publishers", "Median (days)", "P90 (days)",
                        "P25 (days)", "P75 (days)", "Mean (days)"]], "time_to_first_publish",
                   visible=True)

    st.subheader("Opening-day template selection")
    opened = a.loc[a.open.add(7 * dm.DAY).le(a.end)].copy()
    opened["template_group"] = opened.template_selected.map({
        True: "Selected a template", False: "No template selected",
    })
    table = dm.rate_table(opened, opened.open_to_publish.between(0, 7), "template_group")
    st.caption("First publish within seven days of first builder open. Groups reflect recorded behavior.")
    plot(table, "Group", value, horizontal=True, key="activation_by_template")
    st.subheader("Activation by signup cohort")
    st.caption("Each cohort includes signups with seven days of follow-up. Date selections can cut across a week or month.")
    monthly = st.radio("Cohort size", ["Week", "Month"], horizontal=True, key="activation_cohort_size")
    plot(dm.weekly_activation(a, "M" if monthly == "Month" else "W-SUN"), "Period", value,
         key="activation_cohorts", line=True)


def retention(s, a, value):
    left, right = st.columns(2)
    with left:
        start, end = date_window("Cohort dates", s.accounts.signup_date.min(), s.cutoff, key="retention_dates")
    with right:
        anchor = st.radio("Cohort starts at", ["First publish", "Signup"], horizontal=True,
                          key="retention_anchor")
    cohort = a.loc[a.launch.between(start, end)] if anchor == "First publish" else a.loc[a.signup_date.between(start, end)]
    if anchor == "First publish":
        eligible = cohort.loc[cohort.launch.add(28 * dm.DAY).le(cohort.end)]
        n = int(eligible.repeat_28d.sum())
        panel = dm.launch_cohorts(cohort, date_window=(start, end))
        done = panel.loc[panel.Window.eq("Weeks 9-12")]
        c = st.columns(2)
        stat(c[0], "New 28-day repeat publishing", pct(n, len(eligible)), f"{n:,} of {len(eligible):,} first publishers")
        stat(c[1], "Active in weeks 9-12", pct(done.Count.sum(), done.Eligible.sum()),
             f"{done.Count.sum():,} of {done.Eligible.sum():,} initially active accounts with full follow-up")
        st.subheader("Product activity after first publish")
        st.caption("Initially active accounts were subscribed on day 28 and had activity in the first four weeks. Cohorts have full follow-up through week 12. A partial month reflects a date filter.")
        plot(panel, "First publish month", value, group="Window", key="launch_retention")
        dimension = st.selectbox("Repeat publishing breakdown", list(dm.DIMENSIONS), key="repeat_dimension")
        plot(dm.rate_table(eligible, eligible.repeat_28d, dm.DIMENSIONS[dimension]), "Group", value,
             horizontal=True, key="repeat_by_dimension")
    else:
        mature = cohort.loc[cohort.eligible_90d]
        c = st.columns(2)
        n = int(mature.canceled_90d.sum())
        stat(c[0], "90-day subscription churn", pct(n, len(mature)), f"{n:,} of {len(mature):,} signups with 90 days of follow-up")
        canceled = int(cohort.canceled.sum())
        stat(c[1], "All observed cancellations", pct(canceled, len(cohort)),
             f"{canceled:,} of {len(cohort):,} selected signups through the data cutoff")
        st.subheader("90-day churn by cohort segment")
        dimension = st.selectbox("Compare by", ["First-publish timing", *dm.DIMENSIONS], key="churn_split")
        split = "publish_timing" if dimension == "First-publish timing" else dm.DIMENSIONS[dimension]
        order = ["Signup days 0-7", "Signup days 8-14", "Unpublished by day 14"] if split == "publish_timing" else None
        if split == "publish_timing":
            st.caption("Publishing groups use accounts still subscribed after signup day 14. The chart counts cancellations in those groups through day 90.")
        else:
            st.caption("The chart counts all cancellations by signup day 90 among signups with 90 days of follow-up.")
        plot(dm.churn_90d(cohort, split), "Group", value, horizontal=True, key="churn_by_segment", select=order)


def diagnostics(s, a, value):
    st.subheader("Quiet accounts by traffic and leads")
    earliest = s.accounts.signup_date.min()
    quiet_end = st.date_input("Quiet window ends", value=s.cutoff.date(), min_value=earliest.date(),
                              max_value=s.cutoff.date(), key="quiet_end")
    quiet = dm.quiet_accounts(s, a, pd.Timestamp(quiet_end))
    st.caption(f"{quiet.Count.sum():,} accounts were active in the previous 28 days, went quiet in the selected 28 days, and were still subscribed.")
    plot(quiet, "Group", value, horizontal=True, key="quiet_account_breakdown", select=quiet.Group.tolist())

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
        if uploads is None:
            st.session_state.pop("uploaded_snapshot", None)
            s = load_bundled_snapshot()
        else:
            s = load_uploaded_snapshot(uploads)
    except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load the CSVs. {exc}")
        st.stop()
    with st.sidebar:
        filters = {}
        for label, col in dm.DIMENSIONS.items():
            options = sorted(s.accounts[col].dropna().unique().tolist())
            filters[label] = st.multiselect(label, options=options, placeholder="All", key=col)
        value = st.radio("Show charts as", ["Rate", "Count"], horizontal=True)
    a = dm.select_accounts(s.accounts, filters)
    st.caption(f"Data through {s.cutoff:%d %b %Y} � {len(a):,} of {len(s.accounts):,} sub-accounts selected")
    if filters["First builder product"]:
        st.caption("Product is assigned from the first recorded builder open. The complete signup base includes accounts without a recorded open.")
    if a.empty:
        st.info("No accounts match these filters.")
        st.stop()
    pages = {"Product health": health, "Onboarding": onboarding,
             "Retention & churn": retention, "Case-study diagnostics": diagnostics}
    current = st.radio("Dashboard view", list(pages), horizontal=True, label_visibility="collapsed")
    st.divider()
    pages[current](s, a, value)


if __name__ == "__main__":
    main()
