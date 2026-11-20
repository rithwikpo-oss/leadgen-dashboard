# Lead Generation Dashboard

Explore onboarding, account activity, retention, churn, and the validation recovery opportunity in the four case-study CSVs. Each view has its own filters for agency size, device, plan, industry, and first builder product. Onboarding also filters by opening-day template selection. Its Compare by control groups both the activation breakdown and the time-to-publish chart. The funnel uses the selected signup dates and sidebar filters.

## Deploy on Streamlit Community Cloud

1. Put the contents of this folder at the root of a **private** GitHub repository. Keep all four exports in the `data/` folder. Include the `.streamlit` folder if you want the supplied theme.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, then select the private repository, its branch, and **`app.py`** as the main file. Streamlit installs the libraries listed in `requirements.txt`.
3. Once the app opens, confirm its visibility is **Private**. Invite reviewers by email in the app’s sharing controls. They can filter the dashboard and download the summary tables.

The dashboard opens on the bundled May 31, 2026 snapshot. The bundled exports are compressed as `.csv.gz`. To refresh the shared version, replace those files or add the corresponding uncompressed `.csv` files with the same base names. **Filters → Data source → Upload four CSVs** previews another set of exports in the current session.

## Run locally

From this folder with Python 3.12 or 3.13:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Metric definitions

| Metric | Denominator or time anchor |
| --- | --- |
| Day-7 activation | Signup cohort with 7 days of follow-up |
| Time to first publish | Days from signup to first successful publish, among publishers |
| Median publish time headline and cohort trend | Days to first publish among accounts that published within 14 days and have 14 days of follow-up |
| Creators still unpublished by signup day 7 | Share of accounts that created an asset by signup day 7 and had no matched successful publish by then, among creators with 7 days of follow-up |
| Lead coverage and repeat publishing | Accounts already published and subscribed when the calendar week or month starts |
| Leads per 100 recorded visitors | Total leads divided by total recorded visitors across those accounts, multiplied by 100 |
| Active share | Share of published accounts observed during the period that edited a live asset or published another one |
| 28-day active return | Share of previously active accounts that take a qualifying action again in the next 28 days. Two adjacent 28-day windows end on the last day of the displayed calendar period |
| 56-day builder inactivity | Share of at-risk accounts at the start of the period that reach 56 days without qualifying product activity |
| Quiet accounts with 5+ leads | Share of still-subscribed accounts that had qualifying activity in the first of two 28-day windows, had no qualifying activity in the second, and received 5+ leads in the second |
| Subscription churn | Share of period-start accounts that record a cancellation during the period |
| 90-day churn diagnostic | Signup cohorts with 90 days of follow-up. The first-publish timing breakdown uses accounts still subscribed after day 14 |
| Validation recovery scenario | Accounts with a first recorded validation error, a complete 7-day window, and no first publish in that window |

`dashboard_metrics.py` reuses the assignment's data cleaning and activity definitions from `leadgen_supporting_analysis.py`. The dashboard uses the accounts export's reporting cutoff. It shows aggregate results and summary downloads.
