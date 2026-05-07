# Data

The datasets used in this project are proprietary to **OSF HealthCare** and **Illinois State University** and were provided exclusively to registered participants of the ISU Data Science Competition (2026).

They are **not available for public distribution** and are therefore not included in this repository.

---

## Files

| File | Size | Description |
|------|------|-------------|
| `train.csv` | ~37 MB | Labeled training set — contains all 20 feature columns and the `NO_SHOW_FLG` target variable |
| `test.csv` | ~55 MB | Unlabeled test set — contains the same 20 feature columns; `NO_SHOW_FLG` is what the model predicts |
| `sample_submission.csv` | < 1 MB | Expected submission format — two columns: `ID` and `NO_SHOW_FLG` |

---

## Variables

| Variable | Type | Description |
|----------|------|-------------|
| `NO_SHOW_FLG` | Binary | Target — `1` = patient did not arrive and did not cancel; `0` = attended or appropriately cancelled |
| `ID` | Integer | Unique row identifier — required in the submission file |
| `DAYS_BETWEEN_CATEGORY` | Ordinal | Days between scheduling date and appointment date, binned into 6 categories |
| `CONTACT_MONTH_CODE` | Categorical | Anonymized code representing the appointment month |
| `CONTACT_DAY_OF_WEEK_CODE` | Categorical | Anonymized code representing the day of the week |
| `APPT_HOUR_CODE` | Categorical | Anonymized code representing the hour of the appointment |
| `APPT_BLOCK_CODE` | Categorical | Anonymized code representing the time block within the hour |
| `LENGTH` | Ordinal | Appointment length in minutes — `15`, `30`, `45`, or `60` |
| `PATIENT_EMPLOYMENT_STATUS_CATEGORY` | Nominal | Patient employment status at time of appointment |
| `PATIENT_MARITAL_STATUS_CATEGORY` | Nominal | Patient marital status at time of appointment |
| `AGE_CATEGORY` | Ordinal | Patient age at time of appointment, discretized into 10 age bands |
| `VISIT_TYPE` | Nominal | Type of visit — Physical Therapy, Occupational Therapy, or Other |
| `DEPT_NOSHOW_RATE_CATEGORY` | Ordinal | Historical no-show rate at the department level, grouped into 6 categories |
| `DEPT_AVG_APPT2DOC_CATEGORY` | Ordinal | Mean appointment-to-doctor wait time at the department level, binned into 10 categories |
| `DEPT_AVG_ROOM2DOC_CATEGORY` | Ordinal | Mean room-to-doctor wait time at the department level, binned into 5 categories |
| `PROV_NOSHOWRATE_CATEGORY` | Ordinal | Historical no-show rate at the provider level, grouped into 8 categories |
| `PROV_AVG_APPT2DOC_CATEGORY` | Ordinal | Mean appointment-to-doctor wait time at the provider level, binned into 10 categories |
| `PATIENT_NOSHOWRATE_CATEGORY` | Nominal | Patient's prior no-show history — 2 levels |
| `PATIENT_AVG_APPT2DOC_CATEGORY` | Ordinal | Mean appointment-to-doctor wait time for this patient's prior visits, binned into 10 categories |
| `ANY_ED_IN_PRIOR_YEAR` | Binary | Whether the patient had an ED visit in the 365 days prior to the appointment — `Yes` / `No` |
| `ANY_IP_IN_PRIOR_YEAR` | Binary | Whether the patient had an inpatient visit in the 365 days prior to the appointment — `Yes` / `No` |
| `MYCHART_ACTIVATED` | Binary | Whether the patient has an active MyChart patient portal account — `0` / `1` |

---

## Access

This data was made available through the **OSF HealthCare × Illinois State University Data Science Competition**. For access inquiries, please contact Illinois State University directly.
