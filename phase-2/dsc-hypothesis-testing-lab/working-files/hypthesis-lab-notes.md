# Hypothesis Lab Notes

## High Level Summary

### Data

cdc 2017-2018 annual survey

#### Prior to each statistical test, you will need to perform some data preparation, which could include:

- Filtering out rows with irrelevant values
- Transforming data from codes into human-readable values
- Binning data to transform it from numeric to categorical
- Creating new columns based on queries of the values in other columns

### Key Stakeholders and Questions

Chief Analytics Officer and Chief Marketing Officer

1) How does health status, represented by average number of days with bad physical health in the past month (PHYSHLTH), differ by state?

2) Digging deeper into the data, what are some factors that impact health (demographics, behaviors, etc.)?

### EDA/Cleaning

'PHYSHLTH' column:

- 1 row not asked or missing
- Column: 91-92
- Type of Variable: Num
- SAS Variable Name: 'PHYSHLTH'
- Question Prologue:
- Question: Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?

'POORHLTH':

- based on 'PHYSHLTH' and 'MENTHLTH' responses
- Section 02.01, 'PHYSHLTH', is 88 (aka no issues) and Section 2.02, 'MENTHLTH', is 88 (aka no issues)