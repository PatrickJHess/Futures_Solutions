# ***Setup The Notebook***

* **Get `financial_quant`**  
* **Import needed libraries and modules**  
* **Instantiate FredReader, MASSIVEReader, and static_futures_data**

```py
# Use Python's built-in urllib to fetch the remote setup script
import urllib.request

# Download, decode, and execute the updater script directly from GitHub
updater_url = "https://raw.githubusercontent.com/PatrickJHess/quant_repo/master/fq_updater.py"
exec(urllib.request.urlopen(updater_url).read().decode('utf-8'))

# Run the installer/updater function and alias the package as 'fq'
fq = import_financial_quant()

# Import libraries and modules
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime, date
from IPython.display import clear_output

# Instantiate the data pipelines from the financial_quant package
fred_data = fq.FredReader()
massive_data = fq.MASSIVEReader(calls_per_minute=5)
static_futures_data from fq.static_futures_data
```
**Required Functions**

The following needed custom functions defined in the **Using Futures to Isolate Sources of Risk and Return** are:
* `futures_data`
* `predict_actual`
* `plot_tracking_error`

*Tip: It is highly recommended that you run the setup and import these functions before you attempt to complete this challenge.*

---

# Step 1

## ***Fetch the Data***

```py
# 1. Futures Data
df_mar6 = futures_data('H6', '2025-12-11', '2026-03-12', clear_logs=True, wait_to_clear=True)
df_jun6 = futures_data('M6', '2026-03-12', '2026-06-11', clear_logs=True, wait_to_clear=True)
df_futures_raw = pd.concat([df_mar6, df_jun6])
df_futures = df_futures_raw.dropna()
display(df_futures.loc['2026-03-12'])

# 2. Target Portfolios
df_spfin = static_futures_data('SPFIN_Index')
df_sptech = static_futures_data('SPTECH_Index')
df_tri = df_spfin.join(df_sptech, how='inner')
df_tri.index = pd.to_datetime(df_tri.index)
display(df_tri.loc['2026-03-12'])

# 3. SOFR Index
sofr_index = fred_data.get_series('SOFRINDEX', start_date=df_futures.index[0], end_date=df_futures.index[-1])
sofr_index.index = pd.to_datetime(sofr_index.index)
display(sofr_index.loc['2026-03-12'])
```
## 

## ***Join and Clean The Data***

```py
# 1. Left join the portfolios onto the futures data
df_futures_cash = df_futures.join(df_tri, how='left')

# 2. Left join the SOFR index and drop missing data
df_futures_cash_sofr = df_futures_cash.join(sofr_index, how='left').dropna()

# 3. Rename the column for easier access later
df_futures_cash_sofr.rename(columns={'SOFRINDEX': 'sofr_index'}, inplace=True)

# 4. Safeguard: Remove duplicate columns if the cell is run multiple times
df_futures_cash_sofr = df_futures_cash_sofr.loc[:, ~df_futures_cash_sofr.columns.duplicated()]

# Display the roll date to verify
display(df_futures_cash_sofr.loc['2026-03-11':'2026-03-12'])
```
##

---

# ***Step 2***

## ***Calculate Percentage Changes***

```py
# 1. Calculate percentage changes for the entire DataFrame
df_pct_dup = df_futures_cash_sofr.pct_change()

# 2. Filter out the duplicate roll-date row (keep only the first instance)
df_pct_clean = df_pct_dup[~df_pct_dup.index.duplicated(keep='first')]

# Display the roll dates to verify the duplicate is gone
display(df_pct_clean.loc['2026-03-11':'2026-03-12'])
```
## 

## ***Calculate Excess Returns***

```py
# 1. Isolate the Futures (assuming your columns are ES, NQ, RTY, YM)
futures_cols = ['ES', 'NQ', 'RTY', 'YM']
df_excess = df_pct_clean[futures_cols].copy()

# 2. Subtract SOFR from the target portfolios
portfolios = df_pct_clean[['SPFINTR_Index', 'SPTECHTR_Index']]
excess_portfolios = portfolios.sub(df_pct_clean['sofr_index'], axis=0)

# 3. Combine and Clean
df_excess = pd.concat([df_excess, excess_portfolios], axis=1).dropna()

# Display the final excess return dataframe
display(df_excess.head())
```
##

---

# ***Step 3***

## ***Use OLS to Estimate Futures Allocations***

```py
import statsmodels.api as sm
import numpy as np

# 1. Independent (X) variables for both portfolios
futures = df_excess_returns[['SP500_FUT', 'RUSS_FUT', 'NSDQ_FUT', 'DJIA_FUT']].dropna()

# 2. Add a constant to the futures data
X = sm.add_constant(futures).rename(columns={'const': 'Delta Funding'})

# --- Financials OLS ---
y_fin = df_excess_returns['SPFINTR_Index'].dropna()
results_fin = sm.OLS(y_fin, X).fit()

df_res_fin = pd.DataFrame([results_fin.params, results_fin.bse], index=['Allocation', 'Std. Error'])
df_res_fin['R-squared'] = [results_fin.rsquared, np.nan]
print("Financials Allocations:")
display(df_res_fin)

# --- Information Technology OLS ---
y_tech = df_excess_returns['SPTECHTR_Index'].dropna()
results_tech = sm.OLS(y_tech, X).fit()

df_res_tech = pd.DataFrame([results_tech.params, results_tech.bse], index=['Allocation', 'Std. Error'])
df_res_tech['R-squared'] = [results_tech.rsquared, np.nan]
print("Info Tech Allocations:")
display(df_res_tech)
```
---

# ***Step 4***

## ***Calculate the Cumulative Tracking Error***

```py
# 1. Clean the entire dataset at once so X and y align perfectly
df_clean = df_excess_returns.dropna()

# 2. Define the futures toolkit (X) once for both models
df_x = df_clean[['SP500_FUT','RUSS_FUT','NSDQ_FUT','DJIA_FUT']]

# --- Financial Sector Portfolio ---
df_y_fin = df_clean['SPFINTR_Index']

# Train up to and including the roll date
NOBS_fin = len(df_y_fin.loc[:'2026-03-12']) + 1

# Get forecast and actuals
forecast_fin, actual_fin = predict_actual(df_y_fin, df_x, NOBS=NOBS_fin)

# Calculate cumulative tracking error index
financial_error = (1 + pd.Series(actual_fin - forecast_fin, index=df_y_fin.index)).cumprod()


# --- Information Technology Sector Portfolio ---
df_y_tech = df_clean['SPTECHTR_Index']

# Train up to and including the roll date
NOBS_tech = len(df_y_tech.loc[:'2026-03-12']) + 1

# Get forecast and actuals
forecast_tech, actual_tech = predict_actual(df_y_tech, df_x, NOBS=NOBS_tech)

# Calculate cumulative tracking error index
infotech_error = (1 + pd.Series(actual_tech - forecast_tech, index=df_y_tech.index)).cumprod()
```
---

# ***Step 5***

## ***Ready To Graph***

You have made it to the finish line! All the heavy lifting is done.

Now, just pass your cumulative tracking error data into the custom <code>plot_tracking_error</code> function provided earlier in the notebook. Be sure to label them clearly using the <code>portfolio_name</code> argument.

If your data pipeline and forecasting models worked correctly, you ought to get the exact same charts shown at the very top of this challenge. Great work!
</blockquote>

