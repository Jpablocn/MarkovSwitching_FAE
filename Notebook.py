
import numpy as np
import pandas as pd
from MarkovSwitching_FAE import MarkovSwitching_FAE
from pandas_datareader.data import DataReader
from datetime import datetime

from statsmodels.tsa.regime_switching.tests.test_markov_autoregression import rgnp

usrec = DataReader('USREC', 'fred', start=datetime(1947, 1, 1), end=datetime(2013, 4, 1))
# Data RGNP, Hamilton
dta_hamilton = pd.Series(rgnp, index=pd.date_range('1951-04-01', '1984-10-01', freq='QS'))

# Fit()
mod_hamilton = MarkovSwitching_FAE(dta_hamilton, k_regimes=2, order=4, switching_ar=False)
res_hamilton = mod_hamilton.fit()

print(res_hamilton.summary())
print(res_hamilton.expected_durations)

"""
 PLOT DATA
 
import matplotlib.pyplot as plt 

dta_hamilton.plot(title='Growth rate of Real GNP', figsize=(12, 3))
plt.show()

plt.plot(res_hamilton.filtered_marginal_probabilities[0])
plt.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='gray', alpha=0.3)
plt.xlim(dta_hamilton.index[4], dta_hamilton.index[-1])
plt.ylim(0, 1)
plt.title('Filtered probability of recession')
plt.show()
"""

# Data FEDFUNDS
from statsmodels.tsa.regime_switching.tests.test_markov_regression import fedfunds

dta_fedfunds = pd.Series(fedfunds, index=pd.date_range('1954-07-01', '2010-10-01', freq='QS'))

# Fit(). A switching mean is the default of the MarkovRegession model
mod_fedfunds = MarkovSwitching_FAE(dta_fedfunds, k_regimes=2, order=2)
res_fedfunds = mod_fedfunds.fit()

print(res_fedfunds.summary())