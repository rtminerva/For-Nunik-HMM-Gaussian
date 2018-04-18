"To call functions, tools from Library"
from __future__ import print_function

import datetime

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
try:
    from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
    # For Matplotlib prior to 1.5.
    from matplotlib.finance import (
        quotes_historical_yahoo as quotes_historical_yahoo_ochl
    )

from hmmlearn.hmm import GaussianHMM

# print(__doc__)

"Import data from excel file"
from xlrd import open_workbook
book = open_workbook('data.xlsx')
sheet = book.sheet_by_index(0)

x = []
y = []

for k in range(1,sheet.nrows):
    x.append(str(sheet.row_values(k)[1-1]))
    y.append(str(sheet.row_values(k)[2-1]))

x = np.asarray(map(float, x))
y = np.asarray(map(float, y))

X = np.reshape(y,(-1,1))

"Run Gaussian HMM"
# Make an HMM instance and execute fit
n_comp = 2
model = GaussianHMM(n_components=n_comp, covariance_type="full", n_iter=1000).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)
print("done fitting to HMM")

"Print All hidden state parameter"
print("Transition matrix")
print(model.transmat_)
print()

print("Means and Variance of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("variance = ", np.diag(model.covars_[i]))
    print()

"Hidden state"
result = []
test = hidden_states[0]
for ind, i in enumerate(hidden_states):
    if n_comp == 1 and ind == 0:
        result.append([0,0,test])
        
    if i != test:
        if len(result) == 0:
            result.append([0,ind-1,test])
        else:
            result.append([result[-1][1]+1,ind-1,test])
        test = i
for i in range(0,len(result)):
    result[i][0] = x[result[i][0]]
    result[i][1] = x[result[i][1]]
    result[i][2] = model.means_[result[i][2]][0]

"Print RESULT"
print("Record of all hidden state")
print("**********************************")
print("TIME Start","   ","TIME End","      ","VALUE")
for i in range(0,len(result)):
    print(result[i][0], "    ", result[i][1], "     ", result[i][2])

   
"Plot data and result"
x_plot = []
y_plot = []
for i in result:
    x_plot.append(i[0])
    x_plot.append(i[1])
    
    y_plot.append(i[2])
    y_plot.append(i[2])

plt.figure(1)
plt.title("hmm Gaussian method fitting result vs data")
plt.plot(x,y, 'r')#, x,y, 'bo')
plt.plot(x_plot, y_plot, 'k')
plt.savefig("resultdatan5")
plt.show()