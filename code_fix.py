"To call functions, tools from Library"
from __future__ import print_function
import datetime
import numpy as np
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM, GMMHMM
from hmmlearn.base import _BaseHMM
from matplotlib._image import GAUSSIAN
# from Cython.Includes.libcpp import pair

"INPUT"
#number of state
n = 4
#covariance type
covar_type = "full"
#number of iteration
iterr = 1000
#figure name
figname = "result__n_%d" % n

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
model = GaussianHMM(n_components=n, covariance_type=covar_type, n_iter=iterr).fit(X)
# model = GMMHMM(n_components=n_comp, n_iter=1000).fit(X)

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
    if n == 1 and ind == 0:
        result.append([0,0,test,test])
        
    if i != test:
        if len(result) == 0:
            result.append([0,ind-1,test,test])
        else:
            result.append([result[-1][1]+1,ind-1,test, test])
        test = i
for i in range(0,len(result)):
    result[i][0] = x[result[i][0]]
    result[i][1] = x[result[i][1]]
    result[i][2] = model.means_[result[i][2]][0]

"Analysis"
x_i = []
for i in result:
    x_i.append(i[3])
x_i_plus = x_i[:]

del x_i[-1] #remove last element 
x_i_plus.pop(0) #remove first element

X_i = x_i[:]
X_i_plus = x_i_plus[:]
pairr = [[x_i[0],x_i_plus[0]]]
for j in range(1, len(x_i)):
    if [x_i[j],x_i_plus[j]] not in pairr:
        pairr.append([x_i[j],x_i_plus[j]])

density = []
for j in range(0, len(pairr)):
    density.append([pairr[j],0])
    for k in range(0, len(x_i)):
        if pairr[j][0] == x_i[k] and pairr[j][1] == x_i_plus[k]:
            density[-1][-1] += 1

print('x_i     ', x_i)
print('x_i_plus', x_i_plus)            
print('list of [x_i,x_i_plus]', pairr)
print('List of [[x_i,x_i_plus], number of repetition]', density)

# k = 1
# while k == 1:
#     if len(X_i) > 0:
#         density.append([X_i[0], X_i_min[0], 1])
#         X_i.pop(0)
#         X_i_min.pop(0)
#         jj = []
#         for j in range(0, len(X_i)):
#             if X_i[j] == density[-1][0] and X_i_min[j] == density[-1][1]:
#                 density[-1][-1] += 1
#                 jj.append(j)
#         for j in jj:
#             X_i.pop(j)
#             X_i_min.pop(j)    
#     else:
#         k == 0   

"Print RESULT"
print("Record of all hidden state")
print("**********************************")
print("No.","   ","TIME Start"," - ","TIME End","    ","VALUE","    ","Hidden State {}th")
for i in range(0,len(result)):
    print(i, "    ",result[i][0], " - ", result[i][1], "   ", result[i][2], "   ", result[i][3])

   
# "Plot data and result"
# x_plot = []
# y_plot = []
# for i in result:
#     x_plot.append(i[0])
#     x_plot.append(i[1])
#     
#     y_plot.append(i[2])
#     y_plot.append(i[2])
# 
# plt.figure(1)
# plt.title("hmm Gaussian method fitting result vs data")
# plt.plot(x,y, 'r')#, x,y, 'bo')
# plt.plot(x_plot, y_plot, 'k')
# # plt.savefig("diag101000") 
# plt.savefig("%s.png" % figname)
# plt.show()