"To call functions, tools from Library"
from __future__ import print_function
import datetime
import numpy as np
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM, GMMHMM
from hmmlearn.base import _BaseHMM
from mpl_toolkits.mplot3d import Axes3D
import os

"INPUT"
#number of state
n = 10
#covariance type
covar_type = "full"
#number of iteration
iterr = 1000
#figure name
figname1 = "result__n_%d" % n

figname2 = "result__analysis1_3d_scatterplot"

figname3 = "result__analysis1_colormapplot"

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results_0/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

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

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)
print(hidden_states)

"Hidden state"
result = []
test = hidden_states[0]
for ind, i in enumerate(hidden_states):
    if n == 1 and ind == 0:
        result.append([0,0,test,test,0])
         
    if i != test:
        if len(result) == 0:
            result.append([0,ind-1,test,test,0])
        else:
            result.append([result[-1][1]+1,ind-1,test, test,0])
        test = i
for i in range(0,len(result)):
    result[i][0] = x[result[i][0]]
    result[i][1] = x[result[i][1]]
    result[i][2] = model.means_[result[i][2]][0]
    result[i][4] = result[i][1] - result[i][0]
 
"Analysis 1: x_i affects x_i+1 for x_i is hidden state in result"
x_i = []
for i in result:
    x_i.append(i[3])
x_i_plus = x_i[:]
 
del x_i[-1] #remove last element 
x_i_plus.pop(0) #remove first element
 
##Create pair of hidden state
# ##1 based on hidden state vestor
# X_i = x_i[:]
# X_i_plus = x_i_plus[:]
# pairr = [[x_i[0],x_i_plus[0]]]
# for j in range(1, len(x_i)):
#     if [x_i[j],x_i_plus[j]] not in pairr:
#         pairr.append([x_i[j],x_i_plus[j]])
         
##2 based on combination of hidden state
X_i = []
xx = []
yy = []
zz = []
XY = []
for i in range(0,n):
    X_i.append(i)
pairr = []
for i in range(0, n):
    XY.append(i)
    for j in range(0, n):
        pairr.append([i,j])
        xx.append(i)
        yy.append(j)
         
density = []
for j in range(0, len(pairr)):
    density.append([pairr[j],0])
    for k in range(0, len(x_i)):
        if pairr[j][0] == x_i[k] and pairr[j][1] == x_i_plus[k]:
            density[-1][-1] += 1
    zz.append(density[-1][-1])
     
"Analysis 3 : x_i affects x_i+1 for x_i is hidden state in hidden_state"
x_i3 = hidden_states[:]
x_i3 = x_i3.tolist()
x_i3_plus = hidden_states[:]
x_i3_plus = x_i3_plus.tolist()
zz3 = []

print 
del x_i3[-1] #remove last element 
x_i3_plus.pop(0) #remove first element

density3 = []
for j in range(0, len(pairr)):
    density3.append([pairr[j],0])
    for k in range(0, len(x_i3)):
        if pairr[j][0] == x_i3[k] and pairr[j][1] == x_i3_plus[k]:
            density3[-1][-1] += 1
    zz3.append(density3[-1][-1])
 
"Analysis 2 : Total time domain"
analysis_2 = []
for i in range(0,n):
    analysis_2.append([i,0,0])
    for j in result:
        if j[3] == i:
            analysis_2[-1][1] += 1
            analysis_2[-1][2] += j[4]
 
"Print RESULT"
# print("Record of all hidden state")
# print("**********************************")
# print("No.","   ","TIME Start"," - ","TIME End","    ","VALUE","         ","Hidden State {}th","          ","Time domain")
# for i in range(0,len(result)):
#     print(i, "    ",result[i][0], " - ", result[i][1], "   ", result[i][2], "   ", result[i][3], "         ", result[i][4])
 
"Print All hidden state parameter"
# print("Transition matrix")
# print(model.transmat_)
# print()
 
"Print list pair of x_i,x_i+1"
print("record of list pair of x_i,x_i+1")
print("hidden_states original", hidden_states)
print('x_i     ', x_i3)
print('x_i_plus', x_i3_plus)            
print('list of [x_i,x_i_plus]', pairr)
print('List of [[x_i,x_i_plus], number of repetition]', density3)
  
print("Means and Variance of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("total number with this hidden state = ", analysis_2[i][1])
    print("mean = ", model.means_[i])
    print("total time_domain = ", analysis_2[i][2])
#     print("variance = ", np.diag(model.covars_[i]))
    print()
 
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
# plt.savefig("diag101000") 
plt.savefig(results_dir + "%s.png" % figname1)
plt.close()
 
"Plot Analysis 3"
##3D scatter plot
plt.figure(2)
plt.title("Hidden state mapping")
fig = plt.figure(2)
ax = fig.gca(projection='3d')
for i in range(0,len(density)):
    if density[i][1] != 0:
        ax.scatter(density[i][0][0],density[i][0][1],density[i][1],color='b') 
        ax.text(density[i][0][0],density[i][0][1],density[i][1],  '(%s,%s)%s' % (str(density[i][0][0]),str(density[i][0][1]),str(density[i][1])), size=7, zorder=1,  
 color='k') 
# for i,j,k in zip(xx,yy,zz):
#     ax.annotate(str(zz),xyz=(i,j,k))
ax.set_xlabel('Hidden State')
ax.set_ylabel('Hidden State')
ax.set_zlabel('Number')
plt.savefig(results_dir + "%s.png" % figname2)

##colormap plot
X3, Y3 = np.meshgrid(XY, XY)
zz3 = np.asarray(zz3)
Z3 = zz3.reshape(n, n)
plt.figure(3)
plt.title("hidden state mapping")
plt.imshow(Z3,cmap='gist_ncar',origin='lower',interpolation='bilinear')
# plt.pcolor(X3,Y3,Z3)
ax.set_xlabel('Hidden State')
ax.set_ylabel('Hidden State')
plt.colorbar()
plt.savefig(results_dir + "%s.png" % figname3)
plt.show()
