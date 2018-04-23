"To call functions, tools from Library"
from __future__ import print_function
import datetime
import numpy as np
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM, GMMHMM
from hmmlearn.base import _BaseHMM
from matplotlib._image import GAUSSIAN
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D
# from Cython.Includes.libcpp import pair

"INPUT"
#number of state
<<<<<<< HEAD
n = 15
=======
n = 3
>>>>>>> branch 'master' of https://github.com/rtminerva/NunikHMM-Gaussian.git
#covariance type
covar_type = "full"
#number of iteration
iterr = 1000
#figure name
figname1 = "Data2n=15RTV300mVCurrent4%d" % n

figname2 = "3DData2n=15RTV300mVCurrent4"

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

"Analysis 1: x_i affects x_i+1"
x_i = []
for i in result:
    x_i.append(i[3])
x_i_plus = x_i[:]

del x_i[-1] #remove last element 
x_i_plus.pop(0) #remove first element

##Create pair of hidden state
# ## based on hidden state vestor
# X_i = x_i[:]
# X_i_plus = x_i_plus[:]
# pairr = [[x_i[0],x_i_plus[0]]]
# for j in range(1, len(x_i)):
#     if [x_i[j],x_i_plus[j]] not in pairr:
#         pairr.append([x_i[j],x_i_plus[j]])
        
## based on combination of hidden state
X_i = []
xx = []
yy = []
zz = []
for i in range(0,n):
    X_i.append(i)
pairr = []
for i in range(0, n):
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

"Analysis 2 : Total time domain"
analysis_2 = []
for i in range(0,n):
    analysis_2.append([i,0,0])
    for j in result:
        if j[3] == i:
            analysis_2[-1][1] += 1
            analysis_2[-1][2] += j[4]

"Print RESULT"
print("Record of all hidden state")
print("**********************************")
print("No.","   ","TIME Start"," - ","TIME End","    ","VALUE","         ","Hidden State {}th","          ","Time domain")
for i in range(0,len(result)):
    print(i, "    ",result[i][0], " - ", result[i][1], "   ", result[i][2], "   ", result[i][3], "         ", result[i][4])

"Print All hidden state parameter"
# print("Transition matrix")
# print(model.transmat_)
# print()

"Print list pair of x_i,x_i+1"
print("record of list pair of x_i,x_i+1")
print('x_i     ', x_i)
print('x_i_plus', x_i_plus)            
print('list of [x_i,x_i_plus]', pairr)
print('List of [[x_i,x_i_plus], number of repetition]', density)
 
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
plt.savefig("%s.png" % figname1)
plt.close()
plt.show()

"Plot Analysis 1 Result"
# xx = np.linspace(0,n,10)
# yy = np.linspace(0,n,10)
# XX, YY = np.meshgrid(xx,yy)
# sigmaa = 0.1
# #parameter to set in plot
# for i in range(0,len(pairr)):
#     mu_x = pairr[i][0]
#     mu_y = pairr[i][1]
#     sigma_x = sigmaa
#     sigma_y = sigmaa
#     globals()['Z_%s' % i] = bivariate_normal(XX,YY,sigma_x,sigma_y,mu_x,mu_y)
##plotting
plt.figure(2)
fig = plt.figure(2)
ax = fig.gca(projection='3d')
# for i in range(0,len(pairr)):
#     globals()['plo%s' % i] = ax.plot_surface(XX, YY, globals()['Z_%s' % i],cmap='viridis',linewidth=0)
# ax.scatter(xx, yy, zz)
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
plt.savefig("%s.png" % figname2)
plt.show()