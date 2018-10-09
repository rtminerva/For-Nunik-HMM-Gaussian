import matplotlib.pyplot as plt 
import os
import numpy
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def pic_2d(coef,set,sol):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'withexperiment1/')
    
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
      
#     '''1. EC'''
#     fig = plt.figure(1)
#     plt.title('%s%f' % ('Vessel Growth at t=',set['t']))
#     plt.xlabel('X')
#     plt.ylabel('Y') 
#     ax = fig.add_subplot(111)
#     for i in range(0,len(sol['matrix_tip'])):
#         if isinstance(sol['matrix_tip'][i][-1], int) == False:
#             x_p = []
#             y_p = []
#             for j in range(0,len(sol['matrix_tip'][i])):
#                 x_p.append(sol['matrix_tip'][i][j][0]*set['Hh'])
#                 y_p.append(sol['matrix_tip'][i][j][1]*set['Hh'])
#             globals()['plo%s' % i] = ax.plot(x_p, y_p, 'c', color ='k')
#         else:
#             x_p = []
#             y_p = []
#             for j in range(0,len(sol['matrix_tip'][i])-1):
#                 x_p.append(sol['matrix_tip'][i][j][0]*set['Hh'])
#                 y_p.append(sol['matrix_tip'][i][j][1]*set['Hh'])
#             globals()['plo%s' % i] = ax.plot(x_p, y_p, 'c', color ='k')
#     x_pp = []
#     y_pp = []
#     for tip in sol['tip_cell']:
#         x_pp.append(tip[0]*set['Hh'])
#         y_pp.append(tip[1]*set['Hh'])
#     ax.scatter(x_pp, y_pp, marker = 'o', s = 5, color ='r')
# #     '''Backward Marker'''
# #     if len(sol['backward_list']) > 0:
# #         x_pp = []
# #         y_pp = []
# #         for tip in sol['backward_list']:
# #             x_pp.append(tip[0]*set['Hh'])
# #             y_pp.append(tip[1]*set['Hh'])
# #         ax.scatter(x_pp, y_pp, marker = '^', s = 10, color ='c')
#     plt.xlim((set['Hh'],coef['X']-set['Hh']))
#     plt.ylim((set['Hh'],coef['Y']-set['Hh']))
#     sol['stEC'] +=1  
#     flag = 'EC=%s' % str(sol['stEC']) 
#     plt.savefig(results_dir + "%s.png" % flag)
#     plt.close()
    
    '''Record for ecm, vegf'''
    x_sub_axis = numpy.arange(0, coef['X']+set['Hh'], set['h'])
    y_sub_axis = numpy.arange(0, coef['Y']+set['Hh'], set['h'])
    x_sub_axis, y_sub_axis = numpy.meshgrid(x_sub_axis, y_sub_axis)
      
    c_sol = numpy.zeros((set['Nx']/2+1, set['Ny']/2+1))
    for j, y in enumerate(range(0,set['Ny']+1,2)):
        for i, x in enumerate(range(0,set['Nx']+1,2)):
            c_sol[i,j] = sol['c'][x,y]
       
       
#     '''2. Continuous Plot VEGF & ECM'''
# #     if set['k'] % 100 == 0:
#     '''Only VEGF'''
#     fig1 = plt.figure(2)
#     plt.title('%s%f' % ('VEGF Distribution at t=',set['t']))
#     plt.pcolormesh(y_sub_axis, x_sub_axis, c_sol, vmin = 0, vmax = 1, cmap="Wistia", shading = 'gouraud')
#     sol['VEGF'] +=1  
#     flag = 'VEGF=%s' % str(sol['VEGF']) 
#     plt.colorbar()
#     plt.savefig(results_dir + "%s.png" % flag)
#     plt.close()
      
      
    '''3. Continuous Plot VEGF & ECM'''
#     if set['k'] % 1000 == 0:
    '''MERGE_cn'''
    fig13 = plt.figure(6)
    plt.title('%s%f' % ('VEGF and Vessel t=',set['t']))
    ax = fig13.add_subplot(111)
    '''vegf'''
#     c_sol = numpy.ma.masked_array(c_sol, c_sol < 0.0001)#-.5)
    ax.pcolormesh(y_sub_axis, x_sub_axis, c_sol, vmin = 0, vmax = 1, cmap="Wistia", shading = 'gouraud')
    '''tip cell'''
    x_p = []
    y_p = []
    for tip in sol['tip_cell']:
        x_p.append(tip[0]*set['Hh'])
        y_p.append(tip[1]*set['Hh'])
    ax.scatter(x_p, y_p, marker = 'o', s = 10, color ='r')
#     '''Backward Marker'''
#     if len(sol['backward_list']) > 0:
#         x_pp = []
#         y_pp = []
#         for tip in sol['backward_list']:
#             x_pp.append(tip[0]*set['Hh'])
#             y_pp.append(tip[1]*set['Hh'])
#         ax.scatter(x_pp, y_pp, marker = '^', s = 10, color ='c')
    '''Vessel Growth'''
    for i in range(0,len(sol['matrix_tip'])):
        if isinstance(sol['matrix_tip'][i][-1], int) == False:
            x_p = []
            y_p = []
            for j in range(0,len(sol['matrix_tip'][i])):
                x_p.append(sol['matrix_tip'][i][j][0]*set['Hh'])
                y_p.append(sol['matrix_tip'][i][j][1]*set['Hh'])
            globals()['plo%s' % i] = ax.plot(x_p, y_p, 'c', color ='k')
        else:
            x_p = []
            y_p = []
            for j in range(0,len(sol['matrix_tip'][i])-1):
                x_p.append(sol['matrix_tip'][i][j][0]*set['Hh'])
                y_p.append(sol['matrix_tip'][i][j][1]*set['Hh'])
            globals()['plo%s' % i] = ax.plot(x_p, y_p, 'c', color ='k')
    plt.xlim((set['Hh'],coef['X']-set['Hh']))
    plt.ylim((set['Hh'],coef['Y']-set['Hh']))
    sol['Merge_cn'] +=1
    flag = 'Merge_cn=%s' % str(sol['Merge_cn']) 
    plt.savefig(results_dir + "%s.png" % flag)
    plt.close()

    return