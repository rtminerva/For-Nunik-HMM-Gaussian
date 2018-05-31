from random import randint, sample, uniform
import numpy
import math as m

def system_2d(coef, set, sol):
#     c_o = sol['c'][:]
    
    for y in range(0,set['Ny']+1,2):
        for x in range(0,set['Nx']+1,2):
            sol['c'][x,y] = (1+m.sin((0.5-x*set['Hh'])*m.pi))* set['ga_1']/4* (m.sin(2*m.pi*set['et_1']*set['t'])+1)
            sol['c_t'][x,y] = 2*m.pi*set['et_1']*m.cos(2*m.pi*set['et_1']*set['t']) * (1+m.sin((0.5-x*set['Hh'])*m.pi))* set['ga_1']/4
                     
#     sol['tip_cell_area'] = []
    
    return sol