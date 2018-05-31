import numpy
import math as m
from click import _winconsole

def cprof(coef,set,sol):
    for y in range(0,set['Ny']+1,2):
        for x in range(0,set['Nx']+1,2):
            sol['c'][x,y] = (1+m.sin(2*m.pi*set['al_1']*x*set['Hh']-set['et_1']*set['t']))*m.exp(-set['xi_1']*set['t'])
        
    return sol

def ctprof(coef,set,sol):
    for y in range(0,set['Ny']+1,2):
        for x in range(0,set['Nx']+1,2):
            sol['c'][x,y] = (1+m.sin((0.5-x*set['Hh'])*m.pi))* set['ga_1']/4* (m.sin(2*m.pi*set['et_1']*set['t'])+1)
            sol['c_t'][x,y] = 2*m.pi*set['et_1']*m.cos(2*m.pi*set['et_1']*set['t']) * (1+m.sin((0.5-x*set['Hh'])*m.pi))* set['ga_1']/4
    return sol



def init_2d_(coef,set,sol): #2.1.1
    sol['c'] = numpy.zeros((set['Nx']+1,set['Ny']+1))
    sol['c_t'] = numpy.zeros((set['Nx']+1,set['Ny']+1))

    sol = ctprof(coef,set,sol)
    return sol
        