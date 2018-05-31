from init_2d import init_2d_ #2.1.1
from init_tip_2d import init_tip_2d_ #2.1.2

import numpy
import random
from random import randint

def initial_prof(coef, set, sol):
    ##For continuous function
    sol = init_2d_(coef,set,sol) #2.1.1
    ##For discrete function
    sol = init_tip_2d_(coef,set,sol) #2.1.2
    #print 'initial tips:', sol['matrix_tip']
    return sol