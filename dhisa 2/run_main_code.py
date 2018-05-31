from coef_setting import declare_coef #1
import main_code as main #2
import picture_2d as pic #3

import numpy
from timeit import default_timer as timer 
import time
import matplotlib.pyplot as plt 
# from mpmath.functions.rszeta import coef

#declare coefficients & initial settings
coef, set, sol = declare_coef() #1

#to plot interactively
plt.ion()

while set['t'] <= set['T'] and set['k'] < set['Nt']:
    sol = main.boolean_1_iter(coef, set, sol) #2                 
    if sol['stop_iter'] >=100000:
        set['k'] = sol['stop_iter']
    '''PRINTING RESULT AS CONTROL'''  
    print 'at Time', set['t']
    print 'Total Tips:', len(sol['matrix_tip'])
    print 'Total Stop Tips:', len(sol['sp_stop'])
    print 'tip stop:', sol['sp_stop']
    print 'Tip Cell Pos:', sol['tip_cell']
    for e, i in enumerate(sol['matrix_tip']):
        if e in sol['sp_stop']:
            print 'stop', e, ':', i[-1], ',', sol['cause'][e], ',', 'Length of sprout:', len(i)
        else:
            print 'RUN', e, ':', i[-1], ',', 'Length of sprout:', len(i)
#     print 'Backward at time step:', sol['backward_count']
    print 'Max Value of c, n', sol['c'].max(),',', sol['n'].max()
    print 'Min Value of c, n', sol['c'].min(),',', sol['n'].min()
    
    if set['k'] % 10 == 0:
        pic.pic_2d(coef,set,sol) #3

    '''Recording Time'''         
    ttime = time.clock()
    if ttime >= 3600: #jam
        jam = int(ttime/3600)
        sisa = ttime - jam*3600
        if sisa == 0:
            print 'total time of processing:', jam, 'hours', 0, 'minutes', 0, 'seconds'
        elif sisa > 60:
            menit = int(sisa/60)
            detik = sisa - menit*60
            if detik == 0:
                print 'total time of processing:', jam, 'hours', menit, 'minutes', 0, 'seconds'
            else:
                print 'total time of processing:', jam, 'hours', menit, 'minutes', detik, 'seconds'
        else:
            print 'total time of processing:', jam, 'hours', 0, 'minutes', sisa, 'seconds'
    elif ttime >= 60: #menit
        menit = int(ttime/60)
        detik = ttime - menit*60
        print 'total time of processing:', 0, 'hours', menit, 'minutes', detik, 'seconds'
    else:
        print 'total time of processing:', 0, 'hours', 0, 'minutes', ttime, 'seconds'
#     print 'tip cell:', sol['tip_cell']
    
    if set['k'] % 100 == 0: #here
        print '''All Coefficients:'''
        print coef
        print '''All Set'''
        print set
    print 'total time of processing:', time.clock()
    print '***************************************************'
    print     
        
    set['t'] += set['dt']
    set['k'] += 1
     
print '*************DONE*****************'
print '''All Coefficients:'''
print coef
print '''All Set'''
print set
raw_input()
# plt.show(block=True)

