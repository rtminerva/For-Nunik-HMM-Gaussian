from collections import OrderedDict

def declare_coef():
    '''Coefficients'''
    #to store coefficients
    coef = {}
    set = {}
    sol = {}

    '''measurement'''
    ra = 0.06 #cm 
    x = ra*2 
    y = ra*2 
    T_1 = 86400 #s
    
    '''Tip (n)'''
    #diffusion
    d_n = 10**(-10) #cm^2s^(-1)  
    coef['D_n'] = d_n*T_1/(ra**2+ra**2)
#     coef['D_n'] = 1.8*10**(-4) #aubert 
    #chemotaxis
#     ki_n = 2600-750 #cm^2 s^(-1) M^(-1) #stokes 1990
    ki_n = 2600-1000 #cm^2 s^(-1) M^(-1) #aubert estimation 650 - 750
    c_o = 10**(-10) #M
    coef['Ki_n'] = ki_n*T_1*c_o/(ra**2+ra**2)
#     coef['Ki_n'] = 0.133 #aubert

#     '''VEGF (c)'''
#     #diffusion
# #     d_c = 2.9*10**(-7) #cm^2s^(-1) #Anderson and Chaplain, Bray
#     d_c = 5.6*10**(-10) #aubert estimation 5.6*10**(-9) - 1.4*10**(-8)
# #     d_c = 2.8*10**(-8) #miura 2009
#     coef['D_c'] = 0#d_c*T_1/(ra**2+ra**2)
# #     coef['D_c'] = 0.01 #aubert
#     coef['Nu'] = 5#1
#     coef['lam'] = 0.05#0.05
 
    '''Branching & Mitosis'''
    coef['T_branch'] = 0.25
#     coef['T_mitosis'] = 1000#0.5

    set['al_1'] = 0 #gradient c
    set['be_1'] = 2.5 #gradient c_t
    
#     set['xi_1'] = 1
    
    set['ga_1'] = 1 #amplitude of vegf
    set['et_1'] = 1 #oscilation vel vegf
    
    set['u'] = 2 #area of patch
    
    
    
    '''Spatial and Temporal Meshes Number'''
    ##set dictionaries tidak pernah berubah
    coef['X'] = 1
    coef['Y'] = 1
    set['T'] = 5.002
    set['Nt'] = 100000
    set['theta'] = 0.5
    set['rad'] = 0.01/ra
    
    set['h'] = 0.005 #0.01#
    set['dt'] = 0.005#001
    
    set['Hh'] = set['h']/2
    set['Nx'] = int(coef['X']/set['Hh'])
    set['Ny'] = int(coef['Y']/set['Hh'])
    
    set['O_x'] = set['Nx']/2*set['Hh']
    set['O_y'] = set['Ny']/2*set['Hh']
    
    '''Initial Setting'''
    set['t'] = 0
    set['k'] = 0
       
    '''To store solutions'''
    ##sol dictionaries dapat berubah
    sol['stEC'] = 0
    sol['Merge_cn'] = 0
    sol['Merge_cnd'] = 0
    sol['Merge_fn'] = 0
    sol['VEGF'] = 0
    sol['VEGF1'] = 0
    sol['ECM'] = 0
    sol['stop_iter'] = 0
    '''For Anastomosis'''
#     sol['pp'] = {}
#     sol['PP'] = [] 
    sol['backward'] = {}
    sol['backward_count'] = []
    sol['backward_list'] = []
    sol['tip_tip_anas'] = []
    sol['cause'] = {}

    return coef, set, sol