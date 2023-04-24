import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate
from scipy import integrate
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from scipy import optimize


def solve_marshak(n):
    if n == 3:
        xmax = 1.121
    elif n == 0:
        xmax = 1.232
    elif n == 3.5:
        xmax = 1.11110
    T0 = 1
    
    g = lambda x, xmax: ((n+3)*xmax*(-x+xmax)/(n+4))**(1/(n+3))
    gprime = lambda x, xmax: -(((n+3)/(n+4))*xmax)**(1/(n+3))*1/(n+3)*(xmax-x)**((-2-n)/(n+3))
    RHSfun = lambda t,v: np.array([v[1], v[1]*(-v[0]**(-n)*t/(n+4) - (n+3)*v[0]**2*v[1])/(v[0]**3)])


    tol = 1e-9
    IC = [g(xmax-tol,xmax),gprime(xmax-tol,xmax)]
    output = integrate.DOP853(RHSfun, xmax,IC,0)
    opt_fun = lambda xm: integrate.solve_ivp(RHSfun, (xm-tol,0), [g(xm-tol,xm),gprime(xm-tol,xm)], method='DOP853', max_step = 1e-4).y[0][-1] - T0
    minval = optimize.brentq(opt_fun, 1.0,1.3)
    print(minval, 'minval')

    sol_marshak = integrate.solve_ivp(RHSfun, (minval-tol,0), IC, method='DOP853', max_step = 1e-4)

    print(sol_marshak.y[0][-1], "should be 1.0")
    # plt.ion()
    # plt.plot(sol_marshak.t, sol_marshak.y[0])
    # plt.show()

    return sol_marshak


# t_final = 1
# Cv = 0.3
# c=29.998     #cm/ns
# a=0.01372
# kappa0 = 300
# K = a*c*4/(3*(n+4)*kappa0*Cv*(T0**n))
# A = sqrt(1/(2*K)) 
