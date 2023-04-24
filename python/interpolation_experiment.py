import matplotlib.pyplot as plt
from .marshak_ode import solve_marshak
import numpy as np
import numba
from scipy import interpolate
import time

# Numba accelerated implementation of scipy's splev
def custom_splrep(x, y, k=3):
    
    """
    Custom wrap of scipy's splrep for calculating spline coefficients, 
    which also check if the data is equispaced.
    
    """
    
    # Check if x is equispaced
    x_diff = np.diff(x)
    equi_spaced = all(np.round(x_diff,5) == np.round(x_diff[0],5))
    dx = x_diff[0]
    
    # Calculate knots & coefficients (cubic spline by default)
    t,c,k = interpolate.splrep(x,y, k=k) 
    
    return (t,c,k,equi_spaced,dx) 

@numba.njit(cache=True)
def numba_splev(x, t, c, k, equi_spaced, dx):
    
    """
    Custom implementation of scipy's splev for spline interpolation, 
    with additional section for faster search of knot interval, if knots are equispaced.
    Spline is extrapolated from the end spans for points not in the support.
    
    """
    # t,c,k, equi_spaced, dx = coeff
    
    t0 = t[0]
    
    n = t.size
    x = np.array([x])
    m = x.size
    
    k1  = k+1
    k2  = k1+1
    nk1 = n - k1
    
    l  = k1
    l1 = l+1
    
    y = np.zeros(m)
    
    h  = np.zeros(20)
    hh = np.zeros(19)

    for i in range(m):
        
       # fetch a new x-value arg
       arg = x[i]
       
       # search for knot interval t[l] <= arg <= t[l+1]
       if(equi_spaced):
           l = int((arg-t0)/dx) + k
           l = min(max(l, k1), nk1)
       else:
           while not ((arg >= t[l-1]) or (l1 == k2)):
               l1 = l
               l  = l-1
           while not ((arg < t[l1-1]) or (l == nk1)):
               l = l1
               l1 = l+1
       
       # evaluate the non-zero b-splines at arg.    
       h[:]  = 0.0
       hh[:] = 0.0
       
       h[0] = 1.0
       
       for j in range(k):
       
           for ll in range(j+1):
               hh[ll] = h[ll]
           h[0] = 0.0
       
           for ll in range(j+1):
               li = l + ll 
               lj = li - j - 1
               if(t[li] != t[lj]):
                   f = hh[ll]/(t[li]-t[lj])
                   h[ll] += f*(t[li]-arg)
                   h[ll+1] = f*(arg-t[lj])
               else:
                   h[ll+1] = 0.0
                   break
       
       sp = 0.0
       ll = l - 1 - k1
       
       for j in range(k1):
           ll += 1
           sp += c[ll]*h[j]
       y[i] = sp
    
    return y

# xs = np.linspace(0, 1.1)
# sol = solve_marshak(7/2)
# interp_coeffs =   custom_splrep(np.flip(sol.t), np.flip(sol.y[0]), k=3)
# interp_ob = numba_splev(xs, interp_coeffs)
# plt.plot(xs, interp_ob, 'o')
# plt.plot(sol.t, sol.y[0], '-')
# plt.show()