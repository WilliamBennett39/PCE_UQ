from numba import njit, types, prange
import ctypes
from numba.extending import get_cython_function_address
import numpy as np
from scipy.stats import qmc
from scipy.special import eval_hermitenorm


_dble = ctypes.c_double
_lng = ctypes.c_long
addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0_1eval_legendre")
functype = ctypes.CFUNCTYPE(_dble, _dble, _dble)
eval_legendre_float64_fn = functype(addr)

@njit
def numba_eval_legendre_float64(n, x):
      return eval_legendre_float64_fn(n, x)

@njit('float64(int64, float64)')
def Pn(n,x):
    tmp = 0.0

    z = x
    # fact = np.sqrt((2*n+1)/(b-a)) #*(x>=a)*(x<=b)
    # tmp[count] = sc.eval_legendre(n,z)*fact
    tmp = numba_eval_legendre_float64(n, z)
    return tmp 

_dble = ctypes.c_double
addr = get_cython_function_address("scipy.special.cython_special", "eval_hermitenorm")
functype = ctypes.CFUNCTYPE(_lng, _dble, _dble)
eval_hermitenorm_float64_fn = functype(addr)

@njit
def numba_eval_hermite_float64(n, x):
      return eval_hermitenorm_float64_fn(n, x)

# @njit('float64(int64, float64)')
# def He(n,x):
#     tmp = 0.0
#     z = x

#     # tmp = numba_eval_hermite_float64(n, z)
#     tmp = eval_hermitenorm(n,z)

#     return tmp 
@njit
def He(m,x):
    n = m - 1
    if n == -1:
        return 1 
    elif n == 0:
        return x
    elif n == 1:
        return -1 + x**2
    elif n == 2:
        return x*(-3 + x**2)
    elif n == 3:
        return 3 - 6*x**2 + x**4
    elif n == 4:
        return x*(15 - 10*x**2 + x**4)
    elif n == 5:
        return -15 + 45*x**2 - 15*x**4 + x**6
    elif n ==6:
       return x*(-105 + 105*x**2 - 21*x**4 + x**6)
    elif n == 7:
        return 105 - 420*x**2 + 210*x**4 - 28*x**6 + x**8
    elif n == 8:
        return x*(945 - 1260*x**2 + 378*x**4 - 36*x**6 + x**8)
    elif n == 9:
        return -945 + 4725*x**2 - 3150*x**4 + 630*x**6 - 45*x**8 + x**10
    elif n == 10:
        return x*(-10395 + 17325*x**2 - 6930*x**4 + 990*x**6 - 55*x**8 + x**10)
    elif n == 11:
        return 10395 - 62370*x**2 + 51975*x**4 - 13860*x**6 + 1485*x**8 - 66*x**10 + x**12
    elif n == 12:
        return x*(135135 - 270270*x**2 + 135135*x**4 - 25740*x**6 + 2145*x**8 - 78*x**10 + x**12)
    elif n == 13:
        return -135135 + 945945*x**2 - 945945*x**4 + 315315*x**6 - 45045*x**8 + 3003*x**10 - 91*x**12 + x**14
    elif n == 14:
       return x*(-2027025 + 4729725*x**2 - 2837835*x**4 + 675675*x**6 - 75075*x**8 + 4095*x**10 - 105*x**12 + x**14)
    elif n == 15:
        return 2027025 - 16216200*x**2 + 18918900*x**4 - 7567560*x**6 + 1351350*x**8 - 120120*x**10 + 5460*x**12 - 120*x**14 + x**16
    elif n == 16:
        return x*(34459425 - 91891800*x**2 + 64324260*x**4 - 18378360*x**6 + 2552550*x**8 - 185640*x**10 + 7140*x**12 - 136*x**14 + x**16)
    elif n == 17:
        return -34459425 + 310134825*x**2 - 413513100*x**4 + 192972780*x**6 - 41351310*x**8 + 4594590*x**10 - 278460*x**12 + 9180*x**14 - 153*x**16 + x**18
    elif n == 18:
        return x*(-654729075 + 1964187225*x**2 - 1571349780*x**4 + 523783260*x**6 - 87297210*x**8 + 7936110*x**10 - 406980*x**12 + 11628*x**14 - 171*x**16 + x**18)
    elif n > 18:
        print('only up to n=18 is defined')
        assert(0)
@njit
def make_expansion_1d(coeffs, basis, NN, x):
    res = 0
    for n in range(0,NN):
        res += coeffs[n] * basis(n, x)
    return res

@njit
def make_expansion_1d_h(coeffs,basis, NN, x):
    res = 0
    for n in range(0,NN):
        res += coeffs[n] * He(n, x)
    return res

@njit
def sampler_sobol(n, coeffs, NN, sample):
    return_array = np.zeros(2**n)
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    for i in range(2**n):
        a1 = make_expansion_1d(coeffs[0], Pn, NN, sample[i,0]*2-1) 
        a2 =  make_expansion_1d(coeffs[1], Pn, NN, sample[i,1]*2-1)
        a3 = make_expansion_1d(coeffs[2], Pn, NN, sample[i,2]*2-1)
        a4 = make_expansion_1d(coeffs[3], Pn, NN, sample[i,3]*2-1)
        
        return_array[i] = a1*a2*a3*a4 

    # print(make_expansion_1d(coeffs[0], Pn, NN, 0.5), "0")
    # print(make_expansion_1d(coeffs[1], Pn, NN, 0.5), "1")
    # print(make_expansion_1d(coeffs[2], Pn, NN, 0.5), "2")
    # print(make_expansion_1d(coeffs[3], Pn, NN, 0.5), "3")

    return return_array

@njit
def sampler_normal(n, coeffs, NN, sample):
    return_array = np.zeros(2**n)
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    for i in range(2**n):
        a1 = make_expansion_1d_h(coeffs[0], He, NN, sample[i,0]) 
        a2 =  make_expansion_1d_h(coeffs[1], He, NN, sample[i,1])
        a3 = make_expansion_1d_h(coeffs[2], He, NN, sample[i,2])
        a4 = make_expansion_1d_h(coeffs[3], He, NN, sample[i,3])
        
        return_array[i] = a1*a2*a3*a4 

    return return_array

