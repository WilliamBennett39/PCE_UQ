from numba import njit, types, prange, cfunc, carray
import ctypes
import numba
from numba.extending import get_cython_function_address
from numba.types import intc, CPointer, float64
import numpy as np
from scipy.stats import qmc
from scipy.special import eval_hermitenorm
import math
from .interpolation_experiment import numba_splev, numba_splev_list
from .marshak_ode import solve_marshak


_dble = ctypes.c_double
_lng = ctypes.c_long
addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0_1eval_legendre")
functype = ctypes.CFUNCTYPE(_dble, _dble, _dble)
eval_legendre_float64_fn = functype(addr)

@njit
def numba_eval_legendre_float64(n, x):
      return eval_legendre_float64_fn(n, x)

# @njit('float64(int64, float64)')
# def Pn(n,x):
#     tmp = 0.0

#     z = x
#     # fact = np.sqrt((2*n+1)/(b-a)) #*(x>=a)*(x<=b)
#     # tmp[count] = sc.eval_legendre(n,z)*fact
#     tmp = numba_eval_legendre_float64(n, z)
#     return tmp 

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
@njit
def Pn(n, x):
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    elif n == 2:
        return (-1 + 3*x**2)/2.
    elif n == 3:
        return (x*(-3 + 5*x**2))/2.
    elif n == 4:
        return (3 - 30*x**2 + 35*x**4)/8.
    elif n == 5:
        return (x*(15 - 70*x**2 + 63*x**4))/8.
    elif n == 6:
        return (-5 + 21*x**2*(5 - 15*x**2 + 11*x**4))/16.
    elif n == 7:
        return (x*(-35 + 315*x**2 - 693*x**4 + 429*x**6))/16.
    elif n == 8:
        return (35 - 1260*x**2 + 6930*x**4 - 12012*x**6 + 6435*x**8)/128.
    elif n == 9:
        return (x*(315 - 4620*x**2 + 143*x**4*(126 - 180*x**2 + 85*x**4)))/128.
    elif n == 10:
        return (-63 + 3465*x**2 + 143*x**4*(-210 + 630*x**2 - 765*x**4 + 323*x**6))/256.
    elif n == 11:
        return (x*(-693 + 13*x**2*(1155 - 6930*x**2 + 16830*x**4 - 17765*x**6 + 6783*x**8)))/256.
    elif n == 12:
        return (231 + 13*x**2*(-1386 + 17325*x**2 + 17*x**4*(-4620 + 9405*x**2 - 8778*x**4 + 3059*x**6)))/1024.
    elif n == 13:
        return (x*(3003 - 90090*x**2 + 765765*x**4 + 323*x**6*(-8580 + 7*x**2*(2145 - 1794*x**2 + 575*x**4))))/1024.
    elif n == 14:
        return (-429 + 45045*x**2 - 765765*x**4 + 323*x**6*(15015 - 45045*x**2 + 69069*x**4 - 52325*x**6 + 15525*x**8))/2048.
    elif n == 15:
        return (x*(-6435 + 255255*x**2 + 323*x**4*(-9009 + 5*x**2*(9009 + 23*x**2*(-1001 + 1365*x**2 - 945*x**4 + 261*x**6)))))/2048.
    # elif n > 15:
    #     print('only up to n=15 is defined')
    #     assert(0)
    #     return 0.0
@njit
def Pn2(n, x):
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    elif n == 2:
        return (-1 + 3*x**2)/2.
    elif n == 3:
        return (x*(-3 + 5*x**2))/2.
    elif n == 4:
        return (3 - 30*x**2 + 35*x**4)/8.
    elif n == 5:
        return (x*(15 - 70*x**2 + 63*x**4))/8.
    elif n == 6:
        return (-5 + 21*x**2*(5 - 15*x**2 + 11*x**4))/16.
    elif n == 7:
        return (x*(-35 + 315*x**2 - 693*x**4 + 429*x**6))/16.
    elif n == 8:
        return (35 - 1260*x**2 + 6930*x**4 - 12012*x**6 + 6435*x**8)/128.
    elif n == 9:
        return (x*(315 - 4620*x**2 + 143*x**4*(126 - 180*x**2 + 85*x**4)))/128.
    elif n == 10:
        return (-63 + 3465*x**2 + 143*x**4*(-210 + 630*x**2 - 765*x**4 + 323*x**6))/256.
    elif n == 11:
        return (x*(-693 + 13*x**2*(1155 - 6930*x**2 + 16830*x**4 - 17765*x**6 + 6783*x**8)))/256.
    elif n == 12:
        return (231 + 13*x**2*(-1386 + 17325*x**2 + 17*x**4*(-4620 + 9405*x**2 - 8778*x**4 + 3059*x**6)))/1024.
    elif n == 13:
        return (x*(3003 - 90090*x**2 + 765765*x**4 + 323*x**6*(-8580 + 7*x**2*(2145 - 1794*x**2 + 575*x**4))))/1024.
    elif n == 14:
        return (-429 + 45045*x**2 - 765765*x**4 + 323*x**6*(15015 - 45045*x**2 + 69069*x**4 - 52325*x**6 + 15525*x**8))/2048.
    elif n == 15:
        return (x*(-6435 + 255255*x**2 + 323*x**4*(-9009 + 5*x**2*(9009 + 23*x**2*(-1001 + 1365*x**2 - 945*x**4 + 261*x**6)))))/2048.
    elif n == 16:
        return (6435 - 875160*x**2 + 19399380*x**4 - 162954792*x**6 + 669278610*x**8 - 1487285800*x**10 + 1825305300*x**12 - 1163381400*x**14 + 300540195*x**16)/32768.
    elif n == 17:
        return (109395*x - 5542680*x**3 + 81477396*x**5 - 535422888*x**7 + 1859107250*x**9 - 3650610600*x**11 + 4071834900*x**13 - 2404321560*x**15 + 583401555*x**17)/32768.
    elif n == 18:
        return (-12155 + 2078505*x**2 - 58198140*x**4 + 624660036*x**6 - 3346393050*x**8 + 10039179150*x**10 - 17644617900*x**12 + 18032411700*x**14 - 9917826435*x**16 + 2268783825*x**18)/65536.
    elif n == 19:
        return (-230945*x + 14549535*x**3 - 267711444*x**5 + 2230928700*x**7 - 10039179150*x**9 + 26466926850*x**11 - 42075627300*x**13 + 39671305740*x**15 - 20419054425*x**17 + 4418157975*x**19)/65536.
    elif n == 20:
        return (46189 - 9699690*x**2 + 334639305*x**4 - 4461857400*x**6 + 30117537450*x**8 - 116454478140*x**10 + 273491577450*x**12 - 396713057400*x**14 + 347123925225*x**16 - 167890003050*x**18 + 34461632205*x**20)/262144.
    elif n == 21:
        return (969969*x - 74364290*x**3 + 1673196525*x**5 - 17210021400*x**7 + 97045398450*x**9 - 328189892940*x**11 + 694247850450*x**13 - 925663800600*x**15 + 755505013725*x**17 - 344616322050*x**19 + 67282234305*x**21)/262144.
    elif n == 22:
        return (-88179 + 22309287*x**2 - 929553625*x**4 + 15058768725*x**6 - 124772655150*x**8 + 601681470390*x**10 - 1805044411170*x**12 + 3471239252250*x**14 - 4281195077775*x**16 + 3273855059475*x**18 - 1412926920405*x**20 + 263012370465*x**22)/524288.
    elif n == 23:
        return (-2028117*x + 185910725*x**3 - 5019589575*x**5 + 62386327575*x**7 - 429772478850*x**9 + 1805044411170*x**11 - 4859734953150*x**13 + 8562390155550*x**15 - 9821565178425*x**17 + 7064634602025*x**19 - 2893136075115*x**21 + 514589420475*x**23)/524288.
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
def make_expansion_4d(coeffs, basis, NNi, NNj, NNk, NNm, x1, x2, x3, x4):
    res = 0
    # print(NNi, NNj, NNk, NNm, 'Ns')
    # print(coeffs[0, 0, 0, 0], 'coeffs 0')
    for i in prange(0,NNi+1):
        for j in prange(0, NNj+1):
            for k in prange(0, NNk+1):
                for m in prange(0, NNm+1):
                    res += coeffs[i, j, k, m] * b_prod(x1, x2, x3, x4, i, j, k, m, basis)
    return res

@njit
def make_expansion_1d_2(coeffs, basis, NNi, NNj, NNk, NNm, x1, x2, x3, x4):
    res = 0
    # print(NNi, NNj, NNk, NNm, 'Ns')
    # print(coeffs[0, 0, 0, 0], 'coeffs 0')
    for i in range(0,NNi+1):
        res += coeffs[i, 0, 0, 0] * b_prod(x1, x2, x3, x4, i, 0 , 0, 0, basis)
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
    return return_array

@njit
def sampler_sobol_4d(n, coeffs, NN, sample):
    return_array = np.zeros(2**n)
    sample = sample*2-1
    a1 = 0
    for i in range(2**n):
        a1 = make_expansion_4d(coeffs, Pn, NN, NN, NN, NN, sample[i,0], sample[i,1], sample[i,2], sample[i,3]) 
        return_array[i] = a1
        
    return return_array
@njit
def sampler_normal_4d(n, coeffs, NN, sample):
    return_array = np.zeros(2**n)
    a1 = 0
    for i in range(2**n):
        a1 = make_expansion_4d(coeffs, He, NN, NN, NN, NN, sample[i,0], sample[i,1], sample[i,2], sample[i,3]) 
        return_array[i] = a1

    return return_array
@njit
def sampler_normal_3d(n, coeffs, NN, sample):
    return_array = np.zeros(2**n)
    a1 = 0
    for i in range(2**n):
        a1 = make_expansion_4d(coeffs, He, 0, NN, NN, NN, sample[i,0], sample[i,1], sample[i,2], sample[i,3]) 
        return_array[i] = a1
        # print(a1)
    return return_array

@njit
def sampler_sobol_3d(n, coeffs, NN, sample):
    return_array = np.zeros(2**n)
    a1 = 0
    for i in range(2**n):
        a1 = make_expansion_4d(coeffs, Pn, 0, NN, NN, NN, sample[i,0]*2-1, sample[i,1]*2-1, sample[i,2]*2-1, sample[i,3]*2-1) 
        return_array[i] = a1
    return return_array
@njit
def sampler_sobol_1d(n, coeffs, NN, sample):
    return_array = np.zeros(2**n)
    a1 = 0
    for i in range(2**n):
        # if (24.999999999999996*(-0.4 + 0.8610328431001736*(xeval**2)**0.2857142857142857)) >sample[i,0]*2-1:
        #     a1 = 0
        # else:
        a1 = make_expansion_1d_2(coeffs, Pn, NN, 0, 0, 0, sample[i,0]*2-1, sample[i,1]*2-1, sample[i,2]*2-1, sample[i,3]*2-1) 
        return_array[i] = a1
    return return_array
@njit
def sampler_normal_1d(n, coeffs, NN, sample):
    return_array = np.zeros(2**n)
    a1 = 0
    for i in range(2**n):
        a1 = make_expansion_4d(coeffs, He, NN, 0, 0, 0, sample[i,0], sample[i,1], sample[i,2], sample[i,3]) 
        return_array[i] = a1
    return return_array

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
@njit 
def b_prod(x1, x2, x3, x4, l1, l2, l3, l4, basis):
    res = basis(l1, x1) * basis(l2, x2) * basis(l3, x3) * basis(l4, x4)
    return res
@njit 
def b_prod2(x1, x2, x3, x4, l1, l2, l3, l4, basis):
    res = Pn2(l1, x1) * Pn2(l2, x2) * Pn2(l3, x3) * Pn2(l4, x4)
    # print(res, 'b prod', l1, l2, l3, l4, x1, x2, x3, x4)
    return res
 
# @njit 
# def He_prod(x1, x2, x3, x4, l1, l2, l3, l4):
#     return He(l1, x1) * He(l2, x2) * He(l3, x3) * He(l4, x4)
@njit
def Afunc(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4):
    Asq = (kappa0 + a2*x2) * (rho0 + a3*x3)**2 * (cv + a4*x4) / ((T0+a1*x1)**n)

    return np.sqrt(Asq) / omega
@njit
def Afunc2(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4):
    Asq = (kappa0 + a2*x2) * (rho0 + a3*x3)**2 * (cv + a4*x4) / ((T0+a1*x1)**n)

    return np.sqrt(Asq) / omega

@njit
def interpolated_T(x, xmax, tt, c, kK, equi_spaced, dx):
    if 0.0 <= x < xmax:
        res = numba_splev(x, tt, c, kK, equi_spaced, dx)
        # if abs(res[0]) > 1.0:
            # print(res, x)
        return res
    else:
        return np.array([0.0])
@njit
def interpolated_T2(x, xmax, tt, c, kK, equi_spaced, dx):
    # res = x * 0
    # for ix, xx in enumerate(x):
    #     if 0.0 <= xx < xmax:
    #         res[ix] = numba_splev(xx, tt, c, kK, equi_spaced, dx)
        if 0.0 <= x < xmax:
            res = numba_splev(x, tt, c, kK, equi_spaced, dx)
        else:
            res = np.array([0.0])

            # if abs(res[0]) > 1.0:
                # print(res, x)
        return res 



def jit_F1(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True)
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx,n)
        return jitted_function(values)
    return LowLevelCallable(wrapped.ctypes)


# marshak_sol = solve_marshak(7/2)
# interpolator_T_coeffs = custom_splrep(np.flip(marshak_sol.t), np.flip(marshak_sol.y[0]))

# @jit_F1
# def F1_Pn(args):
#     x1 = args[0]
#     x2 = args[1]
#     x3 = args[3]
#     x4 = args[4]
#     x = args[5]
#     t = args[6]
#     T0 = args[7]
#     kappa0 = args[8]
#     rho0 = args[9]
#     cv = args[10]
#     omega = args[11]
#     n = args[12]
#     a1 = args[13]
#     a2 = args[14]
#     a3 = args[15]
#     a4 = args[16]
#     ximax = args[17]
#     # interpolator_T_coeffs = args[18]
#     l1 = args[19]
#     l2 = args[20]
#     l3 = args[21]
#     l4 = args[22]
#     l4 = args[23]

#     xi = x * Afunc(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4)/math.sqrt(t)
#     integrand = (T0 + a1 * x1) * interpolated_T(xi, ximax, interpolator_T_coeffs) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, Pn)
#     return integrand

def F1(x1, x2, x3, x4, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k_interp, equi_spaced, dx, l1, l2, l3, l4):
    xi = x * Afunc(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4) / math.sqrt(t)
    integrand = (T0 + a1 * x1) * interpolated_T(xi, ximax, tt, c, k_interp, equi_spaced, dx) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, Pn)
    # print(interpolated_T(0.4, ximax, tt, c, k_interp, equi_spaced, dx))
    return integrand
@njit
def F1_custom(x1, x2, x3, x4, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k_interp, equi_spaced, dx, l1, l2, l3, l4):
    xi = x * Afunc2(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4) / math.sqrt(t)
    integrand = xi * 0
    for ix, xx in enumerate(xi):
        integrand[ix] = (T0 + a1 * x1[ix]) * interpolated_T2(xx, ximax, tt, c, k_interp, equi_spaced, dx)[0] * b_prod2(x1[ix], x2, x3, x4, l1, l2, l3, l4, Pn)
    # print(interpolated_T(0.4, ximax, tt, c, k_interp, equi_spaced, dx))
    return integrand
# @njit

@njit
def F1_custom_H(x1, x2, x3, x4, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k_interp, equi_spaced, dx, l1, l2, l3, l4):
    xi = x * Afunc2(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4) / math.sqrt(t)
    integrand = xi * 0
    for ix, xx in enumerate(xi):
        integrand[ix] = (T0 + a1 * x1[ix]) * interpolated_T2(xx, ximax, tt, c, k_interp, equi_spaced, dx)[0] * b_prod(x1[ix], x2, x3, x4, l1, l2, l3, l4, He)
    # print(interpolated_T(0.4, ximax, tt, c, k_interp, equi_spaced, dx))
    return integrand
def F1_H(x1, x2, x3, x4, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
    xi = x * Afunc(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4)/math.sqrt(t)
    # w = np.exp(-x1**2/2) *  np.exp(-x2**2/2) *  np.exp(-x3**2/2) *  np.exp(-x4**2/2)
    integrand = (T0 + a1 * x1) * interpolated_T(xi, ximax, tt, c, k, equi_spaced, dx) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, He)
    return integrand
# @njit
def F1_H_weightf(x1, x2, x3, x4, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
    xi = x * Afunc(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4)/math.sqrt(t)
    w = np.exp(-x1**2/2)
    integrand = w * (T0 + a1 * x1) * interpolated_T(xi, ximax, tt, c, k, equi_spaced, dx)* b_prod(x1, x2, x3, x4, l1, l2, l3, l4, He) / math.sqrt(2*math.pi)
    return integrand
@njit
def F1_H_weightf2(x1, x2, x3, x4, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
    xi = x * Afunc2(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4) / math.sqrt(t)
    integrand = xi * 0
    for ix, xx in enumerate(xi):
        w = np.exp(-x1[ix]**2/2)
        integrand[ix] = w * (T0 + a1 * x1[ix]) * interpolated_T(xx, ximax, tt, c, k, equi_spaced, dx)[0] * b_prod(x1[ix], x2, x3, x4, l1, l2, l3, l4, He) / math.sqrt(2*math.pi)
    return integrand

# def F2(x2, x3, x4, x1, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
#     xi = x * Afunc(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4)/math.sqrt(t)
#     integrand = (T0 + a1 * x1) * interpolated_T(xi, ximax, tt, c, k, equi_spaced, dx) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, Pn)
#     return integrand
@njit
def F2_custom(x2, x3, x4, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k_interp, equi_spaced, dx, l1, l2, l3, l4):
    xi = x * Afunc2(0, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4) / math.sqrt(t)
    integrand = xi * 0
    for ix, xx in enumerate(xi):
        integrand[ix] = (T0 + a1 * 0) * interpolated_T2(xx, ximax, tt, c, k_interp, equi_spaced, dx)[0] * b_prod2(0.0, x2[ix], x3, x4, l1, l2, l3, l4, Pn)
    # print(interpolated_T(0.4, ximax, tt, c, k_interp, equi_spaced, dx))
    return integrand

def F2_H(x2, x3, x4, x1, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
    xi = x * Afunc(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4)/math.sqrt(t)
    w = np.exp(-x1**2/2) *  np.exp(-x2**2/2) *  np.exp(-x3**2/2) *  np.exp(-x4**2/2)
    integrand = w * (T0 + a1 * x1) * interpolated_T(xi, ximax, tt, c, k, equi_spaced, dx) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, He)
    return integrand
# coeffs[i,j,k,m] = integrate.nquad(nb_integrand, [[-L,L], [-L, L], [-L,L]], args = (0, 0, 0, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m))[0]

nb_integrand = cfunc("float64[:](float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64, float64, float64, float64, float64, float64, float64[:], float64[:], int64, int64, float64, int64, int64, int64, int64)")(F1)
# nb_integrand_2 = cfunc("float64[:](float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64, float64, float64, float64, float64, float64, float64[:], float64[:], int64, int64, float64, int64, int64, int64, int64)")(F2_custom)
nb_integrand_He = cfunc("float64[:](float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64, float64, float64, float64, float64, float64, float64[:], float64[:], int64, int64, float64, int64, int64, int64, int64)")(F1_H_weightf)
nb_integrand_2_He= cfunc("float64[:](float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64, float64, float64, float64, float64, float64, float64[:], float64[:], int64, int64, float64, int64, int64, int64, int64)")(F2_H)
# nb_integrand_custom = cfunc("float64[:](float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64, float64, float64, float64, float64, float64, float64[:], float64[:], int64, int64, float64, int64, int64, int64, int64)")(F1_custom)



@njit 
def integrate_quad(a, b, xs, ws, func1, args):
    # val = xs* 0 
    # for ix, xx in enumerate(xs):
    val = func1((b-a)/2*xs + (a+b)/2, *args)
    return (b-a)/2 * np.sum(ws * val)

@njit 
def integrate_quad_hermite(xs, ws, func1, args):
    # val = xs* 0 
    # for ix, xx in enumerate(xs):
    return np.sum(ws * func1(xs, *args))

@njit
def quadruple_integral_nb_1(x2, x3, x4, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
        res = x2 * 0
        for it, ix2 in enumerate(x2):
            wave_cutoff = -1
            if a1 != 0:
                wave_cutoff = (-T0 + ((x**2*(cv + a4*x4)*(a2*ix2 + kappa0)*(a3*x3 + rho0)**2)/(t*ximax**2*omega**2))**(1/n))/a1
            # if abs(wave_cutoff) <= 1:
            #     print(wave_cutoff) 
            left_bound = -1
            right_bound = 1
            if abs(wave_cutoff) < 1:
                left_bound = wave_cutoff
            # if -1 < wave_cutoff < 1:
                # left_bound = wave_cutoff
                # print(left_bound, right_bound, x)
                # left_bound = -1
                # print('adjusting interval')
            # elif wave_cutoff < 1:
            #     right_bound = wave_cutoff
            res[it] = integrate_quad(left_bound, right_bound, xs, ws, F1_custom, args = (ix2, x3, x4, x, t, T0, kappa0, rho0,cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
        return res

@njit
def quadruple_integral_nb_2(x3, x4, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
    res = x3 * 0
    for it, ix3 in enumerate(x3):
        left_bound = -1
        right_bound = 1
        res[it] = integrate_quad(left_bound, right_bound, xs, ws, quadruple_integral_nb_1, args = (ix3, x4, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
    return res

@njit
def quadruple_integral_nb_3(x4, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
    res = x4 * 0
    for it, ix4 in enumerate(x4):
        res[it] = integrate_quad(-1, 1, xs, ws, quadruple_integral_nb_2, args = (ix4, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
    return res

@njit
def quadruple_integral_nb_4(xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
    res = integrate_quad(-1, 1, xs, ws, quadruple_integral_nb_3, args = (xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
    return res


@njit
def quadruple_integral_nb_h1(x2, x3, x4, xspn, wspn, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
        res = x2 * 0
        tol = 1e-16
        logtol = math.log(tol)
        for it, ix2 in enumerate(x2):
            wave_cutoff = (-T0 + ((x**2*(cv + a4*x4)*(a2*ix2 + kappa0)*(a3*x3 + rho0)**2)/(t*ximax**2*omega**2))**(1/n))/a1
            right_bound = 2*math.sqrt(-2*logtol)
            val_right = F1_H_weightf2(np.array([right_bound]), ix2, x3, x4, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4) 
            if val_right > 1e-16:
                print('cut off too much', val_right, right_bound)
                # print('adjusting interval')
            # elif wave_cutoff < 1:
            #     right_bound = wave_cutoff
            res[it] = integrate_quad(wave_cutoff, right_bound, xspn, wspn, F1_H_weightf2, args = (ix2, x3, x4, x, t, T0, kappa0, rho0,cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
            # res[it] = integrate_quad_hermite(xspn, wspn, F1_custom_H, args = (ix2, x3, x4, x, t, T0, kappa0, rho0,cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))

        return res

@njit
def quadruple_integral_nb_h2(x3, x4, xspn, wspn, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
    res = x3 * 0
    for it, ix3 in enumerate(x3):
        res[it] = integrate_quad_hermite(xs, ws, quadruple_integral_nb_h1, args = (ix3, x4, xspn, wspn, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
    return res

@njit
def quadruple_integral_nb_h3(x4, xspn, wspn, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
    res = x4 * 0
    for it, ix4 in enumerate(x4):
        res[it] = integrate_quad_hermite(xs, ws, quadruple_integral_nb_h2, args = (ix4,xspn, wspn, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
    return res

@njit
def quadruple_integral_nb_h4(xspn, wspn, xsh, wsh,  x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
    res = integrate_quad_hermite(xsh, wsh, quadruple_integral_nb_h3, args = (xspn, wspn, xsh, wsh, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
    return res

@njit
def triple_integral_nb_1(x3, x4, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
        res = x3 * 0
        for it, ix3 in enumerate(x3):
                # print('adjusting interval')
            # elif wave_cutoff < 1:
            #     right_bound = wave_cutoff

            res[it] = integrate_quad(-1, 1, xs, ws, F2_custom, args = (ix3, x4, x, t, T0, kappa0, rho0,cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
        return res
@njit
def triple_integral_nb_2(x4, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
        res = x4 * 0
        for it, ix4 in enumerate(x4):
                # print('adjusting interval')
            # elif wave_cutoff < 1:
            #     right_bound = wave_cutoff

            res[it] = integrate_quad(-1, 1, xs, ws, triple_integral_nb_1, args =  (ix4, xs, ws, x, t, T0, kappa0, rho0,cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
        return res
@njit
def triple_integral_nb_3(xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
                # print('adjusting interval')
            # elif wave_cutoff < 1:
            #     right_bound = wave_cutoff

        res = integrate_quad(-1, 1, xs, ws, triple_integral_nb_2, args =  (xs, ws, x, t, T0, kappa0, rho0,cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
        return res

@njit
def triple_integral_nb_h1(x3, x4, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
        res = x3 * 0
        for it, ix3 in enumerate(x3):
                # print('adjusting interval')
            # elif wave_cutoff < 1:
            #     right_bound = wave_cutoff

            res[it] = integrate_quad_hermite(xs, ws, F2_custom, args = (ix3, x4, x, t, T0, kappa0, rho0,cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
        return res
@njit
def triple_integral_nb_h2(x4, xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
        res = x4 * 0
        for it, ix4 in enumerate(x4):
                # print('adjusting interval')
            # elif wave_cutoff < 1:
            #     right_bound = wave_cutoff

            res[it] = integrate_quad_hermite(xs, ws, triple_integral_nb_1, args =  (ix4, xs, ws, x, t, T0, kappa0, rho0,cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
        return res
@njit
def triple_integral_nb_h3(xs, ws, x, t, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4):
                # print('adjusting interval')
            # elif wave_cutoff < 1:
            #     right_bound = wave_cutoff

        res = integrate_quad_hermite(xs, ws, triple_integral_nb_2, args =  (xs, ws, x, t, T0, kappa0, rho0,cv, omega, n, a1, a2, a3, a4, ximax, tt, c, k, equi_spaced, dx, l1, l2, l3, l4))
        return res
# def hermite_quad(order = 5):
#     points = np.polynomial.hermite_e.hermeroots(order)
#     weights = points * 0 
#     sqrt_pi = math.sqrt(math.pi)
#     for iw in range(weights.size):
#         weights[iw] = 2**(iw-1) * math.factorial(iw) * sqrt_pi / iw **2 / 


# import chaospy.quadrature as chaos_quad
# int_pnts = 50
# xs_quad, ws_quad = chaos_quad.hermite(int_pnts)
# xs_quad = xs_quad[0]
# ws_quad = ws_quad
# print(xs_quad, 'xs')
# print(ws_quad,'ws')
# print(np.shape(ws_quad), np.shape(xs_quad))
# @njit
# def exp_func(x,b):
#     x_out = np.ones(x.size)
#     return x_out 

# print(np.shape(exp_func(xs_quad,0.0)))
# # print("should be 1.0", integrate_quad_hermite(xs_quad, ws_quad, exp_func, args = (0.0)))
# # print(np.sum(ws_quad * exp_func(xs_quad, 0.0)))
# # print(np.sum(ws_quad))
@njit
def monte_carlo_pn(samples, nsamples, x, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, t, xmax, tt, c, kK, equi_spaced, dx):
    return_array = np.zeros(2**nsamples)
    for i in range(2**nsamples):  
        x1 = samples[i, 0]*2 - 1  
        x2 = samples[i, 1]*2 - 1 
        x3 = samples[i, 2]*2 - 1   
        x4 = samples[i, 3]*2 - 1  
        xi = x * Afunc(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4) / math.sqrt(t)
        res = interpolated_T2(xi, xmax, tt, c, kK, equi_spaced, dx)[0] * (T0 + a1 * x1)
        return_array[i] = res
    return return_array

@njit
def monte_carlo_he(samples, nsamples, x, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4, t, xmax, tt, c, kK, equi_spaced, dx):
    return_array = np.zeros(2**nsamples)
    for i in range(2**nsamples):  
        x1 = samples[i, 0]
        x2 = samples[i, 1]
        x3 = samples[i, 2]
        x4 = samples[i, 3]
        xi = x * Afunc(x1, x2, x3, x4, T0, kappa0, rho0, cv, omega, n, a1, a2, a3, a4) / math.sqrt(t)
        res = interpolated_T2(xi, xmax, tt, c, kK, equi_spaced, dx)[0] * (T0 + a1 * x1)
        return_array[i] = res
    return return_array