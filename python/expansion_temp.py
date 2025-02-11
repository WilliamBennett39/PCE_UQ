import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import math
import h5py
import time
from numba import prange
from scipy.stats import qmc
# import quadpy
from tqdm import tqdm

from .marshak_ode import solve_marshak
from .functions import b_prod, Afunc, Pn, He, interpolated_T, nb_integrand, nb_integrand_He, nb_integrand_2_He
from .functions import integrate_quad, quadruple_integral_nb_4, F1,  quadruple_integral_nb_1, quadruple_integral_nb_3,  quadruple_integral_nb_4
from .functions import quadruple_integral_nb_h4, quadruple_integral_nb_h1, quadruple_integral_nb_h3, triple_integral_nb_3
from .functions import monte_carlo_pn, monte_carlo_he, triple_integral_nb_h3
from .interpolation_experiment import custom_splrep
import chaospy.quadrature as chaos_quad


def opts0(*args, **kwargs):
       return {'limit':50, 'epsabs':1.5e-12, 'epsrel':1.5e-12}

class coefficients_4d:
    def __init__(self):
        self.parameters = np.loadtxt('marshak_pars.csv',delimiter=",", dtype=float)
        self.T0 = self.parameters[0]
        self.ximax = self.parameters[1]
        self.kappa0 = self.parameters[2]
        self.rho0 = self.parameters[3]
        self.cv = self.parameters[4]
        self.a1 = self.parameters[5]
        self.a2 = self.parameters[6]
        self.a3 = self.parameters[7]
        self.a4 = self.parameters[8]
        self.omega = self.parameters[9]
        self.NPn = self.parameters[10]
        self.NHe = self.parameters[11]
        self.n = self.parameters[12]
        self.t = self.parameters[13]
        self.xmaxmax = self.parameters[14]
        self.Npnts = int(self.parameters[15])
        self.xmaxhe = self.parameters[16]
        self.xlist = np.linspace(0, self.xmaxmax, self.Npnts)
        self.xlist_h = np.linspace(0, self.xmaxhe, self.Npnts)
        print(self.a1, 'a1')
        NPn = int(20)
        NHe = int(20)
        self.c_Pn_drive = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_Pn_target = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_Pn_all = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_He_drive = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        self.c_He_target = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        self.c_He_all = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
                
        # self.c_Pn_drive = np.zeros((NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        # self.c_Pn_target = np.zeros((NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        # self.c_Pn_all = np.zeros((NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        # self.c_He_drive = np.zeros((NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        # self.c_He_target = np.zeros((NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        # self.c_He_all = np.zeros((NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        # solve the Marshak ODE for given value of n
        self.sol_marshak()
        self.save_marshak_sol()
    
        # self.interpolator_T = interpolate.interp1d(marshak_sol.t, marshak_sol.y[0], kind = 'cubic')
        self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx  = custom_splrep(np.flip(self.marshak_sol.t), np.flip(self.marshak_sol.y[0]))

    def sol_marshak(self):
        print('solving marshak ODE')
        self.marshak_sol = solve_marshak(self.n)
        print('done')
    # def interpolated_T(self, x):
    #     if 0 <= x <= self.ximax:
    #         return self.interpolator_T(x)
    #     else:
    #         return 0
    
    def integrate_coeffs_4d(self, a1, a2, a3, a4, M, basis, coeffs, L, x):
        dimensionalize = lambda x1, x2, x3, x4: (self.T0 + a1 * x1) * interpolated_T(x * Afunc(x1, x2, x3, x4, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4)/math.sqrt(self.t), self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
        integrand = lambda x1, x2, x3, x4, l1, l2, l3, l4: dimensionalize(x1, x2, x3, x4) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, basis)

        for i in range(M+1):
            for j in range(M+1):
                for k in range(M+1):
                    for m in range(M+1):
                        if basis == Pn:
                            # coeffs[i,j,k,m] = integrate.nquad(self.integrate_coeffs_temp, [[-L,L],[-L,L],[-L,L]], args = (a1, a2, a3, a4, M, basis, L, x, i, j, k, m), opts = [opts0, opts0, opts0])[0]
                            coeffs[i,j,k,m] = integrate.nquad(nb_integrand, [[-L,L],[-L,L],[-L,L],[-L,L]], args = ( x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m), opts = [opts0, opts0, opts0, opts0])[0]
                        elif basis == He:
                            coeffs[i,j,k,m] = integrate.nquad(nb_integrand_He, [[-L,L],[-L,L],[-L,L],[-L,L]], args = ( x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m), opts = [opts0, opts0, opts0, opts0])[0]


        return coeffs
    
    def integrate_coeffs_4d_2(self, a1, a2, a3, a4, M, basis, coeffs, L, x, xs_pnts_pn, xs_ws_pn, xs_pnts, xs_ws):
        dimensionalize = lambda x1, x2, x3, x4: (self.T0 + a1 * x1) * interpolated_T(x * Afunc(x1, x2, x3, x4, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4)/math.sqrt(self.t), self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
        integrand = lambda x1, x2, x3, x4, l1, l2, l3, l4: dimensionalize(x1, x2, x3, x4) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, basis)
        for i in prange(M+1):
            for j in prange(M+1):
                for k in prange(M+1):
                    for m in prange(M+1):
                        if basis == Pn:
                            # coeffs[i,j,k,m] = integrate.nquad(self.integrate_coeffs_temp, [[-L,L],[-L,L],[-L,L]], args = (a1, a2, a3, a4, M, basis, L, x, i, j, k, m), opts = [opts0, opts0, opts0])[0]
                            coeffs[i,j,k,m] = quadruple_integral_nb_4(xs_pnts_pn, xs_ws_pn, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m)
                        elif basis == He:
                            coeffs[i,j,k,m] = quadruple_integral_nb_h4(xs_pnts_pn, xs_ws_pn, xs_pnts, xs_ws, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m)

        return coeffs

    def integrate_coeffs_1d(self, a1, a2, a3, a4, M, basis, coeffs, L, x, xs_pnts_pn, xs_ws_pn, xs_pnts, xs_ws):
        dimensionalize = lambda x1, x2, x3, x4: (self.T0 + a1 * x1) * interpolated_T(x * Afunc(x1, x2, x3, x4, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4)/math.sqrt(self.t), self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
        integrand = lambda x1, x2, x3, x4, l1, l2, l3, l4: dimensionalize(x1, x2, x3, x4) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, basis)
        j = 0
        k = 0 
        m = 0 
        # print(F1(0, 0, 0, 0, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, 0, 0, 0, 0))
        for i in prange(M+1):
            if basis == Pn:
            # coeffs[i,j,k,m] = integrate.nquad(integrand, [[-L,L]], args = (0, 0, 0, i, j, k, m))[0]
                # coeffs[i,j,k,m] =integrate.nquad(nb_integrand, [[-L,L]], args = (0, 0, 0, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m))[0]
                coeffs[i,j,k,m] = quadruple_integral_nb_1(np.array([0.0]), 0.0, 0.0, xs_pnts_pn, xs_ws_pn, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m)
            elif basis == He:
                coeffs[i,j,k,m] = quadruple_integral_nb_h1(np.array([0.0]), 0.0, 0.0, xs_pnts_pn, xs_ws_pn, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m)

                # coeffs[i,j,k,m] = integrate.nquad(nb_integrand_He, [[-np.inf,np.inf]], args = (0, 0, 0, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m))[0]
        return coeffs
    
    def integrate_coeffs_temp(self, x2, x3, x4, a1, a2, a3, a4, M, basis, L, x, i, j, k, m):
        wave_cutoff = (-self.T0 + ((x**2*(self.cv + a4*x4)*(a2*x2 + self.kappa0)*(a3*x3 + self.rho0)**2)/(self.t*self.ximax**2*self.omega**2))**(1/self.n))/a1
        left_bound = -1
        right_bound = 1
        if wave_cutoff > -1:
            left_bound = wave_cutoff
        elif wave_cutoff < 1:
            right_bound = wave_cutoff
        return integrate.nquad(nb_integrand,[[left_bound, right_bound]], args = (x2, x3, x4, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m), opts = [opts0, opts0, opts0])[0]


    def integrate_coeffs_3d(self, a1, a2, a3, a4, M, basis, coeffs, L, x, xs_pnts_pn, xs_ws_pn, xs_pnts, xs_ws):
        dimensionalize = lambda x1, x2, x3, x4: (self.T0 + a1 * x1) * interpolated_T(x * Afunc(x1, x2, x3, x4, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4)/math.sqrt(self.t), self.ximax, self.interp_t, self.interp_c, self.interp_k,self.interp_equi_spaced, self.interp_dx)
        integrand = lambda x2, x3, x4, x1, l1, l2, l3, l4: dimensionalize(x1, x2, x3, x4) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, basis)
        i = 0 
        for j in prange(M+1):
            for k in prange(M+1):
                for m in prange(M+1):
                    if basis == Pn:
                    # coeffs[i,j,k,m] = integrate.nquad(integrand, [[-L,L], [-L, L], [-L,L]], args = (0, i, j, k, m), opts = [opts0, opts0, opts0])[0]
                        # coeffs[i,j,k,m] = integrate.nquad(nb_integrand_2,[[-L,L], [-L, L], [-L,L]], args = (0, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m), opts = [opts0, opts0, opts0])[0]
                        coeffs[i,j,k,m] = triple_integral_nb_3(xs_pnts, xs_ws, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m)

                    elif basis == He:
                        # coeffs[i,j,k,m] = integrate.nquad(nb_integrand_2_He,[[-L,L], [-L, L], [-L,L]], args = (0, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m), opts = [opts0, opts0, opts0])[0]
                        coeffs[i,j,k,m] = triple_integral_nb_h3(xs_pnts, xs_ws, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m)
                        # coeffs[i,j,k,m] = triple_integral_nb_h3(np.array([0.0]), xs_pnts, xs_ws, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m)

        return coeffs

    def make_all_coefficients_pn(self, M, int_pnts = 40):
        tick = time.perf_counter()
        NPn = int(M)
        NHe = int(M)
        self.order_pn = int_pnts
        blank_mat_pn = np.zeros((M+1, M+1, M+1, M+1))
        self.c_Pn_drive = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_Pn_target = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_Pn_all = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        # xs_quad = quadpy.c1.gauss_legendre(int_pnts).points
        # ws_quad = quadpy.c1.gauss_legendre(int_pnts).weights
        xs_quad, ws_quad = chaos_quad.legendre_proxy(int_pnts, domain =(-1,1))
        xs_quad = xs_quad[0]
        for ix, x in enumerate(tqdm(self.xlist)):
            # print(ix/self.xlist.size * 100, "percent complete")
            # print(x, 'x')
            # self.c_Pn_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, He, blank_mat_pn, np.inf, x)
            # self.c_Pn_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            # self.c_Pn_all[ix] = self.integrate_coeffs_4d(self.a1, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_all[ix] = self.integrate_coeffs_4d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)

            # current 
            # self.c_Pn_drive[ix] = self.integrate_coeffs_1d(self.a1, 0.0, 0.0, 0.0, M, Pn, blank_mat_pn, 1, x, xs_quad, ws_quad)
            # self.c_Pn_target[ix] = self.integrate_coeffs_3d(self.a1, self.a2, self.a3, self.a3, M, Pn, blank_mat_pn, 1, x, xs_quad, ws_quad)
            self.c_Pn_all[ix] = self.integrate_coeffs_4d_2(self.a1, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x, xs_quad, ws_quad, xs_quad, ws_quad)
        print('elapsed', time.perf_counter()-tick)
        self.factors(M)

    def make_target_coefficients_pn(self, M, int_pnts = 40):
        tick = time.perf_counter()
        NPn = int(M)
        NHe = int(M)
        self.order_pn = int_pnts
        blank_mat_pn = np.zeros((M+1, M+1, M+1, M+1))
        self.c_Pn_drive = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_Pn_target = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_Pn_all = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        # xs_quad = quadpy.c1.gauss_legendre(int_pnts).points
        # ws_quad = quadpy.c1.gauss_legendre(int_pnts).weights
        xs_quad, ws_quad = chaos_quad.legendre_proxy(int_pnts, domain =(-1,1))
        xs_quad = xs_quad[0]
        for ix, x in enumerate(self.xlist):
            self.c_Pn_target[ix] = self.integrate_coeffs_4d_2(self.a1 * 0, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x, xs_quad, ws_quad, xs_quad, ws_quad)
            # print(ix/self.xlist.size * 100, "percent complete")
            # print(x, 'x')
            # self.c_Pn_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, He, blank_mat_pn, np.inf, x)
            # self.c_Pn_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            # self.c_Pn_all[ix] = self.integrate_coeffs_4d(self.a1, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_all[ix] = self.integrate_coeffs_4d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            # self.c_Pn_target[ix] = self.integrate_coeffs_3d(self.a1, self.a2, self.a3, self.a3, M, Pn, blank_mat_pn, 1, x, xs_quad, ws_quad, xs_quad, ws_quad)
        print('elapsed', time.perf_counter()-tick)
        self.factors(M)

    def make_drive_coefficients_pn(self, M, int_pnts = 40):
        tick = time.perf_counter()
        NPn = int(M)
        NHe = int(M)
        blank_mat_pn = np.zeros((M+1, M+1, M+1, M+1))
        self.c_Pn_drive = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        xs_quad, ws_quad = chaos_quad.legendre_proxy(int_pnts, domain =(-1,1))
        xs_quad = xs_quad[0]
  
        for ix, x in enumerate(tqdm(self.xlist)):
            # print(ix/self.xlist.size * 100, "percent complete")
            # print(x, 'x')
            # self.c_Pn_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, He, blank_mat_pn, np.inf, x)
            self.c_Pn_drive[ix] = self.integrate_coeffs_1d(self.a1, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x, xs_quad, ws_quad, xs_quad, ws_quad)

        print('elapsed', time.perf_counter()-tick)
        self.factors(M)

    def make_all_coefficients_he(self, M, int_pnts = 40):
        tick = time.perf_counter()
        NPn = int(M)
        NHe = int(M)
        self.order_he = int_pnts
        blank_mat_pn = np.zeros((M+1, M+1, M+1, M+1))
        self.c_He_drive = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        self.c_He_target = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        self.c_He_all = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        xs_quad, ws_quad = chaos_quad.hermite(int_pnts)
        xs_quad = xs_quad[0]
        xs_quad_pn, ws_quad_pn = chaos_quad.legendre_proxy(25, domain =(-1,1))
        xs_quad_pn = xs_quad_pn[0]

        # ws_quad = ws_quad[0]
        for ix, x in enumerate(self.xlist_h):
            print(ix/self.xlist.size * 100, "percent complete")
            # print(x, 'x')
            # self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, He, blank_mat_pn, np.inf, x)
            # self.c_He_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            # self.c_He_all[ix] = self.integrate_coeffs_4d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)

            #current
            self.c_He_all[ix] = self.integrate_coeffs_4d_2(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, 1, x, xs_quad_pn, ws_quad_pn, xs_quad, ws_quad)
            # self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, 1, x, xs_quad, ws_quad)
            # self.c_He_target[ix] = self.integrate_coeffs_3d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, 1, x, xs_quad, ws_quad)
       
        print('elapsed', time.perf_counter()-tick)
        self.factors(M)
        
    
    def make_drive_coefficients_he(self, M, int_pnts = 40):
        tick = time.perf_counter()
        NPn = int(M)
        NHe = int(M)
        self.order_he = int_pnts
        blank_mat_pn = np.zeros((M+1, M+1, M+1, M+1))
        self.c_He_drive = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        self.c_He_target = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        self.c_He_all = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        xs_quad, ws_quad = chaos_quad.hermite(int_pnts)
        xs_quad = xs_quad[0]
        xs_quadpn, ws_quadpn = chaos_quad.legendre_proxy(int_pnts, domain =(-1,1))
        xs_quadpn = xs_quadpn[0]
        # ws_quad = ws_quad[0]
        for ix, x in enumerate(self.xlist_h):
            # print(ix/self.xlist.size * 100, "percent complete")
            # print(x, 'x')
            # self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, He, blank_mat_pn, np.inf, x)
            # self.c_He_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            # self.c_He_all[ix] = self.integrate_coeffs_4d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, 1, x, xs_quadpn, ws_quadpn, xs_quad, ws_quad)
       
        print('elapsed', time.perf_counter()-tick)
        # self.factors(M)
        

    def make_target_coefficients_he(self, M, int_pnts = 40):
        tick = time.perf_counter()
        NPn = int(M)
        NHe = int(M)
        self.order_he = int_pnts
        blank_mat_pn = np.zeros((M+1, M+1, M+1, M+1))
        self.c_He_drive = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        self.c_He_target = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        self.c_He_all = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        xs_quad, ws_quad = chaos_quad.hermite(int_pnts)
        xs_quad = xs_quad[0]
        xs_quadpn, ws_quadpn = chaos_quad.legendre_proxy(int_pnts, domain =(-1,1))
        xs_quadpn = xs_quadpn[0]
        # ws_quad = ws_quad[0]
        for ix, x in enumerate(self.xlist_h):
            # print(ix/self.xlist.size * 100, "percent complete")
            # print(x, 'x')
            # self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, He, blank_mat_pn, np.inf, x)
            # self.c_He_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            # self.c_He_all[ix] = self.integrate_coeffs_4d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            self.c_He_target[ix] = self.integrate_coeffs_3d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, 1, x, xs_quadpn, ws_quadpn, xs_quad, ws_quad)
        print('elapsed', time.perf_counter()-tick)
        self.factors(M)
        
    
    def factors(self, M):
        sqrt_2pi = math.sqrt(2*math.pi)
        for ix in range(self.xlist.size):
            for i in range(M+1):
                factorpn1 = (2 * i + 1) / 2
                factorhe1 = 1 / (math.factorial(i))
                self.c_Pn_drive[ix][i, 0, 0, 0] = self.c_Pn_drive[ix][i, 0, 0, 0] * factorpn1
                self.c_He_drive[ix][i, 0, 0, 0] = self.c_He_drive[ix][i, 0, 0, 0] * factorhe1
                for k in range(M+1):
                    for j in range(M+1):
                        factorpn3 = (2 * i + 1) * (2 * j + 1) * (2 * k + 1)  / 8
                        factorhe3 = 1 / (math.factorial(i)* math.factorial(j)* math.factorial(k))
                        self.c_Pn_target[ix][i, j, k, 0] = self.c_Pn_target[ix][i, j, k, 0] * factorpn3
                        self.c_He_target[ix][i, j, k, 0] = self.c_He_target[ix][i, j, k, 0] * factorhe3
                        for l in range(M+1):
                            factorpn4 = (2 * i + 1) * (2 * j + 1) * (2 * k + 1) * (2 * l + 1) / 16
                            factorhe4 = 1 / (math.factorial(i)* math.factorial(j)* math.factorial(k)* math.factorial(l))
                            self.c_Pn_all[ix][i, j, k, l] = self.c_Pn_all[ix][i, j, k, l] * factorpn4
                            self.c_He_all[ix][i, j, k, l] = self.c_He_all[ix][i, j, k, l] * factorhe4



    def save_pn_drive(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('coeffs_4d_drive_pn.hdf5', 'w')
        dset1 = f.create_dataset(f"drive_Pn", data = self.c_Pn_drive)
        # dset2 = f.create_dataset(f"target_Pn_{self.order_pn}", data = self.c_Pn_target)
        # dset3 = f.create_dataset(f"all_Pn_{self.order_pn}", data = self.c_Pn_all)
        dset4 = f.create_dataset("xlist", data = self.xlist)
        f.close()
    def save_pn_target(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('coeffs_4d_target_pn.hdf5', 'w')
        # dset1 = f.create_dataset(f"drive_Pn_{self.order_pn}", data = self.c_Pn_drive)
        dset2 = f.create_dataset(f"target_Pn", data = self.c_Pn_target)
        # dset3 = f.create_dataset(f"all_Pn_{self.order_pn}", data = self.c_Pn_all)
        dset4 = f.create_dataset("xlist", data = self.xlist)
        f.close()
    def save_pn_all(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('coeffs_4d_all_pn.hdf5', 'w')
        # dset1 = f.create_dataset(f"drive_Pn_{self.order_pn}", data = self.c_Pn_drive)
        # dset2 = f.create_dataset(f"target_Pn_{self.order_pn}", data = self.c_Pn_target)
        dset3 = f.create_dataset(f"all_Pn", data = self.c_Pn_all)
        dset4 = f.create_dataset("xlist", data = self.xlist)
        f.close()

    def save_he_drive(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('coeffs_4d_drive_he.hdf5', 'w')
        dset1 = f.create_dataset(f"drive_He", data = self.c_He_drive)
        # dset1 = f.create_dataset(f"target_He_{self.order_he}", data = self.c_He_target)
        # dset3 = f.create_dataset(f"all_He_{self.order_he}", data = self.c_He_all)
        dset4 = f.create_dataset("xlist", data = self.xlist_h)
        f.close()
    def save_he_target(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('coeffs_4d_target_he.hdf5', 'w')
        # dset1 = f.create_dataset(f"drive_He_{self.order_he}", data = self.c_He_drive)
        dset1 = f.create_dataset(f"target_He", data = self.c_He_target)
        # dset3 = f.create_dataset(f"all_He_{self.order_he}", data = self.c_He_all)
        dset4 = f.create_dataset("xlist", data = self.xlist_h)
        f.close()

    def save_he_all(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('coeffs_4d_all_he.hdf5', 'w')
        # dset1 = f.create_dataset(f"drive_He_{self.order_he}", data = self.c_He_drive)
        # dset1 = f.create_dataset(f"target_He_{self.order_he}", data = self.c_He_target)
        dset3 = f.create_dataset(f"all_He", data = self.c_He_all)
        dset4 = f.create_dataset("xlist", data = self.xlist_h)
        f.close()
    
    def save_marshak_sol(self):
        f = h5py.File('marshak_sol.hdf5', 'w')
        dset1 = f.create_dataset("xi", data = self.marshak_sol.t)
        dset1 = f.create_dataset("T", data = self.marshak_sol.y[0])
        f.close()

    def monte_carlo_sampler_pn(self, n):
        self.mc_drive_pn_samples = np.zeros((self.xlist.size, 2**n))
        self.mc_target_pn_samples = np.zeros((self.xlist.size, 2**n))
        self.mc_all_pn_samples = np.zeros((self.xlist.size, 2**n))
        sampler = qmc.Sobol(d=4, scramble=False)
        sample = sampler.random_base2(m=n)


        for ix, x in enumerate(tqdm(self.xlist)):
            self.mc_drive_pn_samples[ix] = monte_carlo_pn(sample, n, x, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, self.a1, 0, 0, 0, self.t, self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
            self.mc_target_pn_samples[ix] = monte_carlo_pn(sample, n, x, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, 0, self.a2, self.a3, self.a4, self.t, self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
            self.mc_all_pn_samples[ix] = monte_carlo_pn(sample, n, x, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, self.a1, self.a2, self.a3, self.a4, self.t, self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
        self.save_mc_pn_samples()

        # Hermite samples
        self.mc_drive_he_samples = np.zeros((self.xlist.size, 2**n))
        self.mc_target_he_samples = np.zeros((self.xlist.size, 2**n))
        self.mc_all_he_samples = np.zeros((self.xlist.size, 2**n))
        sample = np.random.normal(size = (2**n, 4))
        for ix, x in enumerate(tqdm(self.xlist)):
            self.mc_drive_he_samples[ix] = monte_carlo_he(sample, n, x, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, self.a1, 0, 0, 0, self.t, self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
            self.mc_target_he_samples[ix] = monte_carlo_he(sample, n, x, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, 0, self.a2, self.a3, self.a4, self.t, self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
            self.mc_all_he_samples[ix] = monte_carlo_he(sample, n, x, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, self.a1, self.a2, self.a3, self.a4, self.t, self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
        self.save_mc_he_samples()



    def save_mc_pn_samples(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('mc_samples_pn.hdf5', 'w')
        quantile_dat = np.zeros((5, self.xlist.size))
        quantile_dat_targ = np.zeros((5, self.xlist.size))
        quantile_dat_all = np.zeros((5, self.xlist.size))
        for ix, xx in enumerate(self.xlist):
            quantile_dat[0, ix] = np.quantile(self.mc_drive_pn_samples[ix], 0.5)
            quantile_dat[1, ix] = np.quantile(self.mc_drive_pn_samples[ix], 0.2)
            quantile_dat[2, ix] = np.quantile(self.mc_drive_pn_samples[ix], 0.8)
            quantile_dat[3, ix] = np.mean(self.mc_drive_pn_samples[ix])
            quantile_dat[4, ix] = np.std(self.mc_drive_pn_samples[ix])

            quantile_dat_targ[0, ix] = np.quantile(self.mc_target_pn_samples[ix], 0.5)
            quantile_dat_targ[1, ix] = np.quantile(self.mc_target_pn_samples[ix], 0.2)
            quantile_dat_targ[2, ix] = np.quantile(self.mc_target_pn_samples[ix], 0.8)
            quantile_dat_targ[3, ix] = np.mean(self.mc_target_pn_samples[ix])
            quantile_dat_targ[4, ix] = np.std(self.mc_target_pn_samples[ix])

            quantile_dat_all[0, ix] = np.quantile(self.mc_all_pn_samples[ix], 0.5)
            quantile_dat_all[1, ix] = np.quantile(self.mc_all_pn_samples[ix], 0.2)
            quantile_dat_all[2, ix] = np.quantile(self.mc_all_pn_samples[ix], 0.8)
            quantile_dat_all[3, ix] = np.mean(self.mc_all_pn_samples[ix])
            quantile_dat_all[4, ix] = np.std(self.mc_all_pn_samples[ix])
        # dset1 = f.create_dataset(f"drive_He_{self.order_he}", data = self.c_He_drive)
        # dset1 = f.create_dataset(f"target_He_{self.order_he}", data = self.c_He_target)
        dset3 = f.create_dataset(f"drive", data = quantile_dat)
        dset4 = f.create_dataset(f"target", data = quantile_dat_targ)
        dset5 = f.create_dataset(f"all", data = quantile_dat_all)
        f.close()

    

    def save_mc_he_samples(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('mc_samples_he.hdf5', 'w')
        quantile_dat = np.zeros((4, self.xlist.size))
        quantile_dat_targ = np.zeros((3, self.xlist.size))
        quantile_dat_all = np.zeros((3, self.xlist.size))
        for ix, xx in enumerate(self.xlist):
            quantile_dat[0, ix] = np.quantile(self.mc_drive_he_samples[ix], 0.5)
            quantile_dat[1, ix] = np.quantile(self.mc_drive_he_samples[ix], 0.2)
            quantile_dat[2, ix] = np.quantile(self.mc_drive_he_samples[ix], 0.8)
            quantile_dat[3, ix] = np.mean(self.mc_drive_he_samples[ix])


            quantile_dat_targ[0, ix] = np.quantile(self.mc_target_he_samples[ix], 0.5)
            quantile_dat_targ[1, ix] = np.quantile(self.mc_target_he_samples[ix], 0.2)
            quantile_dat_targ[2, ix] = np.quantile(self.mc_target_he_samples[ix], 0.8)

            quantile_dat_all[0, ix] = np.quantile(self.mc_all_he_samples[ix], 0.5)
            quantile_dat_all[1, ix] = np.quantile(self.mc_all_he_samples[ix], 0.2)
            quantile_dat_all[2, ix] = np.quantile(self.mc_all_he_samples[ix], 0.8)
        # dset1 = f.create_dataset(f"drive_He_{self.order_he}", data = self.c_He_drive)
        # dset1 = f.create_dataset(f"target_He_{self.order_he}", data = self.c_He_target)
        dset3 = f.create_dataset(f"drive", data = quantile_dat)
        dset4 = f.create_dataset(f"target", data = quantile_dat_targ)
        dset5 = f.create_dataset(f"all", data = quantile_dat_all)
        f.close()


    # def plot():


        




    
    

    
        
