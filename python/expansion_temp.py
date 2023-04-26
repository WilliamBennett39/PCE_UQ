import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import math
import h5py
import time
# import quadpy

from .marshak_ode import solve_marshak
from .functions import b_prod, Afunc, Pn, He, interpolated_T, nb_integrand, nb_integrand_2, nb_integrand_He, nb_integrand_2_He
from .functions import integrate_quad, quadruple_integral_nb_4, F1,  quadruple_integral_nb_1
from .functions import quadruple_integral_nb_h4
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
        self.xlist = np.linspace(0, self.xmaxmax, self.Npnts)
                
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
    
    def integrate_coeffs_4d_2(self, a1, a2, a3, a4, M, basis, coeffs, L, x, xs_pnts, xs_ws):
        dimensionalize = lambda x1, x2, x3, x4: (self.T0 + a1 * x1) * interpolated_T(x * Afunc(x1, x2, x3, x4, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4)/math.sqrt(self.t), self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
        integrand = lambda x1, x2, x3, x4, l1, l2, l3, l4: dimensionalize(x1, x2, x3, x4) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, basis)
        for i in range(M+1):
            for j in range(M+1):
                for k in range(M+1):
                    for m in range(M+1):
                        if basis == Pn:
                            # coeffs[i,j,k,m] = integrate.nquad(self.integrate_coeffs_temp, [[-L,L],[-L,L],[-L,L]], args = (a1, a2, a3, a4, M, basis, L, x, i, j, k, m), opts = [opts0, opts0, opts0])[0]
                            coeffs[i,j,k,m] = quadruple_integral_nb_4(xs_pnts, xs_ws, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m)
                        elif basis == He:
                            coeffs[i,j,k,m] = quadruple_integral_nb_h4(xs_pnts, xs_ws, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m)

        return coeffs

    def integrate_coeffs_1d(self, a1, a2, a3, a4, M, basis, coeffs, L, x):
        dimensionalize = lambda x1, x2, x3, x4: (self.T0 + a1 * x1) * interpolated_T(x * Afunc(x1, x2, x3, x4, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4)/math.sqrt(self.t), self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
        integrand = lambda x1, x2, x3, x4, l1, l2, l3, l4: dimensionalize(x1, x2, x3, x4) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, basis)
        j = 0
        k = 0 
        m = 0 
        # print(F1(0, 0, 0, 0, x, self.t, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, 0, 0, 0, 0))
        for i in range(M+1):
            if basis == Pn:
            # coeffs[i,j,k,m] = integrate.nquad(integrand, [[-L,L]], args = (0, 0, 0, i, j, k, m))[0]
                coeffs[i,j,k,m] = integrate.nquad(nb_integrand, [[-L,L]], args = (0, 0, 0, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m))[0]
            elif basis == He:
                coeffs[i,j,k,m] = integrate.nquad(nb_integrand_He, [[-L,L]], args = (0, 0, 0, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m))[0]
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


    def integrate_coeffs_3d(self, a1, a2, a3, a4, M, basis, coeffs, L, x):
        dimensionalize = lambda x1, x2, x3, x4: (self.T0 + a1 * x1) * interpolated_T(x * Afunc(x1, x2, x3, x4, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4)/math.sqrt(self.t), self.ximax, self.interp_t, self.interp_c, self.interp_k,self.interp_equi_spaced, self.interp_dx)
        integrand = lambda x2, x3, x4, x1, l1, l2, l3, l4: dimensionalize(x1, x2, x3, x4) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, basis)
        i = 0 
        for j in range(M+1):
            for k in range(M+1):
                for m in range(M+1):
                    if basis == Pn:
                    # coeffs[i,j,k,m] = integrate.nquad(integrand, [[-L,L], [-L, L], [-L,L]], args = (0, i, j, k, m), opts = [opts0, opts0, opts0])[0]
                        coeffs[i,j,k,m] = integrate.nquad(nb_integrand_2,[[-L,L], [-L, L], [-L,L]], args = (0, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m), opts = [opts0, opts0, opts0])[0]
                    elif basis == He:
                        coeffs[i,j,k,m] = integrate.nquad(nb_integrand_2_He,[[-L,L], [-L, L], [-L,L]], args = (0, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m), opts = [opts0, opts0, opts0])[0]
        return coeffs

    def make_all_coefficients_pn(self, M, int_pnts = 40):
        tick = time.perf_counter()
        NPn = int(M)
        NHe = int(M)
        blank_mat_pn = np.zeros((M+1, M+1, M+1, M+1))
        self.c_Pn_drive = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_Pn_target = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_Pn_all = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        # xs_quad = quadpy.c1.gauss_legendre(int_pnts).points
        # ws_quad = quadpy.c1.gauss_legendre(int_pnts).weights
        xs_quad, ws_quad = chaospy.quadrature.legendre_proxy(int_pnts, domain =(-1,1))
        xs_quad = xs_quad[0]
        ws_quad = ws_quad[0]
        print(xs_quad)
        for ix, x in enumerate(self.xlist):
            print(ix/self.xlist.size * 100, "percent complete")
            # print(x, 'x')
            self.c_Pn_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, He, blank_mat_pn, np.inf, x)
            # self.c_Pn_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            # self.c_Pn_all[ix] = self.integrate_coeffs_4d(self.a1, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_all[ix] = self.integrate_coeffs_4d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            self.c_Pn_all[ix] = self.integrate_coeffs_4d_2(self.a1, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x, xs_quad, ws_quad)
        print('elapsed', time.perf_counter()-tick)

    def make_all_coefficients_he(self, M, int_pnts = 40):
        tick = time.perf_counter()
        NPn = int(M)
        NHe = int(M)
        blank_mat_pn = np.zeros((M+1, M+1, M+1, M+1))
        self.c_He_drive = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        self.c_He_target = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        self.c_He_all = np.zeros((self.xlist.size, NHe + 1, NHe + 1, NHe + 1, NHe + 1))
        xs_quad, ws_quad = chaos_quad.hermite(int_pnts)
        xs_quad = xs_quad[0]
        ws_quad = ws_quad[0]
        for ix, x in enumerate(self.xlist):
            print(ix/self.xlist.size * 100, "percent complete")
            # print(x, 'x')
            self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, He, blank_mat_pn, np.inf, x)
            # self.c_He_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            # self.c_He_all[ix] = self.integrate_coeffs_4d(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, np.inf, x)
            self.c_He_all[ix] = self.integrate_coeffs_4d_2(self.a1, self.a2, self.a3, self.a4, M, He, blank_mat_pn, 1, x, xs_quad, ws_quad)
        print('elapsed', time.perf_counter()-tick)



    def make_drive_coefficients_pn(self, M, int_pnts = 40):
        tick = time.perf_counter()
        NPn = int(M)
        NHe = int(M)
        blank_mat_pn = np.zeros((M+1, M+1, M+1, M+1))
        self.c_Pn_drive = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        for ix, x in enumerate(self.xlist):
            print(ix/self.xlist.size * 100, "percent complete")
            # print(x, 'x')
            self.c_Pn_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, Pn, blank_mat_pn, 1, x)
            # self.c_He_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, He, blank_mat_pn, np.inf, x)
        print('elapsed', time.perf_counter()-tick)

    def save_pn(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('drive_coeffs_4d.hdf5', 'w')
        dset1 = f.create_dataset("drive_Pn", data = self.c_Pn_drive)
        dset2 = f.create_dataset("target_Pn", data = self.c_Pn_target)
        dset3 = f.create_dataset("all_Pn", data = self.c_Pn_all)
        dset4 = f.create_dataset("xlist", data = self.xlist)
        f.close()

    def save_he(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('drive_coeffs_4d.hdf5', 'w')
        dset1 = f.create_dataset("drive_He", data = self.c_He_drive)
        dset1 = f.create_dataset("target_He", data = self.c_He_target)
        dset3 = f.create_dataset("all_He", data = self.c_He_all)
        dset4 = f.create_dataset("xlist", data = self.xlist)
        f.close()
    
    def save_marshak_sol(self):
        f = h5py.File('marshak_sol.hdf5', 'w')
        dset1 = f.create_dataset("xi", data = self.marshak_sol.t)
        dset1 = f.create_dataset("T", data = self.marshak_sol.y[0])
        f.close()




    # def plot():


        




    
    

    
        
