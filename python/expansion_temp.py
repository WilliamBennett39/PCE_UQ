import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import math
import h5py
import time

from .marshak_ode import solve_marshak
from .functions import b_prod, Afunc, Pn, He, interpolated_T, nb_integrand, nb_integrand_2
from .interpolation_experiment import custom_splrep

def opts0(*args, **kwargs):
       return {'limit':2, 'epsabs':1.5e-8, 'epsrel':1.5e-8}

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
                        coeffs[i,j,k,m] = integrate.nquad(integrand, [[-L,L],[-L,L],[-L,L],[-L,L]], args = (i, j, k, m), opts = [opts0, opts0, opts0, opts0])[0]
        return coeffs

    def integrate_coeffs_1d(self, a1, a2, a3, a4, M, basis, coeffs, L, x):
        dimensionalize = lambda x1, x2, x3, x4: (self.T0 + a1 * x1) * interpolated_T(x * Afunc(x1, x2, x3, x4, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4)/math.sqrt(self.t), self.ximax, self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx)
        integrand = lambda x1, x2, x3, x4, l1, l2, l3, l4: dimensionalize(x1, x2, x3, x4) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, basis)
        j = 0
        k = 0 
        m = 0 
        for i in range(M+1):
            # coeffs[i,j,k,m] = integrate.nquad(integrand, [[-L,L]], args = (0, 0, 0, i, j, k, m))[0]
            coeffs[i,j,k,m] = integrate.nquad(nb_integrand, [[-L,L]], args = (0, 0, 0, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m))[0]
        return coeffs

    def integrate_coeffs_3d(self, a1, a2, a3, a4, M, basis, coeffs, L, x):
        dimensionalize = lambda x1, x2, x3, x4: (self.T0 + a1 * x1) * interpolated_T(x * Afunc(x1, x2, x3, x4, self.T0, self.kappa0, self.rho0, self.cv, self.omega, self.n, a1, a2, a3, a4)/math.sqrt(self.t), self.ximax, self.interp_t, self.interp_c, self.interp_k,self.interp_equi_spaced, self.interp_dx)
        integrand = lambda x2, x3, x4, x1, l1, l2, l3, l4: dimensionalize(x1, x2, x3, x4) * b_prod(x1, x2, x3, x4, l1, l2, l3, l4, basis)
        i = 0 
        for j in range(M+1):
            for k in range(M+1):
                for m in range(M+1):
                    print(j,k,m)
                    coeffs[i,j,k,m] = integrate.nquad(integrand, [[-L,L], [-L, L], [-L,L]], args = (0, i, j, k, m), opts = [opts0, opts0, opts0])[0]
                    coeffs[i,j,k,m] = integrate.nquad(nb_integrand_2,[[-L,L], [-L, L], [-L,L]], args = (0, x, self.t, self.T0, self.kappa0, self.rho0,self.cv, self.omega, self.n, a1, a2, a3, a4, self.ximax,  self.interp_t, self.interp_c, self.interp_k, self.interp_equi_spaced, self.interp_dx, i, j, k, m), opts = [opts0, opts0, opts0])[0]
        return coeffs

    def make_all_coefficients(self, M):
        tick = time.perf_counter()
        NPn = int(M)
        blank_mat_pn = np.zeros((M+1, M+1, M+1, M+1))
        self.c_Pn_drive = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_Pn_target = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        self.c_Pn_all = np.zeros((self.xlist.size, NPn + 1, NPn + 1, NPn + 1, NPn + 1))
        for ix, x in enumerate(self.xlist):
            print(ix/self.xlist.size * 100, "percent complete")
            print(x, 'x')
            self.c_Pn_drive[ix] = self.integrate_coeffs_1d(self.a1, 0, 0, 0, M, Pn, blank_mat_pn, 1, x)
            self.c_Pn_target[ix] = self.integrate_coeffs_3d(0, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x)
            self.c_Pn_all[ix] = self.integrate_coeffs_3d(self.a1, self.a2, self.a3, self.a4, M, Pn, blank_mat_pn, 1, x)
        print('elapsed', time.perf_counter()-tick)
    def save(self):
    # np.savetxt(name, sample_list)
        f = h5py.File('drive_coeffs_4d.hdf5', 'w')
        dset = f.create_dataset("drive_Pn", (self.xlist.size, self.NPn+1, self.NPn+1, self.NPn+1, self.NPn+1))
        dset = self.c_Pn_drive
        dset = f.create_dataset("target_Pn", (self.xlist.size, self.NPn+1, self.NPn+1, self.NPn+1, self.NPn+1))
        dset = self.c_Pn_target
        dset = f.create_dataset("all_Pn", (self.xlist.size, self.NPn+1, self.NPn+1, self.NPn+1, self.NPn+1))
        dset = self.c_Pn_all
        f.close()



    # def plot():


        




    
    

    
        
