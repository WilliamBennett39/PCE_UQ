from python.sampler import sample_breakout_u, sample_breakout_n
from python.expansion_temp import coefficients_4d
from python import interpolation_experiment
import numpy as np

def rmse(list1, list2):
    return np.mean(np.sqrt((list1-list2)**2))

ob = coefficients_4d()
order = 2
err = 10
tol = 1e-4
ob.make_all_coefficients_pn(0, order)
coeffs_old = ob.c_Pn_all
while err > tol:
    order += 2
    ob.make_all_coefficients_pn(0, order)
    coeffs_new = ob.c_Pn_all
    err = rmse(coeffs_new, coeffs_old)
    coeffs_old = coeffs_new

ob.save_pn(order)