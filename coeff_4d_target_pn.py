from python.sampler import sample_breakout_u, sample_breakout_n
from python.expansion_temp import coefficients_4d
from python import interpolation_experiment
import numpy as np

def rmse(list1, list2):
    return np.mean(np.sqrt((list1-list2)**2))

ob = coefficients_4d()
M = 8
order = 4
err = 10
tol = 1e-8
ob.make_target_coefficients_pn(M, order)
coeffs_old = ob.c_Pn_target

while err > tol:
    order = order + 4
    print(order, 'order')
    print(err, 'RMSE')
    ob.make_target_coefficients_pn(M, order)
    coeffs_new = ob.c_Pn_target
    err = rmse(coeffs_new, coeffs_old)
    coeffs_old = coeffs_new
    ob.save_pn_target()