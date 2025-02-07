from python.sampler import sample_breakout_u, sample_breakout_n
from python.expansion_temp import coefficients_4d
from python import interpolation_experiment
import numpy as np

def rmse(list1, list2):
    return np.mean(np.sqrt((list1-list2)**2))

ob = coefficients_4d()
order = 4
M=4
err = 10
tol = 1e-5
ob.make_target_coefficients_he(M, 0)
ob.make_target_coefficients_he(M, order)
coeffs_old = ob.c_He_target
order_list = [2,4,6,8,10]
it = 0
while err > tol:
    order = order_list[it]
    print(order, 'order')
    print(err, 'RMSE')
    ob.make_target_coefficients_he(M, order)
    coeffs_new = ob.c_He_target
    ob.save_he_target()
    err = rmse(coeffs_new, coeffs_old)
    coeffs_old = coeffs_new
    it += 1