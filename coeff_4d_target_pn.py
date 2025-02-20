from python.sampler import sample_breakout_u, sample_breakout_n
from python.expansion_temp import coefficients_4d
from python import interpolation_experiment
import numpy as np

def rmse(list1, list2):
    return np.mean(np.sqrt((list1-list2)**2))

ob = coefficients_4d()
M = 0
order = 128
err = 10
tol = 1e-8
ob.make_target_coefficients_pn(M, 2)

coeffs_old = ob.c_Pn_target
order_list = [0]
it = 0
while err > tol and it < len(order_list):
    # order = order * 2
    M = order_list[it]
    print(M, 'M')
    print(err, 'RMSE')
    ob.make_target_coefficients_pn(M, order)
    coeffs_new = ob.c_Pn_target
    # err = rmse(coeffs_new, coeffs_old)
    coeffs_old = coeffs_new
    ob.save_pn_target()
    it += 1