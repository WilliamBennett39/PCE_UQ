from python.sampler import sample_breakout_u, sample_breakout_n
from python.expansion_temp import coefficients_4d
from python import interpolation_experiment
import numpy as np

def rmse(list1, list2):
    return np.mean(np.sqrt((list1-list2)**2))

ob = coefficients_4d()
M = 0
order = 8
err = 10
tol = 1e-8
# ob.make_all_coefficients_pn(M, 2)
ob.make_all_coefficients_pn(M, 2)
# ob.save_pn_all()
coeffs_old = ob.c_Pn_all
M_list = [7]
order_list = [15]
it = 0
while err > tol:
    M = M_list[it]
    order = order_list[it]
    print(M, 'M')
    ob.make_all_coefficients_pn(M, order)
    coeffs_new = ob.c_Pn_all
    ob.save_pn_all()
    # err = rmse(coeffs_new, coeffs_old)
    print(err, 'RMSE')
    coeffs_old = coeffs_new
    it += 1