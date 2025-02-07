from python.sampler import sample_breakout_u, sample_breakout_n
from python.expansion_temp import coefficients_4d
from python import interpolation_experiment
import numpy as np

def rmse(list1, list2):
    return np.mean(np.sqrt((list1-list2)**2))

ob = coefficients_4d()
M=6
order = 4
err = 10
tol = 1e-5
# ob.make_all_coefficients_he(0, 0)
ob.make_all_coefficients_he(M, order)
coeffs_old = ob.c_He_all
while err > tol:
    order = order +4
    print(order, 'order')
    ob.make_all_coefficients_he(M, order)
    coeffs_new = ob.c_He_all
    err = rmse(coeffs_new, coeffs_old)
    print(err, 'error')
    coeffs_old = coeffs_new
    ob.save_he_all()