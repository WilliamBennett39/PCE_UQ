from python.sampler import sample_breakout_u, sample_breakout_n
from python.expansion_temp import coefficients_4d
from python import interpolation_experiment
import numpy as np

def rmse(list1, list2):
    return np.mean(np.sqrt((list1-list2)**2))

ob = coefficients_4d()
M=0
order = 32
err = 10
tol = 1e-5
# ob.make_all_coefficients_he(0, 0)
ob.make_all_coefficients_he(M, 2)
M_list = [0,1,2,3,4,5,6,7,8,9,10]
coeffs_old = ob.c_He_all
it = 0
while err > tol:
    M = M_list[it]
    print(order, 'order')
    ob.make_all_coefficients_he(M, order)
    coeffs_new = ob.c_He_all
    # err = 10
    print(err, 'error')
    coeffs_old = coeffs_new
    ob.save_he_all()
    it += 1