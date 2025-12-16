
from python.sampler import sample_breakout_u, sample_breakout_n, sample_T_pn, sample_T_he 
from python.expansion_temp import coefficients_4d
from python import interpolation_experiment

ob = coefficients_4d()
def MC_samples(n=8):

    ob.monte_carlo_sampler_pn(n) # this actually samples he also
    

MC_samples(24)
