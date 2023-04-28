import numpy as np
from .functions import Pn, He, make_expansion_1d, sampler_sobol, sampler_normal
from .functions import sampler_sobol_1d, sampler_sobol_3d, sampler_sobol_4d
import math
import time
from scipy.stats import qmc
import h5py 

def sample_breakout_u(NSAMPLES=1024):
    tic = time.perf_counter()
    xdetector = 0.3

    drivecoeffs = np.loadtxt('drivecoeffs.csv',delimiter=",", dtype=float)
    targetcoeffs = np.loadtxt('targetcoeffs.csv',delimiter=",", dtype = float)
    allcoeffs = np.loadtxt('allcoeffs.csv',delimiter=",", dtype = float)

    NN = len(allcoeffs[0])
    n = int(math.log2(NSAMPLES))
    sampler = qmc.Sobol(d=4, scramble=False)
    sample = sampler.random_base2(m=n)
    samples_drive_pn = sampler_sobol(n, drivecoeffs, NN, sample) * xdetector ** 2 
    samples_target_pn = sampler_sobol(n, targetcoeffs, NN, sample) * xdetector ** 2 
    samples_all_pn = sampler_sobol(n, allcoeffs, NN, sample) * xdetector ** 2 

    print(np.mean(samples_drive_pn), 'mean drive')
    print(np.mean(samples_target_pn), 'mean target')
    print(np.mean(samples_all_pn), 'mean all')
    
    save_results(samples_drive_pn, samples_target_pn, samples_all_pn, "pn_samples.hdf5")
    
    print('-- -- -- -- -- -- -- -- ')
    print(2**n, 'samples')
    print(time.perf_counter()-tic,"time elapsed")
    print('-- -- -- -- -- -- -- -- ')



    return samples_drive_pn,  samples_target_pn,samples_all_pn

def sample_breakout_n(NSAMPLES=1024):
    tic = time.perf_counter()
    xdetector = 0.3

    drivecoeffs = np.loadtxt('drivecoeffs_he.csv',delimiter=",", dtype=float)
    targetcoeffs = np.loadtxt('targetcoeffs_he.csv',delimiter=",", dtype = float)
    allcoeffs = np.loadtxt('allcoeffs_he.csv',delimiter=",", dtype = float)

    NN = len(allcoeffs[0])
    n = int(math.log2(NSAMPLES))
    np.random.seed(0)
    sample = np.random.normal(size = (2**n, 4))

    samples_drive_he = sampler_normal(n, drivecoeffs, NN, sample) * xdetector ** 2 
    samples_target_he = sampler_normal(n, targetcoeffs, NN, sample) * xdetector ** 2 
    samples_all_he = sampler_normal(n, allcoeffs, NN, sample) * xdetector ** 2 


    print(np.mean(samples_drive_he), 'mean drive')
    print(np.mean(samples_target_he), 'mean target')
    print(np.mean(samples_all_he), 'mean all')
    
    save_results(samples_drive_he, samples_target_he, samples_all_he, "he_samples.hdf5")

    
    print('-- -- -- -- -- -- -- -- ')
    print(2**n, 'samples')
    print(time.perf_counter()-tic,"time elapsed")
    print('-- -- -- -- -- -- -- -- ')



    return samples_drive_he,  samples_target_he,samples_all_he




def sample_T_pn(N_samples = 1024):
    # load coefficients
    f1 = h5py.File('coeffs_4d_all_pn.hdf5', 'r+')
    all_coeffs_pn = f1['all_Pn']
    xlist = f1['xlist']
    xs = np.array(xlist)
    a1 = (np.array(all_coeffs_pn))/16
    f1.close()
    
    NN = len(a1[0])
    print(NN, "M")
    n = int(math.log2(N_samples))
    sampler = qmc.Sobol(d=4, scramble=False)
    sample = sampler.random_base2(m=n)

    samples_all = np.zeros((xs.size, 2**n))
    for ix, xx in enumerate(xs):
        samples_all[ix] = sampler_sobol_4d(n, a1[ix], NN, sample)
    save_T_samples(samples_all, 'all_Pn_samples.hdf5')
    print(np.mean(samples_all[0]), 'should be close to 0.4')





def save_results(drive, target, alls, name):
    # np.savetxt(name, sample_list)
    f = h5py.File(name, 'w')
    dset = f.create_dataset("results", (3,len(drive)))
    dset[0] = drive
    dset[1] = target
    dset[2] = alls
    f.close()


def save_T_samples(samples, name):
    # np.savetxt(name, sample_list)
    f = h5py.File(name, 'w')
    dset = f.create_dataset("samples", data = samples)

    f.close()












    
