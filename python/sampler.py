import numpy as np
from .functions import Pn, He, make_expansion_1d, sampler_sobol, sampler_normal
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

def save_results(drive, target, alls, name):
    # np.savetxt(name, sample_list)
    f = h5py.File(name, 'w')
    dset = f.create_dataset("results", (3,len(drive)))
    dset[0] = drive
    dset[1] = target
    dset[2] = alls
    f.close()












    
