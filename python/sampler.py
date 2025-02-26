import numpy as np
from .functions import Pn, He, make_expansion_1d, sampler_sobol, sampler_normal
from .functions import sampler_sobol_1d, sampler_sobol_3d, sampler_sobol_4d
from .functions import sampler_normal_1d, sampler_normal_3d, sampler_normal_4d
import math
import time
from scipy.stats import qmc
import h5py 
import matplotlib.pyplot as plt
from .functions import b_prod
from tqdm import tqdm

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
    f2 = h5py.File('coeffs_4d_drive_pn.hdf5', 'r+')
    drive_coeffs_pn = f2['drive_Pn']
    f3 = h5py.File('coeffs_4d_target_pn.hdf5', 'r+')
    target_coeffs_pn = f3['target_Pn']
    xlist = f2['xlist']
    xs = np.array(xlist)
    a1 = (np.array(all_coeffs_pn))
    a2 = (np.array(target_coeffs_pn))
    a3 =  (np.array(drive_coeffs_pn))
    f1.close()
    f2.close()
    f3.close()
    NN = len(a1[0]-1)
    print(NN, "M")
    n = int(math.log2(N_samples))
    sampler = qmc.Sobol(d=4, scramble=False)
    sample = sampler.random_base2(m=n)

    samples_all = np.zeros((xs.size, 2**n))
    samples_drive = np.zeros((xs.size, 2**n))
    samples_target = np.zeros((xs.size, 2**n))
    for ix, xx in enumerate(tqdm(xs)):
        # if ix/len(xs)*100%10==0:
        #     print(ix/len(xs)*100, 'percent complete')
        # samples_all[ix] = sampler_sobol_4d(n, a1[ix], NN, sample)
        # print(a3[ix][0,0,0,0])
        samples_drive[ix] = sampler_sobol_1d(n, a3[ix], NN, sample)
        # samples_target[ix] = sampler_sobol_3d(n, a2[ix], NN, sample)
        # samples_all[ix] = sampler_sobol_4d(n, a1[ix], NN, sample)

        # if np.any(samples_drive[ix] > 0.44):
        #     print(np.max(samples_drive[ix]), ix)
        # if a3[ix][0,0,0,0] > 0.44:
        #     print(a3[ix][0,0,0,0], ix)
        # if a3[ix][1,0,0,0] > a3[ix][0,0,0,0] + 0.4:
        #     print(ix)
        #     print(np.any(samples_drive[ix] > 0.44))

        # samples_target[ix] = sampler_sobol_3d(n, a2[ix], NN, sample)


    save_T_samples(xs, samples_all, 'all_Pn_samples.hdf5')
    save_T_samples(xs, samples_target, 'target_Pn_samples.hdf5')
    save_T_samples(xs, samples_drive, 'drive_Pn_samples.hdf5')

    print(np.mean(samples_all[0]), 'should be close to 0.4')

def sample_T_he(N_samples = 1024):
    # load coefficients
    f1 = h5py.File('coeffs_4d_all_he.hdf5', 'r+')
    all_coeffs_he = f1['all_He']
    f2 = h5py.File('coeffs_4d_drive_he.hdf5', 'r+')
    drive_coeffs_he = f2['drive_He']
    f3 = h5py.File('coeffs_4d_target_he.hdf5', 'r+')
    target_coeffs_he = f3['target_He']
    xlist = f2['xlist']
    xs = np.array(xlist)
    a1 = (np.array(all_coeffs_he))
    a2 = (np.array(target_coeffs_he))
    a3 =  (np.array(drive_coeffs_he))
    f1.close()
    f2.close()
    f3.close()
    # print(a1[0,0])
    
    NN = len(a3[0])
    print(NN-1, "M")
    n = int(math.log2(N_samples))
    sampler = qmc.Sobol(d=4, scramble=False)
    sample = np.random.normal(size = (2**n, 4))

    samples_all = np.zeros((xs.size, 2**n))
    samples_drive = np.zeros((xs.size, 2**n))
    samples_target = np.zeros((xs.size, 2**n))
    for ix, xx in enumerate(tqdm(xs)):
        # if ix/len(xs)*100%10==0:
        #     print(ix/len(xs)*100, 'percent complete')
        # samples_all[ix] = sampler_normal_4d(n, a1[ix], NN, sample)
        samples_drive[ix] = sampler_normal_1d(n, a2[ix], NN, sample)
        samples_target[ix] = sampler_normal_3d(n, a3[ix], NN, sample)

    save_T_samples(xs, samples_all, 'all_He_samples.hdf5')
    save_T_samples(xs, samples_target, 'target_He_samples.hdf5')
    save_T_samples(xs, samples_drive, 'drive_He_samples.hdf5')

    print(np.mean(samples_all[0]), 'should be close to 0.4')
    print(np.mean(samples_target[0]), 'should be close to 0.4')
    print(np.mean(samples_drive[0]), 'should be close to 0.4')

def save_results(drive, target, alls, name):
    # np.savetxt(name, sample_list)
    f = h5py.File(name, 'w')
    dset = f.create_dataset("results", (3,len(drive)))
    dset[0] = drive
    dset[1] = target
    dset[2] = alls
    f.close()


def save_T_samples(xs, samples, name):
    # np.savetxt(name, sample_list)
    quantile_dat = np.zeros((4, xs.size))
    for ix, xx in enumerate(xs):
        quantile_dat[0, ix] = np.quantile(samples[ix], 0.5)
        quantile_dat[1, ix] = np.quantile(samples[ix], 0.2)
        quantile_dat[2, ix] = np.quantile(samples[ix], 0.8)
        quantile_dat[3, ix] = np.mean(samples[ix])
    f = h5py.File(name, 'w')
    dset = f.create_dataset("samples", data = quantile_dat)
    f.close()

def save_T_samples2(xlist, samples, name):
    # np.savetxt(name, sample_list)
    f = h5py.File(name, 'w')
    save_data = np.zeros((4, xlist.size))
    dset = f.create_dataset("PCE_UQ/samples", data = samples)
    for ix, xx in enumerate(xlist):
        save_data[0, ix] = np.quantile(samples[ix], 0.5)
        save_data[1, ix] = np.quantile(samples[ix], 0.2)
        save_data[2, ix] = np.quantile(samples[ix], 0.8)
        save_data[3, ix] = np.mean(samples[ix])


    f.close()












    
