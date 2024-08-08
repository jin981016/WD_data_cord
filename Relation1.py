#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.odr import *
from scipy import linalg
from scipy import stats
from scipy import optimize
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

All = pd.read_csv(r'/home/jin/WD/all.txt', sep = '\s+', header=0)

plt.rcParams['font.size'] = '16'
T, R, M, Rerr, Merr, Terr = All['Teff'].to_numpy(), All['Radius'].to_numpy(), All['Mass'].to_numpy(), All['Rerr'].to_numpy(), All['Merr'].to_numpy(), All['Terr'].to_numpy()

def task_multirun(pp):
    np.random.seed(seed=pp)
    new_M = np.zeros(len(Merr))
    new_R = np.zeros(len(Rerr))
    new_T = np.zeros(len(Terr))
    for i, dM in enumerate(Merr):
        delM = stats.norm.rvs(loc=0, scale=dM)
        if (delM + M[i]) <= 0:
            delM = 0.5 * delM
        new_M[i] = M[i] + delM
    for i, dR in enumerate(Rerr):
        delR = stats.norm.rvs(loc=0, scale=dR)
        if (delR + R[i]) <= 0:
            delR = 0.5 * delR
        new_R[i] = R[i] + delR
        
    for i, dT in enumerate(Terr):
        delT = stats.norm.rvs(loc=0, scale=dT)
        if (delT + T[i]) <= 0:
            delT = 0.5 * delT
        new_T[i] = T[i] + delT

    log_M = np.log10(new_M)
    log_R = np.log10(new_R)
    x = np.row_stack((log_M, new_T))
    N = 100
    
    TT = np.logspace(-5, 5, N)
    cc = np.linspace(0, 1, N)

    resM1_chi = np.zeros((N, N))
    beta1_chi = np.zeros(N)

    resM1_std = np.zeros((N, N))
    beta1_std = np.zeros(N)

    a_all_Model1 = np.zeros(N)
    c_T_all_Model1 = np.zeros(N)
    b_M_all_Model1 = np.zeros(N)
    for ii, t0 in tqdm(enumerate(TT)):
        def linfit_Model1_chi(beta, x):
            return beta[0] + beta[1] * x[0] + beta[2] * np.log10(x[1] / t0)

        data = RealData(x, log_R)
        linmod_Model1 = Model(linfit_Model1_chi)
        odr_model1 = ODR(data, linmod_Model1, beta0=[1., 1., 1.])
        out_model1 = odr_model1.run()

        a_all_Model1[ii], b_M_all_Model1[ii], c_T_all_Model1[ii] = out_model1.beta

        for jj, beta in enumerate(cc):
            R_test_Model1 = np.log10(new_R / (new_T / t0) ** beta)

            resM1_chi[ii, jj] = np.sum((np.log10(R) - beta * np.log10(T / t0) - b_M_all_Model1[ii] * log_M - a_all_Model1[ii]) ** 2)
            resM1_std[ii, jj] = np.std(R_test_Model1 - b_M_all_Model1[ii] * log_M)
        mm1_std = resM1_std[ii, :].argmin()
        beta1_std[ii] = cc[mm1_std]

        mm1_chi = resM1_chi[ii, :].argmin()
        beta1_chi[ii] = cc[mm1_chi]

    tt_m1_std = np.unravel_index(resM1_std.argmin(), resM1_std.shape)
    qq, beta0_m1_std = tt_m1_std

    tt_m1_chi = np.unravel_index(resM1_chi.argmin(), resM1_chi.shape)
    kk, beta0_m1_chi = tt_m1_chi
    
    R_m1_std = a_all_Model1[qq]
    alpha_m1_std = b_M_all_Model1[qq]
    beta_m1_std = cc[beta0_m1_std]
    beta_m1_std_odr = c_T_all_Model1[qq]
    t_m1_std = TT[qq]
    res_mm1_std = resM1_chi[qq, beta0_m1_std]
    
    Relation_1_std = np.array([[R_m1_std, alpha_m1_std, beta_m1_std,beta_m1_std_odr,t_m1_std, res_mm1_std]])
    
    R_m1_chi = a_all_Model1[kk]
    alpha_m1_chi = b_M_all_Model1[kk]
    beta_m1_chi = cc[beta0_m1_chi]
    beta_m1_chi_odr = c_T_all_Model1[kk]
    t_m1_chi = TT[kk]
    res_mm1_chi = resM1_chi[kk, beta0_m1_chi]
    
    Relation_1_chi = np.array([[R_m1_chi, alpha_m1_chi, beta_m1_chi,beta_m1_chi_odr,t_m1_chi, res_mm1_chi]])
    
    np.savetxt('/home/jin/WD/data_chi/{}_number_data.txt'.format(pp), Relation_1_chi, delimiter=',')
    np.savetxt('/home/jin/WD/data_std/{}_number_data.txt'.format(pp), Relation_1_std, delimiter=',')

if __name__ == '__main__':
    iteration = 50
    num_processes = min(cpu_count(), 10)  # Use at most 10 processes or the number of available CPU cores, whichever is smaller
    start = time.perf_counter()
    
    with Pool(processes=num_processes) as pool:
        pool.map(task_multirun, range(1, iteration + 1))
    
    finish = time.perf_counter()
    print(f'{round(finish-start, 2)} seconds to complete the task')

