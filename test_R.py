#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.odr import *
from scipy import stats
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 데이터 로드
All = pd.read_csv(r'/home/jin/WD/all.txt', sep='\s+', header=0)

# 전역 변수로 전환
T, R, M, Rerr, Merr, Terr = All['Teff'].to_numpy(), All['Radius'].to_numpy(), All['Mass'].to_numpy(), All['Rerr'].to_numpy(), All['Merr'].to_numpy(), All['Terr'].to_numpy()

# 메인 작업 함수
def task_multirun(pp):
    np.random.seed(seed=pp)
    
    # 벡터화된 난수 생성
    delM = stats.norm.rvs(loc=0, scale=Merr)
    delR = stats.norm.rvs(loc=0, scale=Rerr)
    delT = stats.norm.rvs(loc=0, scale=Terr)

    # 조건에 따른 수정
    delM[delM + M <= 0] *= 0.5
    delR[delR + R <= 0] *= 0.5
    delT[delT + T <= 0] *= 0.5
    
    # 새로운 값 계산
    new_M = M + delM
    new_R = R + delR
    new_T = T + delT
    
    log_M = np.log10(new_M)
    log_R = np.log10(new_R)
    x = np.vstack((log_M, new_T))  # DeprecationWarning 해결
    N = 100
    
    TT = np.logspace(-5, 5, N)
    cc = np.linspace(0, 1, N)

    resM1_chi = np.zeros((N, N))
    resM1_std = np.zeros((N, N))

    a_all_Model1 = np.zeros(N)
    b_M_all_Model1 = np.zeros(N)
    c_T_all_Model1 = np.zeros(N)

    # 진행률을 표시하면서 계산
    for ii, t0 in enumerate(TT):
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

    # 최적화 결과
    tt_m1_std = np.unravel_index(resM1_std.argmin(), resM1_std.shape)
    qq, beta0_m1_std = tt_m1_std

    tt_m1_chi = np.unravel_index(resM1_chi.argmin(), resM1_chi.shape)
    kk, beta0_m1_chi = tt_m1_chi
    
    Relation_1_std = np.array([[a_all_Model1[qq], b_M_all_Model1[qq], cc[beta0_m1_std], c_T_all_Model1[qq], TT[qq], resM1_chi[qq, beta0_m1_std]]])
    Relation_1_chi = np.array([[a_all_Model1[kk], b_M_all_Model1[kk], cc[beta0_m1_chi], c_T_all_Model1[kk], TT[kk], resM1_chi[kk, beta0_m1_chi]]])
    
    return Relation_1_std, Relation_1_chi

# 메인 실행 부분
if __name__ == '__main__':
    iteration = 5000
    num_processes = min(cpu_count(), 10)  # 최대 10개 프로세스 사용
    start = time.perf_counter()
    
    # 병렬 처리 수행 및 결과 저장
    with Pool(processes=num_processes) as pool:
        # tqdm을 사용하여 진행 상황 표시
        results = list(tqdm(pool.imap(task_multirun, range(1, iteration + 1)), total=iteration))
    
    # 결과 저장
    for i, (std, chi) in enumerate(results, start=1):
        np.savetxt(f'/home/jin/WD/data_chi/{i}_number_data.txt', chi, delimiter=',')
        np.savetxt(f'/home/jin/WD/data_std/{i}_number_data.txt', std, delimiter=',')
    
    finish = time.perf_counter()
    print(f'{round(finish-start, 2)} seconds to complete the task')

