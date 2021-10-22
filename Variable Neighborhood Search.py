# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:07:21 2021

@author: zongsing.huang
"""

import itertools

import numpy as np

#%% 題庫
benchmark = np.array([[ 0, 19, 92, 29, 49, 78,  6],
                      [19,  0, 21, 85, 45, 16, 26],
                      [92, 21,  0, 24, 26, 87, 47],
                      [29, 85, 24,  0, 76, 17,  8],
                      [49, 45, 26, 76,  0, 90, 27],
                      [78, 16, 87, 17, 90,  0, 55],
                      [ 6, 26, 47,  8, 27, 55,  0]])

#%% 函數定義
def fitness(X, benchmark):
    if X.ndim==1:
        X = X.reshape(1, -1)
        
    P = X.shape[0]
    D = X.shape[1]
    F = np.zeros(P)
    
    for i in range(P):
        X_new = np.append(X[i], X[i, 0])
        
        for j in range(D):
            st = X_new[j].astype(int)
            ed = X_new[j+1].astype(int)
            F[i] += benchmark[st, ed]
    
    return F

def swap(X):
    D = X.shape[0]
    idx = np.arange(D)
    comb = list(itertools.combinations(idx, 2))
    X_new = np.zeros([len(comb), D])
    
    for i, (j, k) in enumerate(comb):
        X_new[i] = X.copy()
        X_new[i, j], X_new[i, k] = X_new[i, k], X_new[i, j]
    
    return X_new

def LS(X1, F1):
    lt = 1
    
    while lt<=lt_max:
        h = 1
        
        while h<=h_max:
            # 更新
            if h==1:
                X2_set = swap(X1)
            elif h==2:
                X2_set = swap(X1) # 通常是insert
            elif h==3:
                X2_set = swap(X1) # 通常是inverse
            
            # 適應值計算
            F2_set = fitness(X2_set, benchmark)
            
            # 取得X2
            idx = F2_set.argmin()
            X2 = X2_set[idx]
            F2 = F2_set[idx]
    
            # 新的解比較好，直接接受
            if F2<F1:
                X1 = X2.copy()
                F1 = F2.copy()
                h = 1
            else:
                h += 1
        
        lt += 1
    
    X2 = X1
    F2 = F1
    
    return X2, F2
                
#%% 參數設定
D = benchmark.shape[1] # 維度
rho_max = 10
lamda_max = 3
lt_max = 10
h_max = 3

#%% 初始化
X = np.random.choice(D, size=D, replace=False) # 初始解
F = fitness(X, benchmark) # 初始適應值

rho = 1
while rho<=rho_max:
    lamda = 1
    
    while lamda<=lamda_max:
        # 更新
        if lamda==1:
            X1_set = swap(X)
        elif lamda==2:
            X1_set = swap(X) # 通常是insert
        elif lamda==3:
            X1_set = swap(X) # 通常是inverse
        
        # 適應值計算
        F1_set = fitness(X1_set, benchmark)
        
        # 取得X1
        idx = F1_set.argmin()
        X1 = X1_set[idx]
        F1 = F1_set[idx]
        
        # Local Search
        X2, F2 = LS(X1, F1)
        
        # 新的解比較好，直接接受
        if F2<F:
            X = X2.copy()
            F = F2.copy()
            lamda = 1
        else:
            lamda += 1
    
    rho = rho + 1