# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:18:48 2020

@author: YangSh

@code_describe : feature_extract function


"""
from __future__ import division
import numpy as np
np.set_printoptions(threshold=np.inf)
import pywt
import math
import nolds
from pyentrp import entropy as ent
from scipy.stats import skew
from scipy.stats import kurtosis

#time domain  
def min_X(X):
    return np.min(X)

def max_X(X):
    return np.max(X)

def std_X(X):
    return np.std(X)

def mean_X(X):
    return np.mean(X)

def var_X(X):
    return np.var(X)
    
def totalVariation(X):
    Max = np.max(X)
    Min = np.min(X)
    return np.sum(np.abs(np.diff(X)))/((Max-Min)*(len(X)-1))

#偏度用来度量分布是否对称。正态分布左右是对称的，偏度系数为0。较大的正值表明该分布具有右侧较长尾部。较大的负值表明有左侧较长尾部
def skew_X(X):
    skewness = skew(X)
    return skewness

#峰度系数（Kurtosis）用来度量数据在中心聚集程度。在正态分布情况下，峰度系数值是3。
#>3的峰度系数说明观察量更集中，有比正态分布更短的尾部；
#<3的峰度系数说明观测量不那么集中，有比正态分布更长的尾部，类似于矩形的均匀分布。
def kurs_X(X):
    kurs = kurtosis(X)
    return kurs

#常用均方根值来分析噪声
def rms_X(X):
    RMS = np.sqrt((np.sum(np.square(X))) * 1.0 / len(X))
    return RMS

def peak_X(X):
    Peak = np.max([np.abs(max_X), np.abs(min_X)])
    return Peak

#峰均比
def papr_X(X):
    Peak = peak_X(X)
    RMS = rms_X(X)
    PAPR = np.square(Peak) * 1.0 / np.square(RMS)
    return PAPR

'''
计算时域10个bin特征
'''
#过滤X中相等的点
def filter_X(X):
    X_new = []
    length = np.shape(X)[0]
    for i in range(1, length):
        if i != 0 and X[i] == X[i-1]:
            continue
        X_new.append(X[i])
    return X_new      

#求X中所有的极大值和极小值点
def minmax_cal(X):
    length = np.shape(X)[0]
    min_value = []
    min_index = []
    max_value = []
    max_index = []
    first = ''
    for i in range(1, length-1):
        if X[i]<X[i-1] and X[i]<X[i+1]:
            min_value.append(X[i])
            min_index.append(i)
        if X[i]>X[i-1] and X[i]>X[i+1]:
            max_value.append(X[i])
            max_index.append(i)
    if len(min_index) and len(max_index):       
        if max_index[0] > min_index[0]:
            first = 'min'
        else:
            first = 'max'
        return min_value, max_value, first
    else:
        return None, None, None
    

#计算所有的极大值和极小值的差值        
def minmax_sub_cal(X):
    min_value, max_value, first = minmax_cal(X)
    if min_value and max_value and first:
        max_length = np.shape(max_value)[0]
        sub = []
        if first == 'min':
            for i in range(max_length-1):
                sub.append(max_value[i] - min_value[i])
                sub.append(max_value[i] - min_value[i+1])
        else:
            for i in range(1, max_length-1):
                sub.append(max_value[i] - min_value[i-1])
                sub.append(max_value[i] - min_value[i])   
        return sub
    else:
        return None     

#计算极大极小值差值占比            
def minmax_percent_cal(X, step=10):  
    X = filter_X(X)
    sub = minmax_sub_cal(X)
    if sub:
        length = int(np.shape(sub)[0])
        max_value = max(sub)
        min_value = min(sub)
        diff = max_value - min_value
        value = diff / step
        nums = []
        sub = np.array(sub)
        for i in range(step):
            scale_min = sub>=min_value+i*value
            scale_max = sub<min_value+(i+1)*value
            scale = scale_min & scale_max
            num = np.where(scale)[0]
            size = np.shape(num)[0]
            nums.append(size)
        nums[-1] = nums[-1] + sum(sub==max_value)
        nums = np.array(nums, dtype = int)
        per = nums / length
        return per
    else:
        return [0,0,0,0,0,0,0,0,0,0]

#非线性特征提取
#采样频率为256Hz,则信号的最大频率为128Hz，进行5层小波分解
def relativePower(X):
    Ca5, Cd5, Cd4, Cd3, Cd2, Cd1 = pywt.wavedec(X, wavelet='db4', level=5)
    EA5 = sum([i*i for i in Ca5])
    ED5 = sum([i*i for i in Cd5])
    ED4 = sum([i*i for i in Cd4])
    ED3 = sum([i*i for i in Cd3])
    ED2 = sum([i*i for i in Cd2])
    ED1 = sum([i*i for i in Cd1])
    E = EA5 + ED5 + ED4 + ED3 + ED2 + ED1
    pEA5 = EA5/E
    pED5 = ED5/E
    pED4 = ED4/E
    pED3 = ED3/E
    pED2 = ED2/E
    pED1 = ED1/E
    return pEA5, pED5, pED4, pED3, pED2, pED1

#nonlinear analysis
#小波熵
def wavelet_entopy(X):
    [pEA5, pED5, pED4, pED3, pED2, pED1] = relativePower(X)
    wavelet_entopy = - (pEA5*math.log(pEA5) + pED5*math.log(pED5)
    + pED4*math.log(pED4) + pED3*math.log(pED3) + pED2*math.log(pED2) + pED1*math.log(pED1))
    return wavelet_entopy

#计算Detrended Fluctuation Analysis值
def DFA(X):
    y = nolds.dfa(X)
    return y

#计算赫斯特指数
def Hurst(X):
    y = nolds.hurst_rs(X)
    return y

#计算Petrosian's Fractal Dimension分形维数值
def Petrosian_FD(X):
    D = np.diff(X)

    delta = 0;
    N = len(X)
    #number of sign changes in signal
    for i in range(1, len(D)):
        if D[i] * D[i-1] < 0:
            delta += 1

    feature = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * delta)))
    
    return feature

#计算样本熵
def sample_entropy(X):
    y = nolds.sampen(X)
    return y

#计算排列熵
#度量时间序列复杂性的一种方法,排列熵H的大小表征时间序列的随机程度，值越小说明该时间序列越规则，反之，该时间序列越具有随机性。
def permutation_entropy(X):
    y = ent.permutation_entropy(X, 4, 1)
    return y

#Hjorth Parameter: mobility and complexity
def Hjorth(X):
    D = np.diff(X)
    D = list(D)
    D.insert(0, X[0])
    VarX = np.var(X)
    VarD = np.var(D)
    Mobility = np.sqrt(VarD / VarX)
    
    DD = np.diff(D)
    VarDD = np.var(DD)
    Complexity = np.sqrt(VarDD / VarD) / Mobility
    
    return Mobility, Complexity
