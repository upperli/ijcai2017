#coding=utf-8

import os
import sys
import datetime
import numpy as np
import math

sys.path.append('..')
import utils.helper as helper
import utils.config as config
from sklearn import linear_model
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
import pandas as pd

start = config.BASEDATE

end = config.PREDICTDATE + datetime.timedelta(days=14)

second_week = config.PREDICTDATE + datetime.timedelta(days=7)

start_weekday = start.weekday()


def median(arr):
    return np.median(filter(lambda x: x!= 0 ,arr))

def mean(arr):
    return np.median(filter(lambda x: x!= 0 ,arr))

def std(arr):
    return np.std(filter(lambda x: x!= 0 ,arr))

def skew_1(arr):
    return skew(filter(lambda x: x!= 0 ,arr))

def kurtosis_1(arr):
    return kurtosis(filter(lambda x: x!= 0 ,arr))

def mad(arr):
    return pd.Series(filter(lambda x: x!= 0 ,arr)).mad()

def getArrMedian(h_sum,suffix):
    res = []
    msg = ['m_3','m_5','m_a','mean_5','mean','mad','std','skew','kurtosis']
    msg = map(lambda x: x + '_' + suffix,msg)
    res.append(median(h_sum[:3]))
    res.append(median(h_sum[:5]))
    res.append(median(h_sum))
    
    res.append(mean(h_sum[:5]))
    res.append(mean(h_sum))

    res.append(mad(h_sum))
    res.append(std(h_sum))
    
    res.append(skew_1(h_sum))
    res.append(kurtosis_1(h_sum))
    
    res = map(lambda x:round(x,2) if not math.isnan(x) else -1,res) 
    
    return res,msg

def getArrMedian_1(h_sum,step,suffix):
    res = []
    msg = ['m_1','m_3','m_6','m_a','std_3','std_6','std_a','mad_3','mad_6','mad_a','skew','kurtosis']
    msg = map(lambda x: x + '_' + suffix,msg)
    
    res.append(median(h_sum[:step]))
    res.append(median(h_sum[:3*step]))
    res.append(median(h_sum[:6*step]))
    res.append(median(h_sum))
    
    res.append(std(h_sum[:3*step]))
    res.append(std(h_sum[:6*step]))
    res.append(std(h_sum))
    
    res.append(mad(h_sum[:3*step]))
    res.append(mad(h_sum[:6*step]))
    res.append(mad(h_sum))
    
    res.append(skew_1(h_sum))
    res.append(kurtosis_1(h_sum))
    res = map(lambda x:round(x,2) if not math.isnan(x) else 0,res) 

    return res,msg

def getMatrixMS(matrix,suffix):
    res = []
    msg = ['m_1','m_3','m_5','m_a','s_1','s_3','s_5','s_a','skew','kurtosis']
    msg = map(lambda x: x + '_' + suffix,msg)
    
    res.append(np.median(matrix[:1]))
    res.append(np.median(matrix[:3]))
    res.append(np.median(matrix[:5]))
    res.append(np.median(matrix))
    
    res.append(np.std(matrix[:1]))
    res.append(np.std(matrix[:3]))
    res.append(np.std(matrix[:5]))
    res.append(np.std(matrix))

    res = map(lambda x:int(x),res) 
    
    h = reduce(lambda x,y: x + y,matrix,[])

    res.append(skew_1(h))
    res.append(kurtosis_1(h))
    return res,msg

'''
def getDiffMS(arr,suffix):
    res = np.array(arr[:-1]) - np.array(arr[1:])
    res,msg = getArrMS(res,suffix+"_diff")
    return res,msg
'''
    
def toArray(data):
    arr = [0] * (end - start).days
    for date in data:
        idx = (date - start).days
        if idx < 0:
            continue
        arr[idx] = data[date]
    return arr

def fillNull(arr):
    for i in xrange(7,len(arr)):
        if arr[i] == 0:             
            arr[i] = -1
    return arr

def outputFeature(fout,date,shop,feature):
    fout.write("%s,%s,%s\n"%(",".join(map(lambda x: str(x),feature)),shop,date.strftime("%Y-%m-%d")))

def getDate(start = config.LABELSTARTDATE,end = config.PREDICTDATE + datetime.timedelta(days = 14)):
    date = start
    while date < end:
        yield date
        date += datetime.timedelta(days = 1)    

def getStartDate(data):
    dic = dict()
    for shop in data:
        shop_data = sorted(data[shop].items(),key = lambda x: x[0])
        dic[shop] = shop_data[0][0]
    return dic

def getLabelStartDate(data):
    dic = dict()
    for shop in data:
        shop_data = sorted(data[shop].items(),key = lambda x: x[0])
        dic[shop] = max(shop_data[0][0] + datetime.timedelta(days=35),config.LABELSTARTDATE)
    return dic

