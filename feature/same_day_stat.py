#coding=utf-8

import os
import sys
import datetime
import numpy as np
import util

sys.path.append('..')
import utils.helper as helper
import utils.config as config

from scipy.stats import skew
from scipy.stats import kurtosis
import pandas as pd

def getMean(arr):
    res = []
    msg = []
    res.append(np.mean(arr[:2]))
    res.append(np.mean(arr[:4]))
    res.append(np.mean(arr[:8]))    
    msg.append('mean_2')
    msg.append('mean_4')
    msg.append('mean_8')
    return res,msg

def getMedian(arr):
    res = []
    msg = []
    res.append(np.median(arr[:4]))
    res.append(np.median(arr[:8]))    
    msg.append('median_4')
    msg.append('median_8')
    return res,msg

def getVar(arr):
    res = []
    msg = []
    res.append(np.var(arr[:4]))
    res.append(np.var(arr[:8]))    
    msg.append('var_4')
    msg.append('var_8')
    return res,msg

def getMad(arr):
    res = []
    msg = []
    res.append(pd.Series(arr[:4]).mad())
    res.append(pd.Series(arr[:8]).mad())
    msg.append('mad_4')
    msg.append('mad_8')
    return res,msg

def getSkew(arr):
    return [skew(arr)],['skew']

def getKurtosis(arr):
    return [kurtosis(arr)],['kurtosis']
        
def getFeature(date,arr,gap):
    gap = int(gap)
    
    idx = (date - util.start).days
    if gap > 6:
        idx -= 7

    week_day = date.weekday()

    history_day = list(reversed(arr[idx - 7 * 8: idx : 7]))

    res = []
    msg = []
    feature_funs = [getMean,getMedian,getVar,getMad,getSkew,getKurtosis]
    
    for fun in feature_funs:
        a,b = fun(history_day)
        res += a
        msg += b
    msg = ['day_stat_%s'%x for x in msg]
    return np.round(res,2),msg 


def run(gap):
    data = helper.loadShopPay()
    start_date = util.getLabelStartDate(data)
    
    fout = open(helper.get_feature_path(gap,'same_day_stat'),'w')
    flag = True
    for shop in xrange(1,2001):
        shop = str(shop)
        print shop
        arr = util.toArray(data[shop])
        for date in util.getDate(start_date[shop]):
            feature,msg = getFeature(date,arr,gap)
            if flag:
                fout.write('%s\n'%(','.join(msg)))
                flag = False
            util.outputFeature(fout,date,shop,feature)
    fout.close()

if __name__ == '__main__':
    run(sys.argv[1])
