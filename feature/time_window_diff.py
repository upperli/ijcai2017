#coding=utf-8

import os
import sys
import datetime
import numpy as np
import util
import math

sys.path.append('..')
import utils.helper as helper
import utils.config as config


def getFeature(date,arr,gap):
    
    idx = (date - util.start).days
    week_day = date.weekday()
    
    end_day = idx - int(gap)

    end_day_weekday = (week_day + 14 - int(gap)) % 7

    step = int(sys.argv[2])
    win = int(sys.argv[3])
    size = int(sys.argv[4])

    mean = []
    for i in xrange(size):
        h = filter(lambda x: x!= 0,arr[end_day - win - step*i: end_day - step * i])
        mean.append(np.mean(h))
    diff = []
    for i in xrange(1,len(mean)):
        if math.isnan(mean[i]) or math.isnan(mean[i-1]):
            diff.append(np.nan)
        else:
            diff.append(mean[i] - mean[i-1])
    res = diff
    msg = ['mean_diff_%d_%d_%d'%(win,step,i) for i in xrange(len(res))]
    
    return np.round(res,1),msg 


def run(gap):
    data = helper.loadShopPay()
    start_date = util.getLabelStartDate(data)
    
    fout = open(helper.get_feature_path(gap,'time_win_diff_%s_%s_%s'%(sys.argv[2],sys.argv[3],sys.argv[4])),'w')
    flag = True
    for shop in xrange(1,2001):
        shop = str(shop)
        print shop

        arr = util.toArray(data[shop])
        #arr = util.fillNull(arr)
        for date in util.getDate(start_date[shop]):
            feature,msg = getFeature(date,arr,gap)
            if flag:
                fout.write('%s\n'%(','.join(msg)))
                flag = False
            util.outputFeature(fout,date,shop,feature)
    fout.close()

if __name__ == '__main__':
    run(sys.argv[1])
