#coding=utf-8

import os
import sys
import datetime
import numpy as np
import util

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
    median = []
    var = []
    std = []
    max_ = []
    min_ = []
    for i in xrange(size):
        h = filter(lambda x: x != 0,arr[end_day - win - step*i: end_day - step * i])
        mean.append(np.mean(h))
        if win > 2:
            if len(h) == 0:
                median.append(np.nan)
                var.append(np.nan)
                std.append(np.nan)
                max_.append(np.nan)
                min_.append(np.nan)
            else: 
                median.append(np.median(h))
                var.append(np.var(h))
                std.append(np.std(h))
                max_.append(np.max(h))
                min_.append(np.min(h))
    res = mean + median + var + std + max_ + min_
    msg = ['mean_%d_%d_%d'%(win,step,i) for i in xrange(len(mean))]
    msg += ['median_%d_%d_%d'%(win,step,i) for i in xrange(len(median))]
    msg += ['var_%d_%d_%d'%(win,step,i) for i in xrange(len(var))]
    msg += ['std_%d_%d_%d'%(win,step,i) for i in xrange(len(std))]
    msg += ['max_%d_%d_%d'%(win,step,i) for i in xrange(len(max_))]
    msg += ['min_%d_%d_%d'%(win,step,i) for i in xrange(len(min_))]
    
    return np.round(res,1),msg 


def run(gap):
    data = helper.loadShopPay()
    start_date = util.getLabelStartDate(data)
    
    fout = open(helper.get_feature_path(gap,'time_win_log_%s_%s_%s'%(sys.argv[2],sys.argv[3],sys.argv[4])),'w')
    flag = True
    for shop in xrange(1,2001):
        shop = str(shop)
        print shop
        arr = util.toArray(data[shop])
        arr = map(lambda x: np.log(x+1),arr)
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
