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

    res = []
    msg = []
    #weeks = []
    histmax = [0] * 7
    histmin = [0] * 7
    for i in xrange(8):
        week = arr[end_day - 7 * (i + 1): end_day - 7 * i]
        argmin = np.argmin(week)
        argmax = np.argmax(week)
        histmax[argmax] += 1
        histmin[argmin] += 1
    res = histmax + histmin
    msg = ["max_count_%d"%i for i in xrange(7)]
    msg += ["min_count_%d"%i for i in xrange(7)]

    return np.round(res,2),msg 


def run(gap):
    data = helper.loadShopPay()
    start_date = util.getLabelStartDate(data)
    
    fout = open(helper.get_feature_path(gap,'min_max_stat'),'w')
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
