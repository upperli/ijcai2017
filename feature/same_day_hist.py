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
    gap = int(gap)
    
    idx = (date - util.start).days
    if gap > 6:
        idx -= 7

    week_day = date.weekday()
    history_day = []
    for i in xrange(1,9):
        history_day.append(arr[idx - 7*i])
    #history_day = list(reversed(arr[idx - 7 * 8: idx : 7]))
    
    res = history_day[3:]
    msg = ["same_day_%d"%i for i in xrange(5)]    
    
    diff1 = list(np.array(history_day[:-1]) - np.array(history_day[1:]))
    diff_msg = ["same_diff_%d"%i for i in xrange(7)]

    res += diff1
    msg += diff_msg

    return res,msg 


def run(gap):
    data = helper.loadShopPay()
    start_date = util.getLabelStartDate(data)
    
    fout = open(helper.get_feature_path(gap,'same_day_hist'),'w')
    flag = True
    for shop in xrange(1,2001):
        shop = str(shop)
        print shop
        arr = util.toArray(data[shop])
        arr = util.fillNull(arr)
        for date in util.getDate(start_date[shop]):
            feature,msg = getFeature(date,arr,gap)
            if flag:
                fout.write('%s\n'%(','.join(msg)))
                flag = False
            util.outputFeature(fout,date,shop,feature)
    fout.close()

if __name__ == '__main__':
    run(sys.argv[1])
