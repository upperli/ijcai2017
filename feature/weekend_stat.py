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

    res = []
    msg = []
    #weeks = []
    for i in xrange(8):
        #weeks.append(arr[end_day - 7 * (i + 1): end_day - 7 * i])
        week = arr[end_day - 7 * (i + 1): end_day - 7 * i]
        wd = []
        for i in xrange(7):
            wi = (end_day_weekday + i ) % 7
            if wi >= 5:
                wd.append(week[wi])
        res.append(np.mean(wd))
        
        msg.append("mean_we_%d"%i)



    return np.round(res,2),msg 


def run(gap):
    data = helper.loadShopPay()
    start_date = util.getLabelStartDate(data)
    
    fout = open(helper.get_feature_path(gap,'weekend_stat'),'w')
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
