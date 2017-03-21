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
    for i in xrange(8):
        week = arr[end_day - 7 * (i + 1): end_day - 7 * i]
        wd = []
        for i in xrange(7):
            wi = (end_day_weekday + i ) % 7
            if wi < 5:
                wd.append(week[wi])
        res.append(np.mean(wd))
        res.append(np.var(wd))
        res.append(np.median(wd))
        res.append(np.max(wd))
        res.append(np.min(wd))
        
        msg.append("mean_wd_%d"%i)
        msg.append("var_wd_%d"%i)
        msg.append("median_wd_%d"%i)
        msg.append("max_wd_%d"%i)
        msg.append("min_wd_%d"%i)



    return np.round(res,1),msg 


def run(gap):
    data = helper.loadShopPay()
    start_date = util.getLabelStartDate(data)
    
    fout = open(helper.get_feature_path(gap,'weekday_stat'),'w')
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
