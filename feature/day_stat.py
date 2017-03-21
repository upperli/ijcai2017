#coding=utf-8

import os
import sys
import datetime
import numpy as np
import util
import collections

sys.path.append('..')
import utils.helper as helper
import utils.config as config


def loadAllData():
    shop = collections.defaultdict(dict)
    shop1 = dict()
    i= 0 
    with open(os.path.join(config.DATASET,'user_pay.txt')) as fin:
        for line in fin:
            tmp = line.strip().split(",")
            date,h = tmp[2].split(" ") 
            h = int(h.split(":")[0])
            shop1.setdefault(tmp[1],[0] * 8)
            shop[tmp[1]].setdefault(date,{'max':0,'min':24})
            shop[tmp[1]][date]['max'] = max(shop[tmp[1]][date]['max'],h)
            shop[tmp[1]][date]['min'] = min(shop[tmp[1]][date]['min'],h)
            shop1[tmp[1]][h / 3] += 1
            if i % 1000 == 0:
                print i
            i += 1
    return shop,shop1

def getFeature(data,data1):
    max_ = 0
    min_ = 24
    long_ = 0
    short_ = 24
    sum_ = 0
    count_ = 0
    for date in data:
        tmp = data[date]
        max_ = max(tmp['max'],max_)
        min_ = min(tmp['min'],min_)
        long_ = max(long_,tmp['max'] - tmp['min'])
        short_ = min(short_,tmp['max'] - tmp['min'])
        sum_ += tmp['max'] - tmp['min']
        count_ += 1
    res = [max_,min_,long_,short_,1.0 * sum_/count_] + list(1.0 * np.array(data1) / np.sum(data1))
    msg = ['max_hour','min_hour','long_time','short_time','mean_time'] + ["hours_%d"%i for i in xrange(8)]
    return np.round(res,2),msg 


def run(gap):

    shopdata,shop1data = loadAllData()
    #data = helper.loadShopPay()
    #start_date = util.getLabelStartDate(data)
    
    fout = open(helper.get_feature_path(gap,'day_feature'),'w')
    flag = True
    for shop in xrange(1,2001):
        shop = str(shop)
        print shop
        feature,msg = getFeature(shopdata[shop],shop1data[shop])
        if flag:
            fout.write('%s\n'%(','.join(msg)))
            flag = False
        tmp = ",".join(map(str,feature))
        fout.write("%s,%s\n"%(tmp,shop))
    fout.close()

if __name__ == '__main__':
    run(sys.argv[1])
