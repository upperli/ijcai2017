#coding=utf-8

import os
import sys
import datetime
import collections
import numpy as np
sys.path.append('..')
import utils.helper as helper
import utils.config as config


#1,湖州,885,8,4,12,2,美食,休闲茶饮,饮品/甜点

def getCityCate2Group():
    res = dict()
    shop_info = helper.loadShopInfo()
    for shop,tmp in shop_info.items():
        res[shop] = (tmp[0],tmp[7])
    return res

def loadShopFeature(filename,group):
    res = collections.defaultdict(dict)
    l = 0
    with open(filename) as fin:
        names = fin.readline().strip().split(',')
        for line in fin:
            tmp = line.strip().split(',')
            grp = group[tmp[-2]]
            res[grp].setdefault(tmp[-1],[])
            res[grp][tmp[-1]].append(map(float,tmp[:-2]))
    return res,len(names),names
    

def getShopFeatures(filename,gap):
    #lis,shop_info = shopInfoStat()
    group = getCityCate2Group()
    res,l,names = loadShopFeature(helper.get_feature_path(gap,filename),group)
    grp_feature = collections.defaultdict(dict)
    for grp,item in res.items():
        for date,features in item.items():
            #print features
            max_feature = list(np.max(features,axis=0))
            min_feature = list(np.min(features,axis=0))
            median_feature = list(np.median(features,axis=0))
            grp_feature[grp][date] = max_feature + min_feature + median_feature
    msg = map(lambda x: 'city_cate2_max_' + x,names) + map(lambda x: 'city_cate2_min_' + x,names) + map(lambda x: 'city_cate2_median_' + x,names)
    with open(helper.get_feature_path(gap,'city_cate2_' + filename),'w') as fout:
        flag = True
        for shop in xrange(1,2001):
            shop = str(shop)
            grp = group[shop]
            if flag:
                fout.write("%s\n"%(",".join(msg)))
                flag = False
            for date in grp_feature[grp]:
                fea = grp_feature[grp][date]
                tmp = ",".join(map(lambda x: str(x),fea))
                fout.write("%s,%s,%s\n"%(tmp,shop,date))

def run(gap):
    getShopFeatures('same_day_hist',gap)
    getShopFeatures('time_win_1_1_21',gap)


if __name__ == '__main__':
    run(sys.argv[1])
