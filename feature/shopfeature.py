#coding=utf-8

import os
import sys
import datetime
import collections

sys.path.append('..')
import utils.helper as helper
import utils.config as config


#1,湖州,885,8,4,12,2,美食,休闲茶饮,饮品/甜点

def shopInfo():
    res = {}
    shop_info = helper.loadShopInfo()
    for shop in shop_info:
        tmp = []
        for i in xrange(2,6):
            if shop_info[shop][i] == '':
                tmp.append(0)
            else:
                tmp.append(shop_info[shop][i])
        res[shop] = tmp
    return res
        

def getShopFeatures(gap):
    shop_info = shopInfo()
    with open(helper.get_feature_path(gap,'shop_s'),'w') as fout:
        fout.write("per_pay,score,comment_cnt,shop_level,id\n")    
        for shop in xrange(1,2001):
            tmp = ",".join(map(lambda x: str(x),shop_info[str(shop)]))
            fout.write("%s,%d,%d\n"%(tmp,shop,shop))

def run(gap):
    getShopFeatures(gap)


if __name__ == '__main__':
    run(sys.argv[1])
