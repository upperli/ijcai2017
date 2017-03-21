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

    lis = []
    for i in xrange(5):
        lis.append(collections.defaultdict(int))
    for shop in shop_info:
        lis[0][ shop_info[shop][0] ] += 1
        lis[1][ shop_info[shop][1] ] += 1
        lis[2][ shop_info[shop][6] ] += 1
        lis[3][ shop_info[shop][7] ] += 1
        lis[4][ shop_info[shop][8] ] += 1
    lis1 = []
    for dic in lis:
        idx = dict(map(lambda x: (x[1],x[0]),enumerate(map(lambda x: x[0],sorted(dic.items(), key = lambda x: -x[1])))))
        lis1.append(idx)
    return shop_info,lis1
        

def getShopFeatures(gap):
    shop_info,lis = shopInfo()
    with open(helper.get_feature_path(gap,'shop_cate_s'),'w') as fout:
        fout.write("city,locate,cate1,cate2,cate3\n")    
        for shop in xrange(1,2001):
            shop = str(shop)
            tmp = []
            tmp.append(lis[0].get(shop_info[shop][0],-1))
            tmp.append(lis[1].get(shop_info[shop][1],-1))
            tmp.append(lis[2].get(shop_info[shop][6],-1))
            tmp.append(lis[3].get(shop_info[shop][7],-1))
            tmp.append(lis[4].get(shop_info[shop][8],-1))
            tmp = ",".join(map(lambda x: str(x),tmp))
            
            fout.write("%s,%s\n"%(tmp,shop))

def run(gap):
    getShopFeatures(gap)


if __name__ == '__main__':
    run(sys.argv[1])
