#coding=utf-8

import os
import sys
import datetime
import collections
import util

sys.path.append('..')
import utils.helper as helper
import utils.config as config

import pandas as pd


#1,湖州,885,8,4,12,2,美食,休闲茶饮,饮品/甜点

def loadAirData():
    data = pd.read_csv(os.path.join(config.DATASET,'alldata.csv'),parse_dates=['Date'],date_parser=lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
    return data


def getShopFeatures(gap):
    airdata = loadAirData()
    shop_info = helper.loadShopInfo()

    with open(helper.get_feature_path(gap,'air_feature'),'w') as fout:
        fout.write("air\n")    
        for shop in xrange(1,2001):
            shop = str(shop)
            city = shop_info[shop][0]
            for date in util.getDate(start = config.LABELSTARTDATE):
                tmp = airdata[airdata.Date == date][city]  
                fout.write("%d,%s,%s\n"%(tmp,shop,date.strftime("%Y-%m-%d")))

def run(gap):
    getShopFeatures(gap)


if __name__ == '__main__':
    run(sys.argv[1])
