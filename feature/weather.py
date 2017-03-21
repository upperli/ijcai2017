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

def loadWeatherData():
    res = collections.defaultdict(dict)
    with open(os.path.join(config.DATASET,'weather_all.csv')) as fin:
        for line in fin:
            tmp = line.strip().split(",")
            city = tmp[0]
            date = datetime.datetime.strptime(tmp[1],'%Y-%m-%d')
            max_tem = int(tmp[2])
            min_tem = int(tmp[3])
            we = tmp[4]
            wind = tmp[-1]
            res[city][date] = [max_tem,min_tem,we,wind]  
    return res

def getFeature(tmp):
    #weather_list = ['雨','小雨','中雨','大雨','暴雨','雪','暴雪','雷','雾','霾']
    weather_dict = {
                '雨':1,
                '小雨':1,
                '中雨':2,
                '大雨':3,
                '暴雨':4,
                '雪':1,
                '暴雪':4,
                '雷':3,
                '雾':1,
                '霾':2
    }
    res = [tmp[0],tmp[1]]
    w_f = 0
    for w in weather_dict:
        if w in tmp[2]:
            w_f = max(w_f,weather_dict[w])

    #w_f = [0] * len(weather_list)
    #for i in xrange(len(weather_list)):
    #    if weather_list[i] in tmp[2]:
    #        w_f[i] = 1

    wind_f = 0
    if '5' in tmp[3] or '6' in tmp[3]:
        wind_f = 3
    elif '4' in tmp[3] or ('3' in tmp[3] and '小于3' not in tmp[3]):
        wind_f = 2
    else:
        wind_f = 1
    res.append(w_f)

    res.append(wind_f)

    msg = ['max_tem','min_tem'] + ['weather'] + ['wind']
    return res,msg

def getShopFeatures(gap):
    weatherdata = loadWeatherData()
    shop_info = helper.loadShopInfo()

    flag = True
    with open(helper.get_feature_path(gap,'weather_feature'),'w') as fout:
        for shop in xrange(1,2001):
            shop = str(shop)
            city = shop_info[shop][0]
            for date in util.getDate(start = config.LABELSTARTDATE):
                feature,msg = getFeature(weatherdata[city][date])
                if flag:
                    fout.write("%s\n"%(",".join(msg)))
                    flag = False
                tmp = ",".join(map(str,feature) )
                fout.write("%s,%s,%s\n"%(tmp,shop,date.strftime("%Y-%m-%d")))

def run(gap):
    getShopFeatures(gap)


if __name__ == '__main__':
    run(sys.argv[1])
