#coding=utf-8
import os
import sys
import datetime
import numpy as np

sys.path.append('..')
import utils.helper as helper
import utils.config as config

hangzhou = set([])
shop_info = helper.loadShopInfo()
for shop in shop_info:
    if shop_info[shop][0] == '杭州':
        hangzhou.add(shop)
print len(hangzhou)
def loadData(gap,expansion=False):
    festival = helper.loadFestival()
    data = []
    com = []
    week_day = (int(gap) + 1) % 7
    with open(helper.get_data_path(gap,'data')) as fin:    
        for line in fin:
            tmp = line.strip().split("#")
            shop,date = tmp[1].split(',')
            #if date < '2016-09-10' and date > '2016-09-01' and shop in hangzhou:
            #    continue
            date = datetime.datetime.strptime(date,'%Y-%m-%d')
            if date in festival:
                continue
            if not expansion and date.weekday() != week_day:
                continue
            data.append(map(lambda x: float(x),tmp[0].split(",")))
            com.append((shop,date))
    return data,com

def split(data,comm):
    train = ([],[])
    valid = ([],[])
    test = ([],[])
    festival = helper.loadFestival()
    for d,c in zip(data,comm):
        date = c[1]
        if date >= config.PREDICTDATE:
            test[0].append(d)
            test[1].append(c)
            continue
        idx = ((config.PREDICTDATE - date).days  - 1) / 7

        train[0].append(d)
        train[1].append(c)
    
    return train,test

def split_lastweek(data,comm,t):
    train = ([],[])
    valid = ([],[])
    test = ([],[])
    #festival = helper.loadFestival()
    #data_clean = helper.loadShopPayClean()
    shop1 = set()
    shop2 = set()
    shopdic = dict()
    for d,c in zip(data,comm):
        shop = c[0]
        shopdic.setdefault(shop,[])
        if c[1] >= (config.PREDICTDATE - datetime.timedelta(days = 100)) and c[1] < (config.PREDICTDATE - datetime.timedelta(days = 7)):
            shopdic[shop].append(d[0])
    for shop in shopdic:
        tmp = shopdic[shop]
        tmp = sorted(tmp)
        if len(tmp) == 0 or tmp[int(len(tmp) * 0.9)] < 70:
            shop1.add(shop)
        else:
            shop2.add(shop)
    
    print len(shop1),len(shop2)
    for d,c in zip(data,comm):
        date = c[1]
        if date >= config.PREDICTDATE:

            if t == 1 and c[0] not in shop1:
                continue
            if t == 2 and c[0] in shop1:
                continue
            test[0].append(d)
            test[1].append(c)
            continue
        idx = ((config.PREDICTDATE - date).days  - 1) / 7

        if idx == 0:
            if t == 1 and c[0] not in shop1:
                continue
            if t == 2 and c[0] in shop1:
                continue
            valid[0].append(d)
            valid[1].append(c)
        else:#if date in data_clea0[c[0]]:
            if t == 1 and d[0] > 90 and c[0] not in shop1:
                continue
            #if t == 2 and d[0] < 50 and c[0] in shop1:
            #    continue
            train[0].append(d)
            train[1].append(c)
    
    return train,valid,test


if __name__ == '__main__':
    data,comm = loadData()
    a,b= split(data,comm)
    print map(lambda x: len(x[0]),a)
