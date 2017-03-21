#coding=utf-8

import config
import os
import datetime
def loadFestival():
    res = set()
    with open(os.path.join(config.DATASET,'festival.txt')) as fin:
        for line in fin:
            tmp = line.strip().split(',')
            date = datetime.datetime.strptime(tmp[0],"%Y-%m-%d")
            for i in xrange(int(tmp[1])):
                res.add(date + datetime.timedelta(days = i))
    return res


def loadShopPay():
   shop_pay = {}
   log_time('load shop pay...')
   with open(os.path.join(config.FEATURE,'shop_per_day')) as fin:
        for line in fin:
            tmp = line.strip().split("\t")
            date = datetime.datetime.strptime(tmp[1],"%Y-%m-%d")
            shop_pay.setdefault(tmp[0],{})
            shop_pay[tmp[0]].setdefault(date,0)
            shop_pay[tmp[0]][date] = int(tmp[2])
   log_time('load shop pay finish!')
   return shop_pay

def loadViewPay():
   shop_pay = {}
   log_time('load shop view...')
   with open(os.path.join(config.FEATURE,'view_per_day')) as fin:
        for line in fin:
            tmp = line.strip().split("\t")
            date = datetime.datetime.strptime(tmp[1],"%Y-%m-%d")
            shop_pay.setdefault(tmp[0],{})
            shop_pay[tmp[0]].setdefault(date,0)
            shop_pay[tmp[0]][date] = int(tmp[2])
   log_time('load shop view finish!')
   return shop_pay


def loadShopInfo():
    shop_info = {}
    log_time('load shop info...')
    with open(os.path.join(config.DATASET,'shop_info.txt')) as fin:
        for line in fin:
            tmp = line.strip().split(',')
            shop_info[tmp[0]] = tmp[1:]
    log_time('finish load shop info...')
    return shop_info

def loadShopPayClean():
   shop_pay = {}
   log_time('load shop pay clean...')
   with open(os.path.join(config.FEATURE,'shop_per_day_clean')) as fin:
        for line in fin:
            tmp = line.strip().split(",")
            date = datetime.datetime.strptime(tmp[1],"%Y-%m-%d")
            shop_pay.setdefault(tmp[0],{})
            shop_pay[tmp[0]].setdefault(date,0)
            shop_pay[tmp[0]][date] = int(tmp[2])
   log_time('load shop pay clean finish!')
   return shop_pay

def log_time(msg):
    print datetime.datetime.now(),msg

def get_feature_path(gap,filename):
    path = os.path.join(config.FEATURE,str(gap))
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.join(path,filename)

def get_data_path(gap,filename):
    path = os.path.join(config.DATA,str(gap))
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.join(path,filename)
if __name__ == '__main__':
    print get_feature_path(0,'1')
    for date in loadFestival():
        print date
