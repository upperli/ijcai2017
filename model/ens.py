#!/bin/sh

import numpy as np
import sys

xgb = {}
rf = {}
bagging_online = {}
bagging_split_1 = {}
bagging_split_shop = {}

with open('../../result/xgb_all.csv') as fin:
    for line in fin:
        tmp = line.strip().split(",")
        shop = tmp[0]
        xgb[shop] = map(lambda x: float(x),tmp[1:])

with open('../../result/rf_all.csv') as fin:
    for line in fin:
        tmp = line.strip().split(",")
        shop = tmp[0]
        rf[shop] = map(lambda x: float(x),tmp[1:])


with open('../../result/bagging_all.csv') as fin:
    for line in fin:
        tmp = line.strip().split(",")
        shop = tmp[0]
        bagging_online[shop] = map(lambda x: float(x),tmp[1:])
'''
with open('./bagging_split_1.csv') as fin:
    for line in fin:
        tmp = line.strip().split(",")
        shop = tmp[0]
        bagging_split_1[shop] = map(lambda x: float(x),tmp[1:])

with open('./bagging_split_shop.csv') as fin:
    for line in fin:
        tmp = line.strip().split(",")
        shop = tmp[0]
        bagging_split_shop[shop] = map(lambda x: float(x),tmp[1:])
'''
for shop in xgb:
    t1 = xgb[shop]
    t2 = rf[shop]
    t3 = bagging_online[shop]
   # t4 = bagging_split_1[shop]
   # t5 = bagging_split_shop[shop]

    res = map(int,np.array(t1) * 0.6 + np.array(t3) * 0.2 + np.array(t2) * 0.2)[:7]
    res += res
    res = ",".join(map(lambda x: str(int(x)),res))
    print shop + "," + res  
        
