#coding=utf-8

import os
import sys
import datetime
import collections
import numpy as np
import math

sys.path.append('..')
import utils.helper as helper
import utils.config as config


def loadSingleFeature(filename):
    res = dict()
    l = 0
    with open(filename) as fin:
        names = fin.readline().strip().split(',')
        for line in fin:
            tmp = line.strip().split(',')
            res[tmp[-1]] = tmp[:-1]
    return res,len(names),names

def loadMixFeature(filename):
    res = collections.defaultdict(dict)
    l = 0
    with open(filename) as fin:
        if not filename.endswith('history'):
            names = fin.readline().strip().split(',')
            l = len(names)
        else:
            names = []
        for line in fin:
            tmp = line.strip().split(',')
            res[tmp[-2]][tmp[-1]] = tmp[:-2]
            l = len(tmp[:-2])
    return res,l,names

def output(data_join,gap):
    with open(helper.get_data_path(gap,'data'),'w') as fout:
        for shop,shop_data in data_join.iteritems():
            for date,fea in shop_data.iteritems():
                res = ",".join(map(str,fea))
                fout.write("%s#%s,%s\n"%(res,shop,date))

def outputnames(names,gap):
    with open(helper.get_data_path(gap,'data.names'),'w') as fout:
        for i,name in enumerate(names):
            fout.write("%d "%i + name + ' q\n')

def run(gap):

    helper.log_time('start loading...')
    date_feature = []
    shop_feature = []
    date_feature = ['date_d']
    shop_feature = ['shop_s','shop_cate_s','day_feature']
    #mix_feature = ['label','history','air_feature','weather_feature']
    mix_feature = ['label','same_day_hist','time_win_1_1_21','time_win_log_1_1_21','time_win_2_2_7','time_win_3_6_7','time_win_7_14_7','weekend_stat','weekday_stat','same_day_stat','min_max_stat','air_feature','weather_feature']
    #mix_feature = ['label','same_day_hist','time_win_1_1_21','time_win_2_2_7','time_win_2_4_7','time_win_3_6_7','time_win_7_7_7','time_win_7_14_7','time_win_diff_1_1_21','weekend_stat','weekday_stat','time_win_diff_1_1_21','min_max_stat','air_feature','weather_feature','same_day_stat','city_cate2_same_day_hist','city_cate2_time_win_1_1_21']
    
    #mix_feature = ['label','same_day_hist','time_win_1_1_21','time_win_2_2_7','time_win_2_4_7','time_win_3_6_7','time_win_7_7_7','time_win_7_14_7','time_win_diff_1_1_21','time_win_diff_1_1_21']
    #mix_feature = ['label','same_day_hist','same_day_stat','same_day_diff_mean','week_hist','week_stat','weekend_stat','weekday_stat','min_max_stat','air_feature','weather_feature','city_cate2_same_day_hist','city_cate2_week_hist','locate_same_day_hist','locate_week_hist','week_hist_2_2','week_hist_2_4','week_hist_3_6','week_hist_4_8']
    #mix_feature = ['label','same_day_hist','same_day_stat','same_day_diff_mean','same_day_reg','weekday_stat','weekend_stat','week_hist','week_hist_diff','week_stat','min_max_stat','shop_cate_s','air_feature','weather_feature']
    data = {'date':[],'shop':[],'mix':[]}
    for d in date_feature:
        data['date'].append(loadSingleFeature(helper.get_feature_path(gap,d)))
    for s in shop_feature:
        data['shop'].append(loadSingleFeature(helper.get_feature_path(gap,s)))
    for m in mix_feature:
        data['mix'].append(loadMixFeature(helper.get_feature_path(gap,m)))
    helper.log_time('finish loading...')
    
    helper.log_time('start mix join...')
    data_join = data['mix'][0][0]
    
    names = data['mix'][0][2][1:]
    for i in xrange(1,len(data['mix'])):
        
        m = data['mix'][i][0]
        l = data['mix'][i][1]
        names += data['mix'][i][2]
        for shop,shop_data in data_join.iteritems():
            for date,fea in shop_data.iteritems():
                if shop in m and date in m[shop]:
                    fea += m[shop][date]
                else:
                    fea += [np.nan] * l

    helper.log_time('start date join...')
    for d,l,n in data['date']:
        names += n
        for shop,shop_data in data_join.iteritems():
            for date,fea in shop_data.iteritems():
                if date in d:
                    fea += d[date]
                else:
                    fea += [np.nan] * l

    
    helper.log_time('start shop join...')
    for s,l,n in data['shop']:
        names += n
        print n
        for shop,shop_data in data_join.iteritems():
            for date,fea in shop_data.iteritems():
                if shop in s:
                    fea += s[shop]
                else:
                    fea += [np.nan] * l
    
    helper.log_time('start write...')
    output(data_join,gap)
    outputnames(names,gap)
if __name__ == '__main__':
    run(sys.argv[1])
