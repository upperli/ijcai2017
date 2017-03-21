#coding=utf-8

import os
import sys
import datetime
import math
import util

sys.path.append('..')
import utils.helper as helper
import utils.config as config


def getDateFeatures(gap):
    festivals = helper.loadFestival()

    with open( helper.get_feature_path(gap,'date_d'),'w') as fout:
        fout.write("if_work,is_fes,week_day\n")
        for date in util.getDate(start = config.LABELSTARTDATE):
            week_day = date.weekday()
            is_fes = 1 if date in festivals else 0
            tmp = 1 if week_day == 0 else 8 - week_day
            if_work = 1 if week_day < 5 else 0
            month_of_week = int(date.strftime("%W")) - int(datetime.datetime(date.year,date.month,1).strftime("%W") )+ 1
            fout.write("%d,%d,%d,%s\n"%(if_work,is_fes,week_day,date.strftime("%Y-%m-%d")))

def run(gap):
    getDateFeatures(gap)


if __name__ == '__main__':
    run(sys.argv[1])
