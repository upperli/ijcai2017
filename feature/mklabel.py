#coding=utf-8

import os
import sys
import datetime

sys.path.append('..')
import utils.helper as helper
import utils.config as config
import math
import util

def run(gap):

    data = helper.loadShopPay()
    start_date = util.getLabelStartDate(data)
    start_date_feature = util.getStartDate(data) 
    fout = open(helper.get_feature_path(gap,'label'),'w')
    target_date = config.PREDICTDATE + datetime.timedelta(days=gap)
    fout.write("label,start_date\n")
    for shop in data:
        shop_data = data[shop]
        for date,num in shop_data.iteritems():
            if start_date[shop] <= date:
                fout.write("%d,%d,%s,%s\n"%(num,(date - start_date_feature[shop]).days,shop,date.strftime('%Y-%m-%d')))
        
        fout.write("%d,%d,%s,%s\n"%(-1,(date - start_date_feature[shop]).days,shop,target_date.strftime('%Y-%m-%d')))
        
    fout.close()
    

if __name__ == '__main__':
    run(int(sys.argv[1]))
        
