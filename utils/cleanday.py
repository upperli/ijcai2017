#coding=utf-8
import os
import config
import helper
import datetime
import numpy as np
from sklearn import linear_model

def regression(data):
    reg = linear_model.LinearRegression()
    reg.fit(data[:,0].reshape(data.shape[0],1),data[:,1])
    return reg.predict(data[:,0].reshape(data.shape[0],1))

def getInOutIndex1(ypred,ytrue):
    in_res = []
    out_res = []
    err = ytrue - ypred
    std = err.std()
    #err_s = sorted(err)
    #q_25 = err_s[int(0.25 * len(err_s))]
    #q_75 = err_s[int(0.75 * len(err_s))]
    low = - std * 3
    high = std * 3
    for i in xrange(len(err)):
        if err[i] < low or err[i] > high:
            out_res.append(i)
        else:
            in_res.append(i)
    return in_res,out_res

def getInOutIndex(ypred,ytrue):
    in_res = []
    out_res = []
    err = ytrue - ypred
    
    err_s = sorted(err)
    q_25 = err_s[int(0.25 * len(err_s))]
    q_75 = err_s[int(0.75 * len(err_s))]
    low = q_25 - (q_75 - q_25) * 1.5
    high = q_75 + (q_75 - q_25) * 1.5
    for i in xrange(len(err)):
        if err[i] < low or err[i] > high:
            out_res.append(i)
        else:
            in_res.append(i)
    return in_res,out_res

def em(data):
    out_data = np.array([]).reshape(0,2)
   
    while True:
        pred = regression(data)
        in_index,out_index = getInOutIndex1(pred,data[:,1])
        if len(out_index) == 0:
            break
        out_data = np.concatenate((out_data,data[out_index,:]))
        data = data[in_index,:]
        break     
    return data,out_data
        
        

def clean_shop(shop_data,festivals):
    days = (config.PREDICTDATE - config.BASEDATE).days
    '''
    flag = False
    hist_min = 99999
    for day in range(days):
        date = config.BASEDATE + datetime.timedelta(days = day)        
        if flag and date not in shop_data and hist_min < 10:
            shop_data[date] = 0 
        if date in shop_data:
            flag = True
            hist_min = min(hist_min,shop_data[date])
    '''
    shop_data = filter(lambda x: x[0] not in festivals,shop_data.items())
    shop_data = sorted(shop_data,key = lambda x:x[0])
    shop_data = np.array(map(lambda x: [(x[0] - config.BASEDATE).days,x[1]],shop_data))
    in_data_all = np.array([]).reshape(0,2)
    out_data_all = np.array([]).reshape(0,2)
    for i in xrange(7):
        weekday_data = shop_data[i::7,:]
        in_data,out_data = em(weekday_data)
        in_data_all = np.concatenate((in_data_all,in_data))
        out_data_all = np.concatenate((out_data_all,out_data))
    in_data_all = map(lambda x: (config.BASEDATE + datetime.timedelta(days = x[0]),x[1]),in_data_all)
    out_data_all = map(lambda x: (config.BASEDATE + datetime.timedelta(days = x[0]),x[1]),out_data_all)
    return in_data_all,out_data_all
    

def clean(shop_pay,festivals):
    res = dict()
    i = 1
    for shop in shop_pay:
       in_data_all,out_data_all = clean_shop(shop_pay[shop],festivals) 
       res[shop] = (in_data_all,out_data_all)
    return res

def output(result):
    with open(os.path.join(config.FEATURE,'shop_per_day_clean'),'w') as f1,\
            open(os.path.join(config.FEATURE,'shop_out_point'),'w') as f2:
       for shop in result:
            for date,num in result[shop][0]:
                f1.write("%s,%s,%d\n"%(shop,date.strftime("%Y-%m-%d"),num)) 
            for date,num in result[shop][1]: 
                f2.write("%s,%s,%d\n"%(shop,date.strftime("%Y-%m-%d"),num)) 

def run():
    festivals = helper.loadFestival()
    shop_pay = helper.loadShopPay()
    res = clean(shop_pay,festivals)
    output(res)

if __name__ == '__main__':
    run()


