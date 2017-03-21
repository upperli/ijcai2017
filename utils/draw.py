#coding=utf-8

import config
import os
import datetime
import numpy as np
import collections
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn import linear_model

def loadUserPay():
   user_pay = {}
   i = 0
   with open('../data.csv') as fin:
        for line in fin:
            tmp = line.strip().split("\t")
            date = datetime.datetime.strptime(tmp[1],"%Y-%m-%d")
            if ( config.PREDICTDATE - date).days > 100 or ( date - config.PREDICTDATE).days > 6:
                continue
            user_pay.setdefault(tmp[0],{})
            user_pay[tmp[0]].setdefault(date,0)
            user_pay[tmp[0]][date] += float(tmp[2])
            if i % 10000 == 0:
                print i
            i += 1
   return user_pay



def draw(s,data):
    data = np.array(sorted(data.items(),key = lambda x: x[0]))
    x = np.array(map(lambda x: [(x - config.BASEDATE).days],list(data[:,0])))
    y = data[:,1]
    plt.plot(x[:,0], y, color='b',linewidth=2)
    
    plt.savefig(os.path.join(config.PIC,'%s.png'%s))    
    plt.close('all')

def run():
    user_pay = loadUserPay()
    i = 0
    for s in user_pay:
        draw(s,user_pay[s])
        i += 1
        print i

if __name__ == '__main__':
    run()





