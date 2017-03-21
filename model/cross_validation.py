#conding=utf-8

import preprocess
import numpy as np
import xgboost as xgb
from sklearn import linear_model
import datetime
import sys
import os
import random
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from  sklearn.ensemble import RandomForestRegressor
sys.path.append('..')
from utils import config
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

def myObjective6(ytrue, ypred):
    fair_constant = 100
    x = ypred - ytrue
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den*den)
    return grad, hess

def myObjective5(ytrue, ypred):
    x = ypred - ytrue
    grad = np.tanh(x)
    hess = 1.0 - grad * grad
    return grad, hess



def evaluate(ypred,ytrue):
    return np.mean(map(lambda x: np.abs(1.0 * (int(max(0,x[0])) - int(x[1])))/(int(max(0,x[0])) + int(x[1])) ,zip(ypred,ytrue)))

def metric(ypred,ytrue):
    return 'loss',evaluate(ypred,ytrue.get_label())


def output_test(pred,test_comm,gap):
    res = dict()
    for p,c in zip(pred,test_comm):
        shop,date = c
        #date = datetime.datetime.strptime(date,'%Y-%m-%d')
        idx = (date - config.PREDICTDATE).days
        res.setdefault(shop,[0] * 14)
        res[shop][idx] = p

    with open(os.path.join(config.RESULT,'gap%d_res_xgb_%s.csv'%(gap,datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")) ),'w') as fout:
        for shop in res:
            tmp = res[shop][:7]
            fout.write("%s,%s\n"%(shop,",".join(map(lambda x:str(int(max(0,x) + 0.5)),tmp +[0] * 7))))

    
def gcv(X,y):
    params = {
                'objective':[myObjective6],
                'max_depth':[8], 
                'learning_rate':[0.05], 
                'n_estimators':[200,400], 
                'gamma':[0], 
                'min_child_weight':[1], 
                'max_delta_step':[0], 
                'subsample':[1], 
                'colsample_bytree':[0.3,0.7], 
                'colsample_bylevel':[1],
                'base_score':[5],
            } 
    cv = ShuffleSplit(n_splits=5, test_size=0.04, random_state=1)
    def scoring(reg,X,y):
        pred = reg.predict(X)
        return -evaluate(pred,y)

    reg = xgb.XGBRegressor(**params)
    gc = GridSearchCV(reg, params,scoring=scoring,cv=cv,verbose=6)
    gc.fit(X,y)

    print 'best params:\n',gc.best_params_
    mean_scores = np.array(gc.cv_results_['mean_test_score'])
    print 'results:',mean_scores
    
    print 'best result:',gc.best_score_

    return gc

def run():

    for gap in xrange(7):
        expansion = False
        data,comm = preprocess.loadData(gap,expansion)
        train_data,test_data = preprocess.split(data,comm) 
        train = np.array(train_data[0])
        test = np.array(test_data[0])
        test_com = test_data[1]

        train_x = train[:,1:]
        train_y = train[:,0]
    
        print train_x.shape

        reg = gcv(train_x,train_y)

        if len(sys.argv) > 1 and sys.argv[1] == 'output':
            test_x = test[:,1:]
            pred = reg.predict(test_x)
            output_test(pred,test_com,gap)


if __name__ == '__main__':
    run()    
