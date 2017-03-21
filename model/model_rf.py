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


def output_test(pred,test_comm,gap,t):
    res = dict()
    for p,c in zip(pred,test_comm):
        shop,date = c
        #date = datetime.datetime.strptime(date,'%Y-%m-%d')
        idx = (date - config.PREDICTDATE).days
        res.setdefault(shop,[0] * 14)
        res[shop][idx] = p
    path = os.path.join(config.DATA,str('rf'))
    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(path,'%d_gap_%d_res_xgb_%s.csv'%(t,gap,datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S") )),'w') as fout:
        for shop in res:
            tmp = res[shop]
            fout.write("%s,%s\n"%(shop,",".join(map(lambda x:str(int(max(0,x) + 0.5)),tmp))))

    
def gcv(X,y):
    params = {
                'objective':[myObjective6],
                'max_depth':[8], 
                'learning_rate':[0.05], 
                'n_estimators':[400], 
                'gamma':[0], 
                'min_child_weight':[1], 
                'max_delta_step':[0], 
                'subsample':[1], 
                'colsample_bytree':[0.7], 
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

def fitRFModel(train_x, train_y, valid_x, valid_y,seed,gap):
    rf_config={"n_estimators":1000,
      #"criterion":'mse',
      "max_features":0.1,
      #"max_features":'sqrt',
      #"max_depth":8,
      #"min_samples_split":50 + seed,
      #"min_samples_leaf":5,
      "random_state":seed,
      "n_jobs":-1,
      "verbose":6,
      "warm_start":True
    }
    #train_x = 
    train_x = np.concatenate((train_x,valid_x))
    train_y = np.concatenate((train_y,valid_y))
    clf = RandomForestRegressor(**rf_config)
    clf.fit(train_x, train_y)
    pred = clf.predict(valid_x)
    e = evaluate(pred,valid_y)
    '''
    last_e = 1
    for i in range(300):
        print "="*20, i, "="*20
        rf_config["n_estimators"] += 10
        clf.set_params(**rf_config)
        clf.fit(train_x, train_y)
        pred = clf.predict(valid_x)
        e = evaluate(pred,valid_y)
        if abs(last_e - e) < 1e-4:
            last_e = e
        else:
            break
        print "Evaluate:", e#, "OOB_SCORE:", clf.oob_score_
    '''
    
    return clf,e,None,pred





def fitModel(train_x,train_y,valid_x,valid_y,seed,gap):
    params = {
                'objective':myObjective6,
                'max_depth':7, 
                'learning_rate':0.03, 
                'n_estimators':400, 
                'gamma':0.1, 
                'min_child_weight':1, 
                'max_delta_step':0, 
                'subsample':0.8, 
                'colsample_bytree':0.7, 
                'colsample_bylevel':0.8,
                'base_score':4,
                'seed':seed
            } 
    if seed > 7:
        params['objective'] = 'reg:linear'
    #train_x = np.concatenate((train_x,valid_x))
    #train_y = np.concatenate((train_y,valid_y))
    reg = xgb.XGBRegressor(**params)
    #mask = train_y < 70
    #w = ~mask + mask *  np.log(train_y)/y * 70/np.log(70)
    reg.fit(train_x,train_y,eval_metric=metric,early_stopping_rounds=10,eval_set=[(train_x,train_y),(valid_x,valid_y)])
    pred = reg.predict(valid_x)
    valid_y_exp = valid_y
    #print sorted(reg._Booster.get_fscore(fmap = os.path.join(config.DATA,'%d/data.names'%gap)).items(),key = lambda x: -x[1])
    ''' 
    fout = open('model.txt','w')

    reg._Booster.dump_model(fout,fmap=os.path.join(config.DATA,'data.names'))

    ax =  xgb.plot_importance(reg._Booster)
    ax1 = xgb.plot_tree(reg._Booster,fmap=os.path.join(config.DATA,'data.names') ,num_trees=5,)
    ax2 = xgb.to_graphviz(reg._Booster,fmap=os.path.join(config.DATA,'data.names') ,num_trees=5,)
    ax.figure.savefig("importance.png")
    ax1.figure.savefig("tree.png")
    ax2.figure.savefig("gra.png")

    err_abs = np.abs(err)
    
    with open('err.csv','w') as fout:
        for y,e,ea,ep in zip(valid_y,err,err_abs,err_p):
            fout.write("%d,%f,%f,%f\n"%(y,e,ea,ep))
    '''
    pred = map(lambda x: int(max(0,x)),pred)
    err = np.abs(pred) - valid_y_exp
    err_p = np.abs(err) / (np.abs(pred) + valid_y_exp)
    e = evaluate(pred,valid_y_exp)

    print e
    
    return reg,e,err_p,pred

def run():

    err = []
    #for gap in xrange(14):
    for i in xrange(7):
        gap = i
        expansion = False
        data,comm = preprocess.loadData(gap,expansion)
        eps = []
        for t in [1,2]:
            train_data,valid_data,test_data = preprocess.split_lastweek(data,comm,t) 
            train = np.array(train_data[0])
            valid = np.array(valid_data[0])
            test = np.array(test_data[0])
            test_com = test_data[1]
    

            train_x = train[:,1:]
            train_y = train[:,0]

            train_x[np.isnan(train_x)] = -1 
            valid_x = valid[:,1:]
            valid_x[np.isnan(valid_x)] = -1 
            valid_y = valid[:,0]
            
            print train_x.shape
            print valid_x.shape 
            regs = []
            preds = []
            for seed in [1,2,3,5,7,9,11,15,20,25,30,40,80]:
            #for seed in [1]:
                reg,e,ep,pred = fitRFModel(train_x,train_y,valid_x,valid_y,seed,gap)    
                #eps += list(ep)
                regs.append(reg)
                preds.append(pred)
                #pred = map(lambda x: int(max(0,x)),pred)
                #er = np.abs(pred) - valid_y
                #err_p = np.abs(er) / (np.abs(pred) + valid_y)
                #eps += list(err_p)
            pred = np.median(preds,axis=0)
            er = np.abs(pred) - valid_y
            err_p = np.abs(er) / (np.abs(pred) + valid_y)
            eps += list(err_p)
            ''' 
            with open('%d_seed%d_gap%d_valid_with_svd'%(t,seed,gap),'w') as fout:
                for com,e,t,p in zip(valid_data[1],ep,valid_y,pred):
                    fout.write("%f\t%f\t%d\t%s\t%s\n"%(e,p,t,com[0],com[1].strftime("%Y-%m-%d")))
            '''
        #print train_x.shape
        #reg = gcv(train_x,train_y)
        #pred = reg.predict(valid_x)
        #e = evaluate(pred,valid_y)
        #print e
    
            if len(sys.argv) > 1 and sys.argv[1] == 'output':
                test_x = test[:,1:]
                test_x[np.isnan(test_x)] = -1 
                preds = []
                for reg in regs:
                    pred = reg.predict(test_x)
                    preds.append(pred)
                pred = np.median(preds,axis=0)
                output_test(pred,test_com,gap,t)
        e = np.mean(eps)
        print e
        err.append(e)

    print np.mean(err)
if __name__ == '__main__':
    run()    
