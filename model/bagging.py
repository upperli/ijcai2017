#conding=utf-8
#-*- coding:utf-8 -*-

import preprocess
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import linear_model
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib
import random
import time
import datetime
import time
import sys
import os

import load_model

sys.path.append('..')
from utils import config

i = 0

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
    ypred = np.array([max(0, x) for x in ypred])
    return np.mean(np.abs((ypred - ytrue)/(ypred + ytrue)))

def metric(ypred,ytrue):
    #return 'loss',evaluate(np.exp(ypred),ytrue.get_label())
    return 'loss',evaluate(ypred,ytrue.get_label())

def resample(train_x,train_y):
    #weight = np.log(train_y) / trian_y
    #w = weight / weight.sum()
    rand = np.random.randint(0,train_y.shape[0],train_y.shape[0])
    return train_x[rand],train_y[rand]

def metric_lgb(ypred, ytrue):
    #return 'loss',evaluate(np.exp(ypred),ytrue.get_label())
    return 'loss',evaluate(ypred,ytrue.get_label()), False

def fitLGBModel(train_x, train_y, valid_x, valid_y, param_dic):
  lgb_train = lgb.Dataset(train_x, train_y)
  lgb_eval = lgb.Dataset(valid_x, valid_y)
  params={"bagging_freq" : 100,
          "xgboost_dart_mode" : True,
          "num_threads":40,
          "verbosity":0
          }
  params.update(param_dic)
  clf = lgb.train(params, lgb_train, num_boost_round=1200, valid_sets=lgb_eval,
                  feval=metric_lgb, early_stopping_rounds=20)
  pred = clf.predict(valid_x)
  print "LGB Eval:", evaluate(pred, valid_y)
  return clf, evaluate(pred, valid_y)

def fitGBRTModel(train_x, train_y, valid_x, valid_y, param_dic):
  train_x[np.isnan(train_x)] = -1
  valid_x[np.isnan(valid_x)] = -1
  rf_config={"loss":"lad",
             "learning_rate":0.02,
             "n_estimators":300,
             "max_depth":8,
            # "criterion":"mae",
             "min_samples_split":2, 
             "subsample":0.8,
             "max_features":"sqrt"}

  rf_config.update(param_dic)
  clf = GradientBoostingRegressor(**rf_config)
  clf.fit(train_x, train_y)
  pred = clf.predict(valid_x)
  print "GBRT Evaluate:", evaluate(pred,valid_y)

  return clf, evaluate(pred,valid_y)

def fitRFModel(train_x, train_y, valid_x, valid_y, param_dic):
  train_x[np.isnan(train_x)] = -1
  train_x[np.isnan(train_x)] = -1
  valid_x[np.isnan(valid_x)] = -1
  rf_config={"n_estimators":300,
      "criterion":'mse',
      #"max_features":'sqrt',
      "oob_score":True,
      "n_jobs":-1, 
      "verbose":0} 

  rf_config.update(param_dic)

  clf = RandomForestRegressor(**rf_config)
  clf.fit(train_x, train_y)
  pred = clf.predict(valid_x)
  print "RF Evaluate:", evaluate(pred,valid_y)
  return clf, evaluate(pred,valid_y)

def fitXGBModel(train_x,train_y,valid_x,valid_y, param_dic):
    params = {
                'objective':myObjective6,
                'max_depth':7, 
                #'learning_rate':0.007,
                'learning_rate':0.02,
                'n_estimators':500,
                'gamma':0.8, 
                'min_child_weight':2, 
                'max_delta_step':0, 
                'subsample':0.8, 
                'colsample_bytree':0.8, 
                'colsample_bylevel':0.9,
                'base_score':10,
                'seed':1
            } 
    params.update(param_dic)

    #train_x = np.concatenate((train_x,valid_x))
    #train_y = np.concatenate((train_y,valid_y))
    reg = xgb.XGBRegressor(**params)
    #reg.fit(train_x,train_y,eval_metric=metric,eval_set=[(train_x,train_y),(valid_x,valid_y)], early_stopping_rounds=30)
    reg.fit(train_x,train_y,eval_metric=metric,eval_set=[(train_x,train_y),(valid_x,valid_y)], verbose=100, early_stopping_rounds=10)
    #reg.fit(train_x,train_y)
    #reg.fit(train_x,np.log(train_y),eval_metric=metric,eval_set=[(train_x,train_y),(valid_x,valid_y)])

    pred = reg.predict(valid_x)
    valid_y_exp = valid_y
    print evaluate(pred,valid_y_exp)

    return reg, evaluate(pred,valid_y_exp)
    
def fitLRModel(train_x, train_y, valid_x, valid_y, param_dic):
    train_x[np.isnan(train_x)] = -1
    valid_x[np.isnan(valid_x)] = -1
    rf_config={"fit_intercept":True,
               "normalize":True,
               "n_jobs":-1}
    rf_config.update(param_dic)
    clf = LinearRegression(**rf_config)
    clf.fit(train_x, train_y)
    pred = clf.predict(valid_x)
    print "LR Eval:", evaluate(pred, valid_y)
    return clf, evaluate(pred,valid_y)

def fitRidgeModel(train_x, train_y, valid_x, valid_y, param_dic):
    train_x[np.isnan(train_x)] = -1
    valid_x[np.isnan(valid_x)] = -1
    rf_config={"alpha":0.9,
               "solver":"svd",#["auto", "svd", "cholesky", "lsqr", "sparse_cg", "cag"]
               "max_iter":10000
              }
    rf_config.update(param_dic)
    #for sol in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
    best_min = 1000
    for sol in ["svd", "cholesky"]:
      rf_config["solver"]  = sol
      clf = Ridge(**rf_config)
      clf.fit(train_x, train_y)
      pred = clf.predict(valid_x)
      if evaluate(pred, valid_y) < best_min:
        best_min = evaluate(pred, valid_y)
      print sol, "Ridge Evaluate:", evaluate(pred,valid_y)
    return clf, best_min 

def fitLinearSVRModel(train_x, train_y, valid_x, valid_y, param_dic):
  train_x[np.isnan(train_x)] = -1
  valid_x[np.isnan(valid_x)] = -1
  rf_config={"C":0.9,
             "loss":"squared_epsilon_insensitive",
             "epsilon":0.1,
             "dual":False,
             "verbose":6,
             "max_iter":4000
      }
  rf_config.update(param_dic)
  clf = LinearSVR(**rf_config)
  clf.fit(train_x, train_y)
  pred = clf.predict(valid_x)
  print "Evaluate:", evaluate(pred,valid_y)
  return clf, evaluate(pred, valid_y)

def fitXGBLRModel(train_x, train_y, valid_x, valid_y):
    reg = fitXGBModel(train_x,train_y,valid_x,valid_y)
    trans = load_model.Transformer()
    trans.load_model("model.xgb.read")
    print "Transform Begin:"
    train_x = np.concatenate((trans.transform_from_api(train_x), train[1]), axis=1)
    valid_x = np.concatenate((trans.transform_from_api(valid_x), valid[1]), axis=1)
    print "Transform End:"
    #reg_1 = fitRidgeModel(train_x,train_y,valid_x,valid_y)
    reg_1 = fitRidgeModel(train_x,train_y,valid_x,valid_y)
    return reg, eval

def output_test(pred, test_comm, gap, model_type, eva, file_path):
    res = dict()
    for p,c in zip(pred,test_comm):
        shop,date = c
        idx = (date - config.PREDICTDATE).days
        res.setdefault(shop,[0] * 14)
        res[shop][idx] = p

    with open(os.path.join(file_path,'res_%d_%s_%.6f.csv'%(gap, model_type, eva)),'w') as fout:
        for shop in res:
            fout.write("%s,%s\n"%(shop,",".join(map(lambda x:str(int(x + 0.5)),res[shop]))))

def bagging(file_path, output_path, model_type, begin_day, end_day, expansion=False, save_model=True, 
            output=True, online=False, steps=1000, train_set="base", valid_set="base", test_set="base", 
	    cv=4, shuffle=False, weight=[50, 15, 15, 5], cla=0):
  final_lis = [] 
  for gap in range(begin_day, end_day):
    print "*"*20, gap, "*"*20
    print "Gap %d Start Time:"%gap, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
    data, comm = preprocess.loadData(gap, expansion=expansion)
    #train, test = preprocess.split(data, comm)
    train,valid,test_online = preprocess.split_lastweek(data, comm, cla)
    
    train_data = np.array(train[0])
    valid_data = np.array(valid[0])
    test_data = np.array(test_online[0])

    train_x = train_data[:,1:]
    train_y = train_data[:,0]
    valid_x = valid_data[:,1:]
    valid_y = valid_data[:,0]
    test_x = test_data[:,1:]
    valid_x_nonan = valid_x.copy()
    valid_x_nonan[np.isnan(valid_x_nonan)] = -1
    test_x_nonan = test_x.copy()
    test_x_nonan[np.isnan(test_x_nonan)] = -1
    if shuffle:
      idx = np.random.permutation(train_y.size)
      train_x = train_x[idx]
      train_y = train_y[idx]

    #train_len = train_data.shape[0]
    #win_size = train_len / cv + 1
    #print >> sys.stderr, train_len
    #print >> sys.stderr, win_size

    clfs = []

    application = ["regression_l2"]*2 + ["huber"] *2 + ["fair"] *6
    boosting = ["dart"] * 1 + ["gbrt"] * 11
    learning_rate = [0.015, 0.02, 0.03, 0.04]
    #metric = ["l1"] + ["l2"] + ["huber"] *4 + ["fair"] *3
    num_leaves = [32]*1 + [64] * 2 + [128] * 2 + [256] 
    feature_fraction = [0.5, 0.6, 0.7, 0.8, 0.9]
    bagging_fraction = [0.5, 0.6, 0.7, 0.8, 0.9]
    lambda_l1 = [0.5, 0.6, 0.7, 0.8, 0.9]
    lambda_l2 = [0.5, 0.6, 0.7, 0.8, 0.9]
    drop_rate = [0.3, 0.5, 0.7, 0.9]
    skip_drop = [0.3, 0.5, 0.7, 0.9]
    huber_delta = [0.6, 0.8, 0.9]
    fair_c = [0.6, 0.8, 0.9]
    max_bin = range(200, 400)
    feature_fraction_seed = range(1, 20)
    bagging_seed = range(1, 20)
    drop_seed = range(1, 20)
    
    for i in range(0, weight[1]):
    	dic = {"application": random.choice(application),
               "boosting": random.choice(boosting),
               "learning_rate": random.choice(learning_rate),
               "num_leaves": random.choice(num_leaves),
               "feature_fraction": random.choice(feature_fraction),
               "bagging_fraction": random.choice(bagging_fraction),
               "lambda_l1": random.choice(lambda_l1),
               "lambda_l2": random.choice(lambda_l2),
               "drop_rate": random.choice(drop_rate),
               "skip_drop": random.choice(skip_drop),
               "max_bin": random.choice(max_bin),
               "huber_delta": random.choice(huber_delta),
               "fair_c": random.choice(fair_c),
               "feature_fraction_seed" : random.choice(feature_fraction_seed),
               "bagging_seed" : random.choice(bagging_seed),
               "drop_seed" : random.choice(drop_seed)
              }
        clfs.append((fitLGBModel, dic))

    obj = [myObjective6] * 10 + ["reg:linear"]
    learning_rate = [0.015, 0.02, 0.03, 0.04]
    seed = range(1, 20)
    max_depth = range(5, 11)
    subsample = [0.5, 0.6, 0.7, 0.8, 0.9]
    colsample_bytree = [0.5, 0.6, 0.7, 0.8, 0.9]
    colsample_bylevel = [0.5, 0.6, 0.7, 0.8, 0.9]
    gamma = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for i in range(0, weight[0]):
    	dic = {"objective": random.choice(obj),
               "learning_rate": random.choice(learning_rate),
               "seed":random.choice(seed),
               "max_depth": random.choice(max_depth),
               "subsample": random.choice(subsample),
               "colsample_bytree": random.choice(colsample_bytree),
               "colsample_bylevel": random.choice(colsample_bylevel),
               "gamma" : random.choice(gamma)
              }
        clfs.append((fitXGBModel, dic))

    max_features = [0.5, 0.6, 0.65, 0.7, 0.8]
    max_depth = range(5, 11) 
    min_samples_leaf = [2, 10, 30, 50]
    random_state = range(0, 8)
    for i in range(0, weight[2]):
    	dic = {"max_features": random.choice(max_features),
               "max_depth": random.choice(max_depth),
               "min_samples_leaf": random.choice(min_samples_leaf),
               "random_state": random.choice(random_state)
              }
        clfs.append((fitRFModel, dic))

    loss = ["lad", "huber"]
    learning_rate = [0.01, 0.02, 0.03, 0.04]
    max_features = [0.5, 0.6, 0.65, 0.7, 0.8]
    max_depth = range(5, 11) 
    subsample = [0.5, 0.6, 0.7, 0.8]
    random_state = range(0, 8)
    for i in range(0, weight[3]):
    	dic = {"loss": random.choice(loss),
               "learning_rate": random.choice(learning_rate),
               "max_features": random.choice(max_features),
               "max_depth": random.choice(max_depth),
               "subsample": random.choice(subsample),
               "random_state": random.choice(random_state)
              }
        clfs.append((fitGBRTModel, dic))
	
    #stage2_train = np.zeros((train_x.shape[0], len(clfs)))
    stage2_valid = np.zeros((valid_x.shape[0], len(clfs)))
    stage2_test = np.zeros((test_x.shape[0], len(clfs)))

    bagging = np.random.randint(0, train_x.shape[0], size=(len(clfs), train_x.shape[0]))

    for idx, clf in enumerate(clfs):
      print "Gap", gap, "Model", idx
      print clf[1] 
      print "Start Time:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
      #skf = list(StratifiedKFold(train_y, cv, shuffle=True, random_state=idx))
      #stage2_valid_temp = np.zeros((valid_x.shape[0], len(skf)))
      #stage2_test_temp = np.zeros((test_x.shape[0], len(skf)))
      #train = train_x[bagging[idx]]
      #test = test_x[bagging[idx]]
      #for i, (train, test) in enumerate(skf):
        #print "Fold:", i, "Start Time:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
      X_train = train_x[bagging[idx]]
      y_train = train_y[bagging[idx]]
      print X_train.shape
      print y_train.shape
        #X_test = train_x[test]
        #y_test = train_y[test]
      if clf[0] in [fitLRModel, fitRidgeModel, fitLinearSVRModel, fitGBRTModel, fitRFModel]:
        X_train[np.isnan(X_train)] = -1
        valid_x_tmp = valid_x_nonan
        test_x_tmp = test_x_nonan
      else:
        valid_x_tmp = valid_x
        test_x_tmp = test_x
      reg, eva = clf[0](X_train, y_train, valid_x, valid_y, clf[1])
      if save_model:
        joblib.dump(reg, "%s/%d_%d.m"%(file_path, gap, idx))
        #stage2_train[test, idx] = reg.predict(X_test)
      #stage2_valid_temp[:, idx] = reg.predict(valid_x)
      #stage2_test_temp[:, i] = reg.predict(test_x_tmp)
      stage2_valid[:,idx] = reg.predict(valid_x_tmp) 
      stage2_test[:,idx] = reg.predict(test_x_tmp) 
      print "Gap ", gap,  "Model %d: %.6f"%(idx, evaluate(reg.predict(valid_x_tmp), valid_y))

    if gap==-1:
        for idx in range(len(clfs)):
            if clfs[idx][0] in [fitLRModel, fitRidgeModel, fitLinearSVRModel, fitGBRTModel, fitRFModel]:
                valid_x_tmp = valid_x_nonan
                test_x_tmp = test_x_nonan
            else:
                valid_x_tmp = valid_x
                test_x_tmp = test_x
            reg = joblib.load("%s/%d_%d.m"%(file_path, gap, idx))
            stage2_valid[:,idx] = reg.predict(valid_x_tmp) 
            stage2_test[:,idx] = reg.predict(test_x_tmp) 

    #print "Final LR:", "Start Time:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
    #reg1, eva1 = fitLRModel(stage2_train, train_y, stage2_valid, valid_y, {})
    #print "Final LR Eval", eva1
    #print "Final Ridge:", "Start Time:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
    #reg2, eva2 = fitRidgeModel(stage2_train, train_y, stage2_valid, valid_y, {})
    #print "Final Ridge Eval", eva2
    #print "Final XGB:", "Start Time:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
    #reg3, eva3 = fitXGBModel(stage2_train, train_y, stage2_valid, valid_y, 
    #                        {'max_depth':2, 'learning_rate':0.03, 'n_estimators':1000, 'seed':5, 
    #                         'gamma':0.9, 'subsample':1.0, 'colsample_bytree':1.0, 'colsample_bylevel':1.0})
    #print "Final XGB Eval", eva3
    eva_avg = evaluate(stage2_valid.mean(1), valid_y)
    eva_med = evaluate(np.median(stage2_valid, axis=1), valid_y)
    print "Final AVG Eval", eva_avg
    print "Final MED Eval", eva_med

    final_lis.append(np.min([eva_avg, eva_med]))

    #print "Best Eval", np.min([eva1, eva2, eva3]) 
    #final_lis.append(np.min([eva1, eva2, eva3]))

    #if save_model:
    #  joblib.dump(reg1, "%s/stage2_avg_%d.m"%(file_path, gap))
    #  joblib.dump(reg3, "%s/stage2_med_%d.m"%(file_path, gap))

    if output:
      #output_test(stage2_test.mean(1), test_online[1], gap, "avg", eva_avg, output_path)
      output_test(np.median(stage2_test, axis=1), test_online[1], gap, "median", eva_med, output_path)

    print "Gap %d Start Time:"%gap, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
  print "Week 1 Eval:", np.average(final_lis)

if __name__ == '__main__':
  file_path = "./Model_Save/Bagging_Version1.0"
  output_path = "./Model_Save/Bagging_Result_Version1.0"
  file_path0 = "./Model_Save/Bagging_Version2.0_0"
  output_path0 = "./Model_Save/Bagging_Result_Version2.0_0"
  file_path1 = "./Model_Save/Bagging_Version2.0_1"
  output_path1 = "./Model_Save/Bagging_Result_Version2.0_1"
  model_type = "lgb" #["lgb", "xgb", "rf", "gbrt"]
  begin_day = 0
  end_day = 7
  save_model = True
  output = True
  online = False
  steps = 1500
  A = "base"
  B = "美食"
  C = "其他"
  expansion = False
  stacking = True
  train_set = A 
  valid_set = A
  test_set = A 
  cv = 4
  shuffle = True
  weight = [20,10,10,0]
  bagging(file_path0, output_path0, model_type, begin_day, end_day, expansion=expansion, save_model=save_model, 
      output=output, online=online, steps=steps, train_set=train_set, valid_set= valid_set, 
      test_set=test_set, cv=cv, shuffle=shuffle, weight=weight, cla=1)    
  bagging(file_path0, output_path0, model_type, begin_day, end_day, expansion=expansion, save_model=save_model, 
      output=output, online=online, steps=steps, train_set=train_set, valid_set= valid_set, 
      test_set=test_set, cv=cv, shuffle=shuffle, weight=weight, cla=2)    
"""
  for i in [A, B, C]:
    train_set = i
    valid_set = i
    test_set = i
    if i != A:
      expansion = True
    else:
      expansion = False
    run(file_path, model_type, begin_day, end_day, expansion=expansion, save_model=save_model, 
        output=output, online=online, steps=steps, train_set=train_set, valid_set= valid_set, test_set=test_set)    
"""
