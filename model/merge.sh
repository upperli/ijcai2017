#!/bin/sh


./shell.sh ../../result/xgb > ../../result/xgb_all.csv
./shell.sh ../../result/rf > ../../result/rf_all.csv


if [ ! -d "../../result/bagging" ]; then
    mkdir ../../result/bagging
fi

cp  ./Model_Save/Bagging_Result_Version2.0_1/* ../../result/bagging/
cp  ./Model_Save/Bagging_Result_Version2.0_0/* ../../result/bagging/

./shell.sh ../../result/bagging > ../../result/bagging_all.csv


python ens.py > ../../result/res.csv    
