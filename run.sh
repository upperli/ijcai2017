#!/bin/sh

cd feature;./run.sh
cd -

cd model

python model.py output
python model_rf.py output
pyton bagging.py

./merge.sh

cd -

python rule.py


