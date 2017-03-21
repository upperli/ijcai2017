#/bin/env python

import os
import sys
import numpy as np


dic = {}
arr = np.zeros(14)

i = 1
for root, dirs, files in os.walk(sys.argv[1]):
    for fn in files:
        if fn.endswith("csv"):
            with open(fn, "r") as fin:
                for line in fin:
                    segs = line.strip("\n").split(",")
                    paras = np.array([int(x) for x in segs[1:]])
                    shop_id = segs[0]
                    dic.setdefault(shop_id, np.zeros(14))
                    dic[shop_id] += paras 

for shop_id in dic:
    arr = [str(int(x)) for x in dic[shop_id]]
    print "%s,%s" %(shop_id, ",".join(arr))
