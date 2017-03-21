#coding=utf-8
import datetime

time_pred = datetime.datetime.strptime("2016-11-01","%Y-%m-%d")

def loadTrick(FILENAME):
    shop_id = set()
    with open(FILENAME, "r") as fin:
	for line in fin:
	    shop_id.add(int(line.strip()))
    return shop_id
def loadTrickWeek2(FILENAME):
    k = [1.0] * 2010
    with open(FILENAME, "r") as fin:
	for line in fin:
	    tmp = line.strip().split(",");	    
	    k[int(tmp[0])] = float(tmp[1])
    return k
def trick1_update(ret, shop_id, data, ratio):
    a = data[4]
    b = data[5]
    if ( data[0] / data[4] >= 3.0 ):
	ret[4] = a * 0.8
	ret[5] = b * 0.8
#	print ret, data[4]
	ret[11] = a* 0.8
	ret[12] = b* 0.8
#	print ret, data[4]
    return ret;

def loadWeather():
    weather = dict();
    with open("dataset/weather_all.csv", "r") as fin:
	for line in fin:
	    tmp = line.strip().split(",")
	    if ( weather.has_key(tmp[0]) == False ):
		weather[tmp[0]] = []
		for i in range(14):
		    weather[tmp[0]].append(dict())
	    days = (datetime.datetime.strptime(tmp[1],"%Y-%m-%d")- \
		    time_pred).days
	    if ( days >= 0 and days < 14 ):
	#	print tmp[0], days, tmp[4].decode('utf-8'), tmp[5].decode('utf-8')
		weather[tmp[0]][days]['max_deg'] = int(tmp[2])
		weather[tmp[0]][days]['min_deg'] = int(tmp[3])
		weather[tmp[0]][days]['type'] = tmp[4]
		weather[tmp[0]][days]['wind'] = tmp[6]
   
    weather_set = dict()
    wind_set = dict()
    deg_set = dict()
    for k,v in weather.items():
	for i in range(14):
	    weather_set.setdefault(v[i]['type'], 0);
	    wind_set.setdefault(v[i]['wind'], 0)
	    deg_set.setdefault(v[i]['max_deg'], 0)
	    deg_set.setdefault(v[i]['min_deg'], 0)
	    weather_set[v[i]['type']] = weather_set[v[i]['type']] + 1;
	    wind_set[v[i]['wind']] = wind_set[v[i]['wind']] + 1
	    deg_set[v[i]['max_deg']] = deg_set[v[i]['max_deg']] + 1 
	    deg_set[v[i]['max_deg']] = deg_set[v[i]['max_deg']] + 1
	   # print i,k, v[i]['type'].decode('utf-8'), v[i]['wind'].decode('utf-8')
    """
    for k,v in weather_set.items() :
	print k,v ;
    for k,v  in wind_set.items():
	print k,v
    for k,v in deg_set.items():
	print k,v 
    """
    ratio = dict()
    for k,v in weather.items():
	ratio[k] = [0] * 14
	for i in range(14):
	    w_type = v[i]['type'] 
	    wind = v[i]['wind']
	    max_deg = v[i]['max_deg']
	    min_deg = v[i]['min_deg']
	    
	    arg1 = 1
	    if '霾' in w_type or '雪' in w_type or '雨' in w_type:
		if '暴雨' in w_type or '大雨' in w_type:
		    arg1 = 0.95
		else:
		    arg1 = 0.97
	    else:
		if '晴' in w_type:
		    arg1 = 1.05
		else:
		    arg1 = 1.02

	    arg2 = 1

	    ratio[k][i] = arg1 * arg2; 
	    
    return ratio;

def loadShopInfo():
    shop_info = dict()
    with open("dataset/shop_info.txt", "r") as fin:
	for line in fin:
	    tmp = line.strip().split(",")
	
def loadShopInfo():
    shop_info = dict()
    with open("dataset/shop_info.txt", "r") as fin:
	for line in fin:
	    tmp = line.strip().split(",")
	    shop_info[int(tmp[0])] = tmp[1]
    return shop_info

    shop_info[int(tmp[0])] = tmp[1]
    return shop_info

def init():
    data = {}
    with open("../result/res.csv","r") as fin:
	for line in fin:
	    tmp = map(lambda x:int(x), line.strip().split(","));
	    data[tmp[0]] = tmp[1:]

    ratio = [1.01, 1.01, 1.03, 1.04, 1.03, 1.01, 0.97
	    #	1,  2,     3,	4,      5,    6,	 7
	    #print ret
	    ,1.01, 1.01, 1.02, 1.1, 1.03, 1.01, 0.97]
	    #	8,  9,	   10,	  11,   12,    13,     14,
    trick1 = loadTrick("dataset/weak_weekend.txt")
    trick_k = loadTrick("dataset/trick_k")
    trick_k_week2 = loadTrickWeek2("dataset/trick_k_week2")
    weather = loadWeather()
    shop_info = loadShopInfo()
    return data, ratio, weather, shop_info, trick1, trick_k, trick_k_week2

def run():

    data, ratio, weather, shop_info, trick1, trick_k, trick_k_week2 = init();
     
    fout = open("../result/total.csv", "w");
    for shop_id in range(1,2001):
	fout.write(str(shop_id)+",")
	ret = data[shop_id]

    ratio_weather = [1, 1, 1, 1.03, 1, 1, 0.97,1, 1, 1, 1.1, 1, 1, 0.97]
 
#	ret = map(lambda (x,y): x*y, zip(ret, ratio))
#	ret = map(lambda (x,y): x*y, zip(ret, weather[shop_info[shop_id]]))
	"""rule1 and rule2"""
	retio_wether = map(lambda (x,y): x*y, zip(ratio_weather, weather[shop_info[shop_id]]))
	ratio = map(lambda(x,y): (x+y)/2, zip(ratio, ratio_weather))
	ret = map(lambda(x,y): x*y, zip(ret, ratio))
    
	"""rule3"""
	if shop_id in trick1:
	    ret = trick1_update(ret,shop_id, data[shop_id], ratio)

	"""rule4"""
	#if shop_info[shop_id] == '西式快餐':
	#    ret[10] *= 1.5;

	"""rule5"""
	if shop_id == 1824:
	    ret = map(lambda x:x*1.3, ret)
	if shop_id == 810:
	    ret = map(lambda x:x*0.5, ret)
	if shop_id == 659:
	    ret = map(lambda x:x*0.8, ret)
	if shop_id == 1556:
	    ret = map(lambda x:x*1.35, ret)

	fout.write(",".join(map(lambda x:str(int(x)),ret))+"\n");
    fout.close();
    
if __name__=="__main__":
    run();
