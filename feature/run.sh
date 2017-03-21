#!/bin/sh

echo "" > log

for i in `seq 0 6`
do
    echo  第$i ===================================== 
    
    python -u mklabel.py $i >> log
    python -u datefeature.py $i >> log
    python -u shopfeature.py $i >> log
 
    python -u shopcategory.py $i >> log
    
    python -u same_day_hist.py $i >> log

    python -u time_window.py $i 1 1 21 >> log 
    python -u time_window_log.py $i 1 1 21 >> log 
    python -u time_window.py $i 2 2 7 >> log 
    python -u time_window.py $i 2 4 7 >> log 
    python -u time_window.py $i 3 6 7 >> log 
    python -u time_window.py $i 7 14 7 >> log 
    python -u time_window.py $i 7 7 7 >> log 

    python -u time_window_diff.py $i 1 1 21 >> log 
    
    python -u weekday_stat.py $i 
    python -u weekend_stat.py $i 
    
    python -u min_max_stat.py $i 
    python -u air.py $i >> log
    python -u weather.py $i >> log
    #python -u city_cate2_history.py $i >> log
    #python -u locate_history.py $i >> log
    
    python -u join.py $i 
    
done

#for i in `seq 1 13`
#do
#    echo
#    cp ../../feature/0/day_feature ../../feature/$i/day_feature
#done
