#!/bin/sh
result_file='results/cora_train_'$1$2
rm $result_file
#$1 data, $2 missing rate $3 gpu, $4 epochs $5 train batch $6 categorical
for a in 1 2 3 4 5
do
    python cora_train.py --data=$1 --missing_rate=$2 --result_file=$result_file --gpu=3 --verbose=1 --num_epochs=300 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 
    #python cora_train.py --data=CiteSeer --missing_rate=0.99 --edge_value_thresh=0.01 --imputation='fp' --categorical=1 --result_file=tmp.txt --gpu=2 --verbose=1 --num_epochs=100 --num_layers=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.2
    # echo $a
    # if [ "$6" -eq "1" ] ;
    # then
    #     python cora_train.py --data=$1 --missing_rate=$2 --edge_value_thresh=0.01 --imputation='zero' --categorical=1 --result_file=$result_file --gpu=$3 --verbose=1 --num_epochs=$4 --num_layers=1 --bs_train_nbd=$5 --bs_test_nbd=-1 --drop_rate=0.4
    # else
    #     python cora_train.py --data=$1 --missing_rate=$2 --edge_value_thresh=0.01 --imputation='zero' --result_file=$result_file --gpu=$3 --verbose=1 --num_epochs=$4 --num_layers=2 --bs_train_nbd=$5 --bs_test_nbd=-1
    # fi
    # if [ "$6" -eq "1" ] ;
    # then
    #     python cora_train_gat.py --data=$1 --missing_rate=$2 --edge_value_thresh=0.01 --imputation='zero' --categorical=1 --result_file=$result_file --gpu=$3 --verbose=1 --num_epochs=$4 --num_layers=2 --bs_train_nbd=$5 --bs_test_nbd=-1
    # else
    #     python cora_train_gat.py --data=$1 --missing_rate=$2 --edge_value_thresh=0.01 --imputation='zero' --result_file=$result_file --gpu=$3 --verbose=1 --num_epochs=$4 --num_layers=2 --bs_train_nbd=$5 --bs_test_nbd=-1
    # fi
    # if [ "$6" -eq "1" ] ;
    # then
    #     python cora_train_gin.py --data=$1 --missing_rate=$2 --edge_value_thresh=0.01 --imputation='zero' --categorical=1 --result_file=$result_file --gpu=$3 --verbose=1 --num_epochs=$4 --num_layers=2 --bs_train_nbd=$5 --bs_test_nbd=-1
    # else
    #     python cora_train_gin.py --data=$1 --missing_rate=$2 --edge_value_thresh=0.01 --imputation='zero' --result_file=$result_file --gpu=$3 --verbose=1 --num_epochs=$4 --num_layers=2 --bs_train_nbd=$5 --bs_test_nbd=-1
    # fi
done    

python mean_std_results.py $result_file