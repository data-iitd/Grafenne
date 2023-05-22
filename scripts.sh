python cora_train.py --data=$1 --missing_rate=0 --categorical=1 --result_file=$result_file --gpu=0 --verbose=1 --num_epochs=300 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1
python cora_train.py --data=$1 --missing_rate=0.5 --categorical=1 --result_file=$result_file --gpu=0 --verbose=1 --num_epochs=300 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1
python cora_train.py --data=$1 --missing_rate=0.9 --categorical=1 --result_file=$result_file --gpu=0 --verbose=1 --num_epochs=300 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1
python cora_train.py --data=$1 --missing_rate=0.99  --result_file=$result_file --gpu=0 --verbose=1 --num_epochs=300 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1
bs_train_nbd=512  #for Neigh mean
bs_train_nbd## 512 for 0.5 , 256 and 0.02 for 0.9 , 256 and 0.01 for 0.99

fognn citeseer -1 (0,.5,.99(categorical 0)), 512 (0.9)
sh repeat_code_fognn.sh CiteSeer 0.5 0 100 512 1
NF citeseer cutoff = 0.001 2 layer fognn
FP citeseer cutoff = 0.01 ( 1 layer fognn)


python cora_train.py --data=CiteSeer --missing_rate=0.99 --edge_value_thresh=0.01 --imputation='fp' --categorical=1 --result_file=tmp.txt --gpu=2 --verbose=1 --num_epochs=100 --num_layers=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.4

Actor
sh repeat_code_fognn.sh Actor 0 0 100 512 1
sh repeat_code_fognn.sh Actor 0.5 1 100 512 1
 sh repeat_code_fognn.sh Actor 0.9 2 100 512 1
 sh repeat_code_fognn.sh Actor 0.99 3 200 -1 1
 NF = 0.001 sh repeat_code_fognn.sh Actor 0.5 1 300 -1 1
 FP = 0.01 sh repeat_code_fognn.sh Actor 0.5 1 300 512 1


Winconsin
sh repeat_code_fognn.sh Wisconsin 0 0 300 -1 1  (weight.= 0.001)
sh repeat_code_fognn.sh Wisconsin 0.5 0 300 -1 0 0  (0 weight)
sh repeat_code_fognn.sh Wisconsin 0.9 3 300 -1 1 0
sh repeat_code_fognn.sh Wisconsin 0.99 3 300 -1 1 0 

NF weight 0 and same as before

FP weight 0.001 : 99% missing , 0 for remaining






GAT
 sh repeat_code_fognn.sh Cora 0 3 100 512 1 
 sh repeat_code_fognn.sh Cora 0.5 3 100 512 1 
 sh repeat_code_fognn.sh Cora 0.9 1 100 256 1
 sh repeat_code_fognn.sh Cora 0.99 3 100 512 0


 sh repeat_code_fognn.sh CiteSeer 0 3 100 512 1 
 sh repeat_code_fognn.sh CiteSeer 0.5 3 100 512 1 
 sh repeat_code_fognn.sh CiteSeer 0.9 1 100 512 1
 sh repeat_code_fognn.sh CiteSeer 0.99 2 100 512 0
 


GIN 


 sh repeat_code_fognn.sh Cora 0 3 100 512 1 
 sh repeat_code_fognn.sh Cora 0.5 3 100 512 1 
 sh repeat_code_fognn.sh Cora 0.9 1 100 512 1
 sh repeat_code_fognn.sh Cora 0.99 2 100 512 0
 
 
 sh repeat_code_fognn.sh Wisconsin 0.99 2 100 64 1
 
 
 
python cora_train.py --data="Cora" --missing_rate=0.99 --gpu=0 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1
python cora_train.py --data="Cora" --missing_rate=0.9 --gpu=0 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1
python cora_train.py --data="Cora" --missing_rate=0.5 --gpu=0 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1
python cora_train.py --data="Cora" --missing_rate=0 --gpu=0 --verbose=1 --num_epochs=200 --num_layers=1  --categorical=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.4

python cora_train.py --data="CiteSeer" --missing_rate=0.99 --gpu=0 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.4
python cora_train.py --data="CiteSeer" --missing_rate=0.9 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=512 --bs_test_nbd=-1 --categorical=1 --drop_rate=0.4
python cora_train.py --data="CiteSeer" --missing_rate=0.5 --gpu=2 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=512 --bs_test_nbd=-1 --categorical=1 --drop_rate=0.4
python cora_train.py --data="CiteSeer" --missing_rate=0 --gpu=0 --verbose=1 --num_epochs=200 --num_layers=2  --categorical=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.4


python cora_train.py --data="DE" --missing_rate=0.99 --gpu=0 --verbose=1 --num_epochs=200 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --categorical=1
python cora_train.py --data="DE" --missing_rate=0.9 --gpu=1 --verbose=1 --num_epochs=200 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --categorical=1
python cora_train.py --data="DE" --missing_rate=0.5 --gpu=2 --verbose=1 --num_epochs=200 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --categorical=1
python cora_train.py --data="DE" --missing_rate=0 --gpu=0 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1  --categorical=1 

python cora_train.py --data="Cora" --missing_rate=0.99 --gpu=0 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1



LP
python cora_train-with_LP.py  --data="Cora" --missing_rate=0.99 --gpu=2 --verbose=1 --num_epochs=1 --num_layers=2  --bs_test_nbd=-1   --plain_lp True --lp_alpha 0.9 --lp_iters 500


LP with logits
ython cora_train-with_LP.py  --data="Cora" --missing_rate=0.99 --gpu=2 --verbose=1 --num_epochs=200 --num_layers=2  --bs_test_nbd=-1   --lp_with_logits True --lp_alpha 0.99 --lp_iters 500

python cora_graphsage.py --data="mimic3" --missing_rate=0 --gpu=0  --num_epochs=100  
python cora_graphsage.py --data="Amazon_Photo" --missing_rate=0 --gpu=0  --num_epochs=100  


python cora_train.py --data="mimic3" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=200 --num_layers=1  --categorical=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.2
54.20% python cora_graphsage.py --data="mimic3_icd" --missing_rate=0 --gpu=0  --num_epochs=100 
50.8 50%
45.51 90%
40.68% 99%
57.24% python cora_train.py --data="mimic3" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=200 --num_layers=1  --categorical=0 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.2
57.89% ython cora_train.py --data="mimic3" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=200 --num_layers=1  --categorical=0 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.4
58.37% python cora_train.py --data="mimic3_icd" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=200 --num_layers=1  --categorical=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.4
python cora_train.py --data="mimic3_icd" --missing_rate=0.99 --gpu=1 --verbose=1 --num_epochs=200 --num_layers=2  --categorical=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.4



python cora_graphsage.py --data="mimic3_expire" --missing_rate=0 --gpu=2  --num_epochs=100 
python cora_train.py --data="mimic3_expire" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=200 --num_layers=1  --categorical=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.4


python baseline_models.py --data="Cora" --missing_rate=0 --gpu=2 --num_epochs=100 --result_file="results/tp" --model_name="pagnn"


python cora_graphsage.py --data="flipkart_10_10" --missing_rate=0 --gpu=2  --num_epochs=100 
python cora_train.py --data="flipkart_10_10" --missing_rate=0 --gpu=3 --verbose=1 --num_epochs=200 --num_layers=1  --categorical=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.4


python cora_graphsage.py --data="flipkart_50" --missing_rate=0 --gpu=2  --num_epochs=100 



python cora_train.py --data='Cora' --missing_rate=0 --categorical=1 --result_file=$result_file --gpu=0 --verbose=1 --num_epochs=300 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1

%74
python cora_train_scale.py --data="CiteSeer" --missing_rate=0 --categorical=1 --gpu=0 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2
Namespace(data='CiteSeer', gpu=0, missing_rate=0.0, categorical=True, verbose=True, num_epochs=1000, num_layers=2, bs_train_nbd=256, bs_test_nbd=-1, drop_rate=0.2, result_file=None, edge_value_thresh=0.01, imputation='zero', heads=4, weight_decay=0)
train dataset, val dataset and test dataset  tensor(1996) tensor(665) tensor(666)


87.8%
python cora_train_scale.py --data="Cora" --missing_rate=0 --gpu=0 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=.4 --otf_sample=1 --fto_sample=1
83.2%
python cora_train_scale.py --data="Cora" --missing_rate=0.9 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --otf_sample=1 --fto_sample=1 --num_obs_samples=100 --num_feat_samples=30 --use_data_x_otf=1 --otf_sample_testing=1

84%
python cora_train_scale.py --data="Cora" --missing_rate=0.9 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --otf_sample=1 --fto_sample=1 --num_obs_samples=100 --num_feat_samples=30 --use_data_x_otf=1 --otf_sample_testing=1

python cora_train_scale.py --data="Cora" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --otf_sample=1 --fto_sample=1 --num_obs_samples=100 --num_feat_samples=30 --use_data_x_otf=1

python cora_train_scale.py --data="CiteSeer" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --otf_sample=1 --fto_sample=1 --num_obs_samples=100 --num_feat_samples=30 --use_data_x_otf=1 --otf_sample_testing=1


82% for full
71 for 0.9 missing rates


python cora_train_scale.py --data="Amazon_Photo" --missing_rate=0.99 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --otf_sample=1 --fto_sample=1 --num_obs_samples=100 --num_feat_samples=20 --use_data_x_otf=0 --otf_sample_testing=1

python cora_train_scale.py --data="Amazon_Photo" --missing_rate=0.99 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 

python cora_train_scale.py --data="Amazon_Computers" --missing_rate=0.99 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --otf_sample=1 --fto_sample=1 --num_obs_samples=100 --num_feat_samples=30 --use_data_x_otf=1 --otf_sample_testing=1

python cora_graphsage.py --data="Physics" --missing_rate=0 --gpu=2  --num_epochs=100 


python cora_train_scale.py --data="Physics" --missing_rate=0 --gpu=2 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=30 --num_feat_samples=30 --use_data_x_otf=1 --otf_sample_testing=1



python cora_train_scale.py --data="Physics" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --categorical=True 


python cora_train_scale.py --data="CS" --missing_rate=0.99 --gpu=2 --verbose=1 --num_epochs=4000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --categorical=True 

--otf_sample=1 --num_obs_samples=30 --num_feat_samples=30 --use_data_x_otf=1 --otf_sample_testing=1

0 96.94
0.5 96.41
0.9 95.49
.99 92.75
gsage 94.3%
python cora_train_scale.py --data="Physics" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.4 --otf_sample=1 --num_obs_samples=50 --num_feat_samples=50 --use_data_x_otf=1 --otf_sample_testing=1

python cora_train_scale.py --data="Squirrel" --missing_rate=0 --gpu=2 --verbose=1 --num_epochs=4000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.4 --otf_sample=1 --num_obs_samples=30 --num_feat_samples=40 --use_data_x_otf=1 --otf_sample_testing=1

python cora_train_scale.py --data="Physics" --missing_rate=0 --gpu=0 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.3 --otf_sample=1 --fto_sample=1 --num_obs_samples=100 --num_feat_samples=100 --use_data_x_otf=1 --otf_sample_testing=1


python cora_train_scale.py --data="Physics" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --otf_sample_testing=1


python cora_train_scale.py --data="Chameleon" --missing_rate=0 --gpu=2 --verbose=1 --num_epochs=4000 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.4 


python cora_train.py --data='Chameleon' --missing_rate=0.99 --result_file=$result_file --gpu=0 --verbose=1 --num_epochs=3000 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1

python cora_graphsage.py --data="Chameleon" --missing_rate=0.99 --gpu=2  --num_epochs=100 

--otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --use_data_x_otf=1 --use_data_x_fto=1 --otf_sample_testing=1
python cora_train_scale.py --data="Chameleon" --missing_rate=0 --gpu=2 --verbose=1 --num_epochs=4000 --num_layers=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.4 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --use_data_x_otf=0 --use_data_x_fto=0 --otf_sample_testing=1


python cora_train_scale.py --data="Physics" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --otf_sample_testing=1
this is good as well 96.6
python cora_train_scale.py --data="Physics" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --otf_sample_testing=1
92.7%
python cora_train_scale.py --data="Physics" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --otf_sample_testing=1
this is good
python cora_train_scale.py --data="Chameleon" --missing_rate=0.99 --gpu=2 --verbose=1 --num_epochs=4000 --num_layers=1 --bs_train_nbd=256 --bs_test_nbd=-1 --drop_rate=0.2 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --use_data_x_otf=0 --use_data_x_fto=0 --otf_sample_testing=1



88%
python cora_train_scale.py --data="Cora" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=6000 --num_layers=2 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.2 --otf_sample=1 --fto_sample=1 --num_obs_samples=30 --num_feat_samples=30  --otf_sample_testing=1 --categorical=True


python cora_graphsage.py --data="CS" --missing_rate=0.99 --gpu=2  --num_epochs=100 

python cora_train_scale.py --data="CS" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --otf_sample_testing=1


python cora_train_scale.py --data="CS" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=30 --num_feat_samples=30 --otf_sample_testing=1


python cora_train_scale.py --data="CS" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.4 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=50 --num_feat_samples=50 --otf_sample_testing=1


python cora_train_scale.py --data="CS" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=5 --num_feat_samples=5 --otf_sample_testing=1


python cora_train_scale.py --data="Physics" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=3 --num_feat_samples=3 --otf_sample_testing=1

after sampling change 
96.8% 96.3 for normal graphsage
python cora_train_scale.py --data="Physics" --missing_rate=0 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=10 --num_feat_samples=10 --otf_sample_testing=1
94.4%   0.9414 for normal graphsage
python cora_train_scale.py --data="CS" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --otf_sample_testing=1

95.22.  94.31%
python cora_train_scale.py --data="Photo" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=2048 --bs_test_nbd=-1 --drop_rate=0 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=100 --num_feat_samples=100 --otf_sample_testing=1


python cora_train_scale.py --data="Physics" --missing_rate=0.99997 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 

87.8%
python cora_train_scale.py --data="Cora" --missing_rate=0 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=30 --num_feat_samples=30 --sampling_in_loop=1


Link Prediction tasks

python cora_train_scale.py --data="Cora" --missing_rate=0 --gpu=2 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.1 --otf_sample=0 --fto_sample=0 --categorical=True
python cora_link.py --data="Cora" --missing_rate=0 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --categorical=True --otf_sample=0 --fto_sample=0 --num_obs_samples=30 --num_feat_samples=30 --sampling_in_loop=0

python cora_link_graphsage.py --data="Cora" --missing_rate=0 --gpu=2  --num_epochs=1000 


python cora_link.py --data="Cora" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.2 --otf_sample=1 --fto_sample=1 --num_obs_samples=30 --num_feat_samples=30 --sampling_in_loop=0

python cora_link_graphsage.py --data="Cora" --missing_rate=0.99 --gpu=2  --num_epochs=1000 
sh repeat_code_link.sh Cora 0.5 3 300 0.001 (drop rate 0 is working)
cora 85%
    python cora_link.py --data=$1 --categorical=True --imputation='zero' --result_file=$result_file --missing_rate=$2 --gpu=3  --num_epochs=500 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.4 --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --verbose=1 --num_feat_samples=15 --sampling_in_loop=0
CiteSeer 0.3
    python cora_link.py --data=$1 --imputation='zero' --result_file=$result_file --missing_rate=$2 --gpu=3  --num_epochs=500 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.3 --otf_sample=0 --fto_sample=0 --num_obs_samples=30 --verbose=1 --num_feat_samples=15 --sampling_in_loop=0
Citeseer 0.99--> 2 layer droprate 0.2
python cora_link.py --data=$1 --categorical=True --imputation='zero' --result_file=$result_file --missing_rate=$2 --gpu=3  --num_epochs=500 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.3 --otf_sample=1 --fto_sample=1 --num_obs_samples=10 --verbose=1 --num_feat_samples=10 --sampling_in_loop=0


python cora_link_graphsage.py --data="Wisconsin" --missing_rate=0 --gpu=2  --num_epochs=1000  --verbose=1

84.5+
python cora_link.py --data="Cora" --missing_rate=0.5 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.1 --categorical=True --otf_sample=0 --fto_sample=0 
also change [20,15] to [-1,-1]

cora+nf 0.01 thresh [20,15] , 0.1 1024 batch size, 

cora+nf 0.0001 remaning same sh repeat_code_link.sh Cora 0.99 2 200 0.0001
batch size = -1
sh repeat_code_link.sh CiteSeer 0.5 2 200 0.001
sh repeat_code_link.sh CiteSeer 0.99 2 200 0.0001 drop rate 0.3
sh repeat_code_link.sh Actor 0.5 2 200 0.0001  0.1 drop rate
Cora droprate 0.1 , num epochs = 100


fp 
sh repeat_code_link.sh CiteSeer 0.5 2 200 0.01   drop rate 0.1
sh repeat_code_link.sh Actor 0.5 3 200 0.001 this is with categorical flag on drop rate 0.1

in case of 0.99, switch off the categorical flag and threshold 0.01

drop rate 0 sh repeat_code_link.sh Cora 0.99 2 500 0.01


CiteSeer 0 0.5 512 hidden size, num layer 1 1000 epochs drop rate 0.2
citeseer 0.99 droprate=0.1, num_epochs=1000 num layers =2
Actor 500 epochs 1 layer droprate 0

gin 
CiteSeer 0 0.5 512 hidden size, num layer 1 1000 epochs drop rate 0.2
citeseer 0.9 0.99 512, 2 300 drop rate 0 lr - 00001, general lr: 0.001
Actor 2 layer 512 drop rate 0.2

Cora 0.9. 0.99 layer 1 droprate 0.1 1000 epochss 256


large scale graphs



python cora_graphsage.py --data="Physics" --missing_rate=0.99 --gpu=2  --num_epochs=100 

python cora_train_scale.py --data="Physics" --missing_rate=0 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=10 --num_feat_samples=10 --sampling_in_loop=0
python cora_train_scale.py --data="OGBN-Products" --missing_rate=0 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=2 --bs_train_nbd=1024 --bs_test_nbd=1024 --drop_rate=0.2 --categorical=False --otf_sample=1 --fto_sample=1 --num_obs_samples=5 --num_feat_samples=5 --sampling_in_loop=1


python cora_train_scale.py --data="Physics" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --sampling_in_loop=0
python cora_train_scale.py --data="Physics" --missing_rate=0.5 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=2048 --bs_test_nbd=-1 --drop_rate=0.5 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=30 --num_feat_samples=30 --sampling_in_loop=1


python cora_train_scale.py --data=Physics --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.4 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=10 --num_feat_samples=10 --sampling_in_loop=0
sh repeat_code_layer1_scale.sh Physics 0.5 1 30 2048 1 0.1  with 30 samples
sh repeat_code_layer1_scale_nf.sh Physics .5 2 30 2048 1 0  (this works better without sampling)
sh repeat_code_scale.sh Physics 0 0 30 2048 1 0.4
sh repeat_code_layer1_scale.sh Physics 0.99999 2 30 1024 0 0.2
python cora_train_scale.py --result_file=$result_file --data=$1 --missing_rate=$2 --gpu=$3 --verbose=1 --num_epochs=$4 --num_layers=1 --bs_train_nbd=$5 --bs_test_nbd=-1 --drop_rate=$7  --otf_sample=0 --fto_sample=0 
python cora_train_scale.py --data="Physics" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=30 --num_layers=1 --bs_train_nbd=2048 --bs_test_nbd=-1 --drop_rate=0.2 --otf_sample=0 --fto_sample=0 

sh repeat_code_layer1_scale.sh Computer 0.99999 3 50 1024 0 0.1

sh repeat_code_scale.sh Computer 0 0 30 2048 1 0.4

python cora_train_scale.py --data=Computer --missing_rate=0 --gpu=0 --verbose=1 --num_epochs=100 --num_layers=1 --bs_train_nbd=4096 --bs_test_nbd=-1 --drop_rate=0.1 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --sampling_in_loop=0

sh repeat_code_layer1_scale.sh Computer 0.99 0 100 1024 0 0.1



grafenne +fp 
sh repeat_code_layer1_scale_fp.sh Physics 0.99999 3 30 1024 0 0.1

sh repeat_code_layer1_scale_nf.sh Physics 0.99999 2 30 1024 0 0.1

python cora_train_scale.py --data="Physics" --imputation='fp' --missing_rate=0.99999 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.1 --categorical=True --otf_sample=0 --fto_sample=0 --num_obs_samples=30 --num_feat_samples=30 --sampling_in_loop=1
sh repeat_code_layer1_scale_fp.sh Physics 0.99 3 30 1024 0 0.1    
0.5 with sampling 50, threshold 0.1

grafenne+nf
sh repeat_code_layer1_scale_nf.sh Physics 0.99 2 30 1024 0 0.1

edge thresh = 0.00001 (different for 0.5)


Computers

fp+grafeene edge threshold =0.001 for 0.99999
sh repeat_code_layer1_scale_fp.sh Computer 0.99999 3 100 1024 0 0.1



BASELINE MODELS
python baseline_models_link.py --data="Cora" --missing_rate=0 --gpu=2  --num_epochs=100 --model_name='gcnmf'
