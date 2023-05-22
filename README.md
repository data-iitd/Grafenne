# Grafenne
## Code


Use following command for training and testing on single snapshot data (low and medium size graphs)

`
python cora_train.py --data=CiteSeer --missing_rate=0.99 --result_file=tmp.txt --gpu=0 --verbose=1 --num_epochs=500 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1
`

or instead of running whole graph as single batch, run it in multiple batches
` 
python cora_train.py --data=CiteSeer --missing_rate=0.99 --result_file=tmp.txt --gpu=2 --verbose=1 --num_epochs=100 --num_layers=2 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.1
`

The above commands are CiteSeer, similar can be for other datasets such as Cora. The datasets used in paper are downloadable from pytorch geometric graph datasets.

also for lower missing rates, use --categorical=1 flag

`
python cora_train.py --data=Cora --missing_rate=0 --result_file=tmp.txt --gpu=0 --verbose=1 --num_epochs=500 --num_layers=2 --bs_train_nbd=1024 --bs_test_nbd=-1 --categorical=True
`
With FP
`
python cora_train.py --data=CiteSeer --missing_rate=0.99 --edge_value_thresh=0.01 --imputation='fp' --categorical=1 --result_file=tmp.txt --gpu=2 --verbose=1 --num_epochs=100 --num_layers=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.2
`
With NM

`python cora_train.py --data=CiteSeer --missing_rate=0.99 --edge_value_thresh=0.001 --imputation='nf' --categorical=1 --result_file=tmp.txt --gpu=2 --verbose=1 --num_epochs=100 --num_layers=2 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.1
`

`
python cora_train.py --data=Cora --missing_rate=0.99 --edge_value_thresh=0.01 --imputation='fp' --categorical=1 --result_file=tmp.txt --gpu=2 --verbose=1 --num_epochs=100 --num_layers=1 --bs_train_nbd=512 --bs_test_nbd=-1 --drop_rate=0.2
`

To run on large scale graphs like Physics, following commands can be run on various missing rates.

`
python cora_train_scale.py --data="Physics" --missing_rate=0 --gpu=1 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=1024 --bs_test_nbd=-1 --drop_rate=0.3 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --num_feat_samples=15 --sampling_in_loop=0
`

`
python cora_train_scale.py --data="Physics" --missing_rate=0.5 --gpu=3 --verbose=1 --num_epochs=1000 --num_layers=1 --bs_train_nbd=2048 --bs_test_nbd=-1 --drop_rate=0.5 --categorical=True --otf_sample=1 --fto_sample=1 --num_obs_samples=30 --num_feat_samples=30 --sampling_in_loop=1
`

`
python cora_train_scale.py --data="Physics" --missing_rate=0.99 --gpu=3 --verbose=1 --num_epochs=30 --num_layers=1 --bs_train_nbd=2048 --bs_test_nbd=-1 --drop_rate=0.2 --otf_sample=0 --fto_sample=0
`



