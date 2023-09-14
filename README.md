# Grafenne
[Paper link](https://proceedings.mlr.press/v202/gupta23b/gupta23b.pdf)
[ICML link](https://icml.cc/Conferences/2023/Schedule?showEvent=24743)

## Citation

```

@InProceedings{pmlr-v202-gupta23b,
  title = 	 {{GRAFENNE}: Learning on Graphs with Heterogeneous and Dynamic Feature Sets},
  author =       {Gupta, Shubham and Manchanda, Sahil and Ranu, Sayan and Bedathur, Srikanta J.},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {12165--12181},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/gupta23b/gupta23b.pdf},
  url = 	 {https://proceedings.mlr.press/v202/gupta23b.html},
  abstract = 	 {Graph neural networks (GNNs), in general, are built on the assumption of a static set of features characterizing each node in a graph. This assumption is often violated in practice. Existing methods partly address this issue through feature imputation. However, these techniques (i) assume uniformity of feature set across nodes, (ii) are transductive by nature, and (iii) fail to work when features are added or removed over time. In this work, we address these limitations through a novel GNN framework called GRAFENNE. GRAFENNE performs a novel allotropic transformation on the original graph, wherein the nodes and features are decoupled through a bipartite encoding. Through a carefully chosen message passing framework on the allotropic transformation, we make the model parameter size independent of the number of features and thereby inductive to both unseen nodes and features. We prove that GRAFENNE is at least as expressive as any of the existing message-passing GNNs in terms of Weisfeiler-Leman tests, and therefore, the additional inductivity to unseen features does not come at the cost of expressivity. In addition, as demonstrated over four real-world graphs, GRAFENNE empowers the underlying GNN with high empirical efficacy and the ability to learn in continual fashion over streaming feature sets.}
}


```

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
 python cora_train.py --data=Cora --missing_rate=0 --categorical=1 --result_file=temp.txt --gpu=0 --verbose=1 --num_epochs=300 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1
`
or instead of whole graph as single batch, run in multi-batch by variyng bs_train_nbd
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



For link prediction tasks, 
`
python cora_link.py --data=$1 --categorical=True --imputation='zero' --result_file=$result_file --missing_rate=$2 --gpu=3  --num_epochs=500 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.4 --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --verbose=1 --num_feat_samples=15 --sampling_in_loop=0
`
for missing rate 0 and 0.5
python cora_link.py --data=Cora --categorical=True --imputation='zero' --result_file='' --missing_rate=0 --gpu=3  --num_epochs=500 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.4 --otf_sample=1 --fto_sample=1 --num_obs_samples=15 --verbose=1 --num_feat_samples=15 --sampling_in_loop=0


for missing rate 0.99

python cora_link.py --data=Cora --imputation='zero' --result_file='' --missing_rate=0.99 --gpu=3  --num_epochs=500 --num_layers=1 --bs_train_nbd=-1 --bs_test_nbd=-1 --drop_rate=0.3 --otf_sample=0 --fto_sample=0 --num_obs_samples=30 --verbose=1 --num_feat_samples=15 --sampling_in_loop=0