# Node Feature Kernels Increase Graph Convolutional Network Robustness
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the official implementation of [*Node Feature Kernels Increase Graph Convolutional Network Robustness*](https://arxiv.org/abs/2109.01785).

<div align=center>
<img src=https://github.com/ChangminWu/RobustGCN/blob/main/img/align.jpg  width="50%">
</div>

It is mainly developed with help of the library [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric). We also thank [Open Graph Benchmark implementation](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/logger.py) for providing an example of `logger.py`.

## Requirements
A virtual environment can be created by `conda` with the given environments file,
```
conda env create -f environments.yml
```

Notice that [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) needs to be installed separately via `Pip`, as
```
conda activate RobustGCN
pip install -r requirements.txt
```

## Usage
### Run
This implementation is able to reproduce experiment results shown in our paper which studies the robustness of Graph Convolutional Networks (GCNs) under structural perturbation, including
+ Asymptotic behaviour of random feature GCN with growing hidden dimension.
+ GCN with separate noise (modelled by a random graph) added in the message-passing step.
+ GCN with noise merged with its original adjacency matrix, e.g. edge deletion/insertion.
+ Experiments on multi-layer GCN.
+ GCN with perturbed adjacency as well as node feature perturbation.
+ Experiments with randomised train/test/valid splits.

To run experiments with `Vanilla GCN`, e.g. on Cora, do
```
python main.py --dataset Cora --out_dir <out dir> --num_layer 1 --readout mlp --exp_type robust --noise_type none --no-merged --epsilon 1.0
```

For `random feature GCN`, do
```
python main.py --dataset Cora --out_dir <out dir> --num_layer 1 --readout mlp --exp_type random-feature --noise_type none --epsilon 1.0 --hiddim 3000
```

For experiments in `theoretical case (separate noise)`, do
```
python main.py --dataset Cora --out_dir <out dir> --num_layer 1 --readout mlp --exp_type robust --noise_type random --no-merged --noise_ratio 1.0 --epsilon 0.5 --identity
```

For experiments in `realistic case (merged noise)`, do
```
python main.py --dataset Cora --out_dir <out dir> --num_layer 1 --readout mlp --exp_type robust --noise_type deletion --merged --noise_ratio 0.5 --epsilon 0.5 --add_kernel --add_identity --normalize --nystrom 
```
in the case of `Edge Deletion`. And do 
```
python main.py --dataset Cora --out_dir <out dir> --num_layer 1 --readout mlp --exp_type robust --noise_type insertion --merged --noise_ratio 0.5 --epsilon 0.5 --add_kernel --add_identity --normalize --nystrom
```
in the case of `Edge Insertion`. 

For `deeper architecture`, simply change the parameter of `num_layer`. 

For experiments in `node feature noise`, e.g. realistic case, do
```
python main.py --dataset Cora --out_dir <out dir> --num_layers 1 --readout mlp --exp_type robust --noise_type insertion --merged --noise_ratio 0.5 --epsilon 0.5 --add_kernel --add_identity --normalize --nystrom --add_feat_noise --feat_noise_ratio 1.0
```

For experiments with `multiple splits` (see supplementary material Section E), e.g. realistic case, do
```
python main.py --dataset Cora --out_dir <out dir> --num_layer 1 --readout mlp --exp_type robust --noise_type insertion --merged --noise_ratio 0.5 --epsilon 0.5 --add_kernel --add_identity --normalize --nystrom --splits 10 --split_type random
```

### Important Parameters
```
Description of important Model Options:
    --hiddim <int>
        dimension of the hidden representation of node embedding, default is 128
    --num_layer <int>
        number of stacked GCN layers, default is 1
    --readout <str>
        choice of readout functions that output prediction score, default is 'mlp'
    --exp_type <str>
        choice of different experiment settings, default is 'random-feature' 
    --noise_type <str>
        choice of different (noise) scenario, default is 'none' 
    --merged <bool>
        whether to merge noise into the original adjacency matrix, default is False
    --add_feat_noise <bool>
        whether to add gaussian noise on the features, default is False
    --add_kernel <bool>
        whether to enhance GCN message-passing with kernel, default is False
    --random_noise_type <str>
        choice of random graph generative model modelling the noise, default is Erdos-Renyi graph
    --kernel_type <str>
        choice of kernel function, default is 'linear'
    --noise_ratio <float>
        ratio between the random noise graph's density and the original graph's density, default is 1.0 
    --feat_noise_ratio <float>
        ratio between the standard deviation of the added gaussian noise and the original node features, default is 1.0 
    --standarize <bool>
        whether to standarize node features, default is False
    --centerize <bool>
        whether to centerize kernel values, default is False
    --add_identity <bool>
        whether to add self-loops to noise/kernel, default is False
    --normalize <bool>
        whether to degree normalize noise/kernel, default is False
    --rf_norm <bool>
        whether to normalize random weights in random feature GCN, default is False
    --split_type <str>
        choice of train/valid/test split of datasets, default is 'public'
    --nystrom <bool>
        whether to use nystrom approximation for computing kernel, default is False
    --epsilon <float>
        coefficient of the propagation following original graph in the GCN message-passing step, default is 1.0
    
```

## Results

>Table 1: Performance of GCN/GIN/GraphSage/GAT with node-feature kernel under perturbation on **Cora**

| deletion | insertion | GCN | GCN-k | GIN | GIN-k | SAGE | SAGE-k | GAT | GAT-k |
|------|------|------|------|------|------|------|------|------|------|
| 0.0 | 0.0 | 76.42 ± 1.55 | 75.42 ± 1.65 | 76.94 ± 1.41 | 77.62 ± 1.74 | 74.77 ± 1.98 | 76.00 ± 2.05 | 76.55 ± 2.23 | 77.45 ± 2.00 |
| 0.5 | 0.0 | 71.46 ± 1.66 | 69.00 ± 2.99 | 70.42 ± 2.03 | 70.23 ± 1.73 | 67.37 ± 1.73 | 70.46 ± 1.86 | 70.86 ± 1.45 | 71.35 ± 1.90 |
| 0.0 | 1.0 | 60.73 ± 2.20 | 70.55 ± 1.52 | 63.87 ± 2.85 | 67.80 ± 2.27 | 66.53 ± 1.80 | 68.52 ± 1.97 | 59.25 ± 1.99 | 64.92 ± 1.55 |
| 0.5 | 0.5 | 53.90 ± 1.88 | 63.79 ± 2.26 | 56.36 ± 2.23 | 62.79 ± 1.56 | 62.06 ± 1.73 | 63.80 ± 2.54 | 52.78 ± 2.37 | 58.01 ± 1.96 |
| 0.5 | 1.0 | 45.04 ± 2.46 | 62.08 ± 2.30 | 49.56 ± 3.40 | 55.24 ± 2.13 | 59.54 ± 1.75 | 62.15 ± 2.32 | 43.97 ± 2.29 | 52.47 ± 1.52 |

In the above table, each row corresponds to one perturbation scenario, where edges are randomly removed and/or added which is controlled by the two parameters: “deletion” and “insertion”, which correspond to the ratio of edges (w.r.t the original number of edges in the original graph) deleted/inserted from/to the graph. For example, the scenario (0.0, 0.0) corresponds to the unperturbed case and (0.5, 0.5) corresponds to the case where 50\% of the original edges are removed and a same number of edges non-existing in the original graph are added.    

Each column corresponds to a GNN model we considered. The appendage "-k" in the model name identifies that the model contains our proposed node-feature kernel. Each model is composed of a single message passing layer and a MLP readout layer. For all "-k" models, the coefficient of the perturbed graph propagation, i.e., $\beta$ in the paper, equals 0.5.

## Contribution
#### Authors: 
+ Mohamed El Amine Seddik
+ Changmin Wu
+ Johannes F. Lutzeyer
+ Michalis Vazirgiannis

If you find our repo useful, please cite:
```
@misc{seddik2021node,
      title={Node Feature Kernels Increase Graph Convolutional Network Robustness}, 
      author={Mohamed El Amine Seddik and Changmin Wu and Johannes F. Lutzeyer and Michalis Vazirgiannis},
      year={2021},
      eprint={2109.01785},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
