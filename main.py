import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')

import os
import os.path as osp
import argparse

from datetime import datetime
import time
import logging
from logger import Logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import numpy as np

from data import LoadData
from models import GCNNet
from transformation import GraphRandomNoise, GraphDeletionNoise, GraphInsertionNoise, NoiseMerge, \
    NodeFeatKernel, NodeFeatKernelSparsification, NodeFeatKernelDirect, NodeFeatNoise


def train(model, data, optimizer, train_mask):
    model.train()
    loss = F.nll_loss(model(data)[train_mask], data.y[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, masks):
    model.eval()
    logits, accs = model(data), []
    for mask in masks:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def main():
    parser = argparse.ArgumentParser(description='Robust-GCN-Node-Classification')

    ### setting
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_cpu', default=False, action='store_true')
    parser.add_argument('--out_dir', type=str, default="./results/")
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='Cora')

    ### model setting
    parser.add_argument('--hiddim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--readout', type=str, default='mlp', choices=['conv', 'mlp'])
    parser.add_argument('--exp_type', type=str, default='random-feature',
                        choices=['mlp', 'gat', 'random-feature', 'robust'])
    parser.add_argument('--noise_type', type=str, default='none',
                        choices=['random', 'insertion', 'deletion', 'none'])
    parser.add_argument('--merged', default=False, action='store_true')
    parser.add_argument('--no-merged', dest='merged', action='store_false')

    parser.add_argument('--add_feat_noise', default=False, action='store_true')
    parser.add_argument('--no-feat_noise', dest='add_feat_noise', action='store_false')

    parser.add_argument('--add_kernel', default=False, action='store_true')
    parser.add_argument('--no-kernel', dest='add_kernel', action='store_false')

    parser.add_argument('--random_noise_type', type=str, default='erdos-renyi',
                        choices=['erdos-renyi', 'sbm', 'barabasi-albert'])

    parser.add_argument('--kernel_type', type=str, default='linear', choices=['linear', 'rbf', 'dist'])
    parser.add_argument('--spectral_norm', type=str, default='none', choices=['none', 'preprocess', 'postprocess'])

    parser.add_argument('--noise_ratio', type=float, default=1.0)
    parser.add_argument('--feat_noise_ratio', type=float, default=1.0)

    parser.add_argument('--standarize', default=False, action='store_true')
    parser.add_argument('--no-standarize', dest='standarize', action='store_false')

    parser.add_argument('--centerize', default=False, action='store_true')
    parser.add_argument('--no-centerize', dest='centerize', action='store_false')

    parser.add_argument('--add_identity', default=False, action='store_true')
    parser.add_argument('--no-identity', dest='add_identity', action='store_false')

    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')

    parser.add_argument('--nystrom', default=False, action='store_true')
    parser.add_argument('--no-nystrom', dest='nystrom', action='store_false')

    parser.add_argument('--rf_norm', default=False, action='store_true')
    parser.add_argument('--no-rf_norm', dest='rf_norm', action='store_false')

    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--split_type', type=str, default='public', choices=['random', 'public'])
    parser.add_argument('--splits', type=int, default=1)

    ### training setting
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()

    args.experiment = "{}-{}-{}-{}".format(args.exp_type, args.noise_type, args.merged, args.add_kernel)

    def name_folder_path():
        path = osp.join(args.out_dir, args.experiment, "{}-{}-{}".format(args.dataset, args.num_layers, args.readout))
        file = ("{}_"*15).format(args.hiddim, args.noise_ratio, args.epsilon,
                                 args.add_feat_noise, args.feat_noise_ratio, args.nystrom,
                                 args.random_noise_type, args.kernel_type,
                                 args.add_identity, args.normalize, args.spectral_norm, args.rf_norm,
                                 args.standarize, args.centerize, args.split_type)
        return path, file

    path, file = name_folder_path()
    path = osp.join(osp.dirname(osp.realpath(__file__)), path)

    log_dir = osp.join(path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    task_name = file + datetime.now().strftime("%m%d_%H%M%S")
    logname = osp.join(log_dir, "%s.log" % (task_name))

    log = logging.getLogger(task_name)
    log.setLevel(logging.INFO)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.FileHandler(filename=logname)  # output to file
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(fmt, datefmt))
    log.addHandler(handler)

    chler = logging.StreamHandler()  # print to console
    chler.setFormatter(logging.Formatter(fmt, datefmt))
    chler.setLevel(logging.INFO)
    log.addHandler(chler)

    log.info("Experiment of model: %s" % (task_name))
    log.info(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() and not args.use_cpu else 'cpu'
    device = torch.device(device)

    dataset_name = args.dataset

    pretrans = []
    trans = []

    if args.add_feat_noise:
        pretrans.append(NodeFeatNoise(noise_ratio=args.feat_noise_ratio, keep_origin=True))

    if args.noise_type == "random":
        pretrans.append(GraphRandomNoise(noise_type=args.random_noise_type, noise_ratio=args.noise_ratio))

    elif args.noise_type == "insertion":
        pretrans.append(GraphInsertionNoise(noise_ratio=args.noise_ratio, method="sparse"))

    elif args.noise_type == "deletion":
        pretrans.append(GraphDeletionNoise(noise_ratio=args.noise_ratio))

    if args.merged == True:
        pretrans.append(NoiseMerge())

    if args.add_kernel == True:
        pretrans.append(NodeFeatKernel(kernel=args.kernel_type,
                                       add_identity=args.add_identity,
                                       approximate=args.nystrom,
                                       standarize=args.standarize,
                                       centerize=args.centerize))

        if args.sparsifier != 'none':
            trans.append(NodeFeatKernelSparsification(sparsifier="origin", ratio=1.0,
                                                      add_identity=args.add_identity, normalize=args.spectral_norm))

    if len(pretrans):
        pretrans = T.Compose(pretrans)
    else:
        pretrans = None

    if len(trans):
        trans = T.Compose(trans)
    else:
        trans = None

    root = osp.join(path, "data")
    data_loader = LoadData(root, dataset_name, pre_transform=pretrans, transform=trans)
    data_loader.load()
    dataset = data_loader.dataset
    data = dataset[0].to(device)

    model = GCNNet(dataset.num_features, args.hiddim, dataset.num_classes, args.num_layers, args.readout,
                   args.exp_type, args.merged, args.epsilon, False,
                   args.add_identity, args.normalize, args.rf_norm)

    logger = Logger(args.runs, args.splits, args, log)

    for split in range(args.splits):
        masks = data_loader.split(split_type=args.split_type, num_train_per_class=20, num_val=500, num_test=1000)

        for run in range(args.runs):
            model.reset_parameters()
            model = model.to(device)

            optimizer = torch.optim.Adam([dict(params=model.conv_layers.parameters(), weight_decay=5e-4),
                                          dict(params=model.readout.parameters(), weight_decay=5e-4)], lr=args.lr)

            for epoch in range(1, 1 + args.epochs):
                loss = train(model, data, optimizer, masks[0])
                result = test(model, data, masks)
                logger.add_result(run, result, split)

                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                    log.info(f'Split: {split + 1:02d}, '
                             f'Run: {run + 1:02d}, '
                             f'Epoch: {epoch:02d}, '
                             f'Loss: {loss:.4f}, '
                             f'Train: {100 * train_acc:.2f}%, '
                             f'Valid: {100 * valid_acc:.2f}% '
                             f'Test: {100 * test_acc:.2f}%')
            logger.print_statistics(run, split)
        logger.print_statistics(None, split)
    logger.print_statistics()

if __name__ == "__main__":
    main()
