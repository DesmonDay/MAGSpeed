import time
import argparse

import numpy as np
import paddle
from paddle.io import DataLoader, DistributedBatchSampler
from pgl.utils.logger import log

from model import GraphSage 
from new_dataset import OgbnMag240, NodeIterDataset, NeighborSampler


def train(args):
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    ogbnmag_dataset = OgbnMag240(args)
    ogbnmag_dataset.prepare_data()
    train_ds = NodeIterDataset(ogbnmag_dataset.train_idx)
    eval_ds = NodeIterDataset(ogbnmag_dataset.val_idx)
    test_ds = NodeIterDataset(ogbnmag_dataset.test_idx)

    train_sampler = DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, drop_last=False)

    model = GraphSage(args.feat_dim, args.classnum, args.layer_num,
                      args.hidden_size)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    criterion = paddle.nn.loss.CrossEntropyLoss()
    optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters()) 
    samples_list = [30, 30, 30]
    ns = NeighborSampler(ogbnmag_dataset.csc_graph, samples_list=samples_list)
    log.info("Start") 
    for e_id in range(1, args.epochs + 1):
        t = time.perf_counter()
        for batch_nodes in train_loader:
            graph_list, neigh_nodes = ns.sample(batch_nodes)
            x = ogbnmag_dataset.x[neigh_nodes]
            y = ogbnmag_dataset.y[batch_nodes]
            out = model(graph_list, x)
            loss = criterion(out, y)
            loss.backward()
            optim.step()
            optim.clear_grad()
        log.info(f"Epoch cost time [{time.perf_counter() - t:.2f}s]")
    log.info("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--feat_mode", type=str, default="gpu", help="cpu, uva, gpu")
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--graph_mode", type=str, default="gpu", help="uva, gpu")
    parser.add_argument("--classnum", type=int, default=172, help="class number")
    parser.add_argument("--layer_num", type=int, default=3) 
    parser.add_argument("--hidden_size", type=int, default=256)

    args = parser.parse_args()
    train(args)
