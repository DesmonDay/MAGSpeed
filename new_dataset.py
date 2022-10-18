import os
import argparse

import numpy as np
import yaml
import paddle
from paddle.io import Dataset
from pgl.graph import Graph
from pgl.utils.logger import log
from ogb.nodeproppred import NodePropPredDataset
from paddle.fluid import core

from dist_feat import DistFeat


class OgbnMag240(object):
    def __init__(self, args):
        self.num_features = 128
        self.feat_mode = args.feat_mode
        self.graph_mode = args.graph_mode

    def prepare_data(self):
        log.info("Preparing dataset...")
        
        # Get feature and graph.
        paper_edge_path = "/paddle/wholegraph/examples/gnn/pgl/paper_cites_paper_graph"
        self.graph = Graph.load(paper_edge_path, mmap_mode="r+")
        self.train_idx = np.load(paper_edge_path + "/train_idx.npy")
        self.val_idx = np.load(paper_edge_path + "/val_idx.npy")
        self.test_idx = np.load(paper_edge_path + "/test_idx.npy")

        paper_feat_path = "/paddle/wholegraph/examples/gnn/pgl/paper_cites_paper_graph/paper_feat.npy"
        self.x = np.load(paper_feat_path, mmap_mode="r")
        self.x = DistFeat(self.x, mode=self.feat_mode)
        self.y = np.load(paper_edge_path + "/label.npy")
        self.y = paddle.to_tensor(self.y, dtype="int64")

        self.prepare_csc_graph()

    def prepare_csc_graph(self):
        log.info("Preparing csc graph...")
        if self.graph_mode == "uva":
            row = core.eager.to_uva_tensor(self.graph.adj_dst_index._sorted_v, 0)
            colptr = core.eager.to_uva_tensor(self.graph.adj_dst_index._indptr, 0)
        elif self.graph_mode == "gpu":
            row = paddle.to_tensor(self.graph.adj_dst_index._sorted_v)
            colptr = paddle.to_tensor(self.graph.adj_dst_index._indptr)
        self.csc_graph = [row, colptr]
        log.info("Finish dataset")


class NodeIterDataset(Dataset):
    def __init__(self, data_index):
        self.data_index = data_index

    def __getitem__(self, idx):
        return self.data_index[idx]

    def __len__(self):
        return len(self.data_index)


class NeighborSampler(object):
    def __init__(self, csc_graph, samples_list, num_nodes, num_edges):
        self.csc_graph = csc_graph
        self.samples_list = samples_list
        self.value_buffer = paddle.full([int(num_nodes)], -1, dtype="int32")
        self.index_buffer = paddle.full([int(num_nodes)], -1, dtype="int32")
        self.eid_perm = np.arange(0, num_edges, dtype="int64")
        self.eid_perm = core.eager.to_uva_tensor(self.eid_perm, 0)

    def sample(self, nodes):
        graph_list = []
        for i in range(len(self.samples_list)):
            neighbors, neighbor_counts = paddle.geometric.sample_neighbors(
                self.csc_graph[0], self.csc_graph[1], nodes, sample_size=self.samples_list[i],
                perm_buffer=self.eid_perm)
            edge_src, edge_dst, out_nodes = \
                paddle.geometric.reindex_graph(nodes, neighbors, neighbor_counts,
                self.value_buffer, self.index_buffer)
            graph = Graph(num_nodes=len(out_nodes),
                          edges=paddle.concat([edge_src.reshape([-1, 1]),
                                               edge_dst.reshape([-1, 1]),
                                               ], axis=-1))
            graph_list.append((graph, len(nodes)))
            nodes = out_nodes

        graph_list = graph_list[::-1]
        return graph_list, nodes
