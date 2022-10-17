import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl

# 先以最简洁的方式写，以便对齐。
class SAGEConv(nn.Layer):
    def __init__(self, input_size, hidden_size, aggregator="mean", bias=True):
        super(SAGEConv, self).__init__()
        self.aggregator = aggregator
        if aggregator is not "mean":
            raise AssertionError("aggregator %s not supported." % (aggregator,))
        self.neigh_linear = nn.Linear(input_size, hidden_size)
        self.self_linear = nn.Linear(input_size, hidden_size)
        if bias:
            self.bias = \
                paddle.create_parameter(shape=[hidden_size], 
                                        dtype="float32",
                                        default_initializer=paddle.nn.initializer.Constant(value=0.0))
        else:
            self.bias = None       

    def forward(self, graph, x, act=None):
        if isinstance(x, paddle.Tensor):
            x = (x, x)
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        output = paddle.geometric.send_u_recv(x[0], src, dst, 
                                              reduce_op=self.aggregator,
                                              out_size=x[1].shape[0])
        output = self.neigh_linear(output)
        output += self.self_linear(x[1])
        if self.bias is not None:
            output = output + self.bias
        if act is not None:
           output = getattr(F, act)(output)
        return output 
       

class GraphSage(nn.Layer):
    def __init__(self, input_size, num_classes, num_layers=1,
                 hidden_size=64, dropout=0.5, **kwargs):
        super(GraphSage, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.LayerList()

        for i in range(num_layers):
            layer_output_dim = hidden_size if i != num_layers - 1 else num_classes
            layer_input_dim = input_size if i == 0 else hidden_size
            self.convs.append(SAGEConv(layer_input_dim, layer_output_dim, "mean"))

    def forward(self, graph_list, x):
        for i, (graph, size) in enumerate(graph_list):
            x_target = x[:size]
            x = self.convs[i](graph, (x, x_target))
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x
