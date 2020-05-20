import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from random import shuffle
from sklearn.metrics import roc_auc_score
from gcn_layer_dl import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_nodes = 100
input_dim = 10

x, edge_index, edge_attr = generate_random_data(num_nodes, input_dim)

output_dim = 128
num_layers = 3
drop_rate = 0.5

graph_layers = GraphNetLayers(input_dim, output_dim, num_layers, drop_rate)
graph_layers_output = graph_layers(x, edge_index, edge_attr)

print(graph_layers_output.size())
