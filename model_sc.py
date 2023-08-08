import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import SAGEConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=10, out_channels=2, embedding_size=128):
        super(GNN, self).__init__()
        self.SCNet = SCNet(in_channels, hidden_channels, num_layers, out_channels, embedding_size)
    
    def forward(self, data):
        x, edge_index, batch, u_index = data.x, data.edge_index, data.batch, data.u_index
        
        out = self.SCNet(x, edge_index, batch, u_index)
        
        return out


class SCNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=10, out_channels=2, embedding_size=128):
        super(SCNet, self).__init__()
     
        self.embedding = nn.Embedding(in_channels, embedding_size)

        self.conv_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        
        self.conv_layers.append(SAGEConv(embedding_size, hidden_channels))
        self.bn_layers.append(BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels))
            self.bn_layers.append(BatchNorm1d(hidden_channels))

        self.MLP3 = nn.Sequential(
                    Linear(hidden_channels, hidden_channels),
                    BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    Linear(hidden_channels, out_channels)
                )

    def forward(self, x, edge_index, batch, u_index):
        # x shape [N, in_channels]
        # edge_index shape [2, E]
        print(x.size())
        h = self.embedding(x)
        print(x.size())
        abc
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        
        out = []
        for u in u_index:
            out.append(h[u])

        out = torch.stack(out, dim=0)

        # ReadOut layer
        x = global_mean_pool(h, batch)  

        # Graph Classifier
        out = self.MLP3(out)    # [B, out_channels]

        return x
    