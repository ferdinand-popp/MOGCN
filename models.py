import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GraphConv
from layer import GraphConvolution

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        self.dp = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dp(x)
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        self.dp = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dp(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(GATEncoder, self).__init__()
        self.in_head = 4
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, 2 * out_channels, heads=self.in_head, dropout=self.dropout, cached=True)
        self.conv2 = GATv2Conv(2 * out_channels * self.in_head, out_channels, heads=1, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class LinGATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(LinGATEncoder, self).__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, out_channels, heads=1, dropout=self.dropout, cached=True)

    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = SAGEConv(2 * out_channels, out_channels, cached=True)
        self.dp = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dp(x)
        return self.conv2(x, edge_index)

class GraphConvEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(GraphConvEncoder, self).__init__()
        self.conv1 = GraphConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GraphConv(2 * out_channels, out_channels, cached=True)
        self.dp = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dp(x)
        return self.conv2(x, edge_index)

class VariationalGraphConvEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(VariationalGraphConvEncoder, self).__init__()
        self.conv1 = GraphConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GraphConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GraphConv(2 * out_channels, out_channels, cached=True)
        self.dp = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dp(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, 64)
        self.lin2 = torch.nn.Linear(64, out_channels)

    def forward(self, z):
        z = torch.relu(self.lin1(z))
        return self.lin2(z)

class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hid, n_out)
        self.dropout = dropout

    def forward(self, input, adj):
        x = self.gc1(input, adj)
        x = F.elu(x)
        x = self.dp1(x)
        x = self.gc2(x, adj)
        x = F.elu(x)
        x = self.dp2(x)
        return self.fc(x)
