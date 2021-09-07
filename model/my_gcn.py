from typing import ForwardRef
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d 


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
    
    
class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, spa_kernel_size):
        super().__init__()
        spa_kernel_size = spa_kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels*spa_kernel_size, 1)
    
    def forward(self, x, A):
        mat_kernel_size = A.shape[0]
        n, c, t, v = x.size()
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, mat_kernel_size, kc // mat_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        # x = torch.matmul(x, A)
        # x = x.view(n, c*mat_kernel_size, t, v)
        # x = x.view(n, c, t, v).contiguous()
        return x


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                            stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
    
    
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual=True):
        super().__init__()
        tem_kernel_size = kernel_size[0]
        spa_kernel_size = kernel_size[1]
        self.gcn = unit_gcn(in_channels, out_channels, spa_kernel_size)
        self.tcn = unit_tcn(out_channels, out_channels, tem_kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        
    # def froward(self, x, A):  注意，这个拼写错误一直没发现
    def forward(self, x, A):
        x = self.tcn(self.gcn(x, A)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A,
                        dtype=torch.float32,
                        requires_grad=False)
        self.register_buffer('A', A)
        
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[1])
        # kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
        self.st_gcn_networks = nn.ModuleList((
            TCN_GCN_unit(in_channels, 64, kernel_size, 1, residual=False, **kwargs),
            TCN_GCN_unit(64, 64, kernel_size, 1, **kwargs),
            TCN_GCN_unit(64, 64, kernel_size, 1, **kwargs),
            TCN_GCN_unit(64, 64, kernel_size, 1, **kwargs),
            TCN_GCN_unit(64, 128, kernel_size, 2, **kwargs),
            TCN_GCN_unit(128, 128, kernel_size, 1, **kwargs),
            TCN_GCN_unit(128, 128, kernel_size, 1, **kwargs),
            TCN_GCN_unit(128, 256, kernel_size, 2, **kwargs),
            TCN_GCN_unit(256, 256, kernel_size, 1, **kwargs),
            TCN_GCN_unit(256, 256, kernel_size, 1, **kwargs),
        ))

        # embedding and concat
        self.embed1 = nn.Conv2d(64, 256, (1, 1))
        self.embed2 = nn.Conv2d(128, 256, (1, 1))
        self.embed3 = nn.Conv2d(256, 256, (1, 1))
        
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        
        self.w1 = nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(1e-6, dtype=torch.float), requires_grad=True)
        self.w3 = nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)
        
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.shape))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        
        # fcn for prediction
        self.fcn = nn.Conv2d(256*3, num_class, kernel_size=1)
        
    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size() # N:batch大小 C:三轴坐标 T:帧数 V：关节点数 M：最大人体数
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for i, (stgcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance), 1):
            x = stgcn(x, self.A * importance)
            if i == 4:
                feature_1 = x
            if i == 7:
                feature_2 = x
            if i == 10:
                feature_3 = x
        
        # global pooling
        
        feature_embed1 = self.relu(self.bn(self.embed1(feature_1)))
        feature_embed1 = F.avg_pool2d(feature_embed1, feature_embed1.size()[2:])
        feature_embed1 = feature_embed1.view(N, M, -1, 1, 1).mean(dim=1)
        
        feature_embed2 = self.relu(self.bn(self.embed2(feature_2)))
        feature_embed2 = F.avg_pool2d(feature_embed2, feature_embed2.size()[2:])
        feature_embed2 = feature_embed2.view(N, M, -1, 1, 1).mean(dim=1)
        
        feature_embed3 = self.relu(self.bn(self.embed3(feature_3)))
        feature_embed3 = F.avg_pool2d(feature_embed3, feature_embed3.size()[2:])
        feature_embed3 = feature_embed3.view(N, M, -1, 1, 1).mean(dim=1)
        
        # feature_embed = self.w1 * feature_embed1 + self.w2 * feature_embed2 + self.w3 * feature_embed3
        feature_embed =  torch.cat((feature_embed1, feature_embed2, feature_embed3), dim=1)
        # feature_embed = self.w1 * feature_embed1
        
        # prediction
        
        x = self.fcn(feature_embed)
        x = x.view(x.size(0), -1)
        
        return x