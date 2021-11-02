import torch
import torch.nn as nn
import torch.nn.functional as F

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
    
    
class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, spa_kernel_size):
        super().__init__()
        self.spa_kernel_size = spa_kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels*spa_kernel_size, (1,1))
        
    def forward(self, x, A):
        assert A.size(0) == self.spa_kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.spa_kernel_size, kc // self.spa_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()


class unit_tcn(nn.Module):
    def __init__(self, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                            stride=(stride, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(x))
        x = self.bn2(self.conv(x))
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual=True):
        super().__init__()
        tem_kernel_size = kernel_size[0]
        spa_kernel_size = kernel_size[1]
        self.gcn = unit_gcn(in_channels, out_channels, spa_kernel_size)
        self.tcn = unit_tcn(out_channels, tem_kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                            nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=1,
                                    stride=(stride, 1)),
                            nn.BatchNorm2d(out_channels))
            
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
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
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
        
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        
        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        
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
        for st_gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = st_gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x