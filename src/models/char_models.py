import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
    
class HighwayNetwork(nn.Module):
    """Highway network"""
    def __init__(self, input_size, name, monitor=None):
        super(HighwayNetwork, self).__init__()
        self.name = name
        self.monitor = monitor
        self.fc_gate = weight_norm(nn.Linear(input_size, input_size, bias=True), dim=None)
        self.fc = weight_norm(nn.Linear(input_size, input_size, bias=True), dim=None)
        
    def forward(self, x, iter=None):
        activation = self.fc_gate(x)
        t = torch.sigmoid(activation)
        if self.monitor and iter:
            name = "character_highway_gate in {}".format(self.name)
            self.monitor.add_histogram(name, t.clone().cpu().data.numpy(), iter)
            name = "character_highway_activatation in {}".format(self.name)
            self.monitor.add_histogram(name, activation.clone().cpu().data.numpy(), iter)
        return torch.mul(t, F.relu(self.fc(x))) + torch.mul(1-t, x)

class CharacterLevelCNN(nn.Module):
    def __init__(self, device, input_dim, filter_num_width, name, monitor=None):
        super(CharacterLevelCNN, self).__init__()
        self.name = name
        self.monitor = monitor
        self.convolutions = nn.ModuleList().to(device=device)
        for filter_width, n_kernels in filter_num_width.items():
            self.convolutions.append(weight_norm(nn.Conv1d(input_dim, n_kernels, kernel_size=filter_width, bias=True).to(device=device), dim=None))
        
        self.highway_input_dim = sum([y for x, y in filter_num_width.items()])

#         self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False).to(device=device)
# 
        self.highway1 = HighwayNetwork(self.highway_input_dim, name, self.monitor).to(device=device)
#         self.highway2 = HighwayNetwork(self.highway_input_dim).to(device=device)

    def forward(self, input, iter=None):
        input = input.squeeze(0)
        input = input.transpose(1, 2)
        x = self.conv_layers(input, iter)
        if self.monitor and iter:
            name = "character_convolutions in {}".format(self.name)
            self.monitor.add_histogram(name, x.clone().cpu().data.numpy(), iter)
#         x = self.batch_norm(x)
#         if self.monitor:
#             name = "character_batchnorm in {}".format(self.name)
#             self.monitor.add_histogram(name, x.clone().cpu().data.numpy(), iter)
        x = self.highway1(x, iter)
#         x = self.highway2(x)
        return x

    def conv_layers(self, x, iter):
        chosen_list = list()
        for i, conv in enumerate(self.convolutions):
            if self.monitor and iter:
                image = torch.cat([kernel for kernel in conv.weight], dim=1).unsqueeze(0)
                name = "convolution kernel in {} width ".format(str(i+1))
                self.monitor.add_image(name, image.clone().cpu().data.numpy(), iter)  
            conv_map = torch.tanh(conv(x))
            chosen = torch.max(conv_map, 2)[0]
            chosen = chosen.squeeze()
            chosen_list.append(chosen)
        
        return torch.cat(chosen_list, 1)
