import torch
from torch import nn 
import torch.nn.functional as F

class Phi(nn.Module):
    def __init__(self,input_size, hidden_size):
        super().__init__()
        # output class for the correct return during forward pass
        self.out = outputClass(None)
        self.lin1 = nn.Linear(input_size, 1024)
        self.lin12 = nn.Linear(input_size, 256)

        self.lin2 = nn.Linear(1024, 1024)
        self.lin3 = nn.Linear(1024, 1024)
        self.lin4 = nn.Linear(1024, hidden_size)

        self.lin22 = nn.Linear(256, 256)
        self.lin32 = nn.Linear(256, hidden_size)

    def forward(self, x):
        seqlen = x.shape[1]
        y = x
        # Repeat Interleave - repeat would allow x - x', 
        # which we can take expectation over dim 1
        x = torch.repeat_interleave(x,seqlen,1) - x.repeat(1,seqlen,1)
        
        #x = torch.cat((x, t), -1)
        #y = torch.cat((y, t[:,0]), -1)
        y = self.lin12(y)
        y = F.leaky_relu(y)
        y = self.lin22(y)
        y = F.leaky_relu(y)
        y = self.lin32(y)

        x = self.lin1(x)
        x = F.leaky_relu(x)

        x = self.lin2(x)
        x = F.leaky_relu(x)

        x = self.lin3(x)
        x = F.leaky_relu(x)

        x = self.lin4(x)

        output = x.reshape(x.shape[0], seqlen, seqlen, -1).mean(1) + y
        self.out.last_hidden_state = output
        
        return self.out
    
class outputClass:
    def __init__(self, last_hidden_state) -> None:
        self.last_hidden_state = last_hidden_state