

import torch


a = torch.randn(224,1,32,32)
b = torch.ones(32,32).bool()

c = a.masked_fill(b, 0)

print(c.shape)
