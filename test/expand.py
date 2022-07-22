import torch



mask = torch.tensor([[0,1],[1,0]], dtype=torch.bool)

a = torch.randn(3,2)

a = a.unsqueeze(-2).expand(3,2,2)

a = a.masked_fill(mask, value=-1)

print(a)

