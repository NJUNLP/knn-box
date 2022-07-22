from torch import nn



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 22)
        self.rnn = nn.RNNCell(44, 55)
        for name, param in self.named_parameters():
            param.requires_grad = False
        
        for name, param in self.named_parameters():
            if "rnn" in name:
                param.requires_grad = True
    

    def forward(self, x):
        return x


if __name__ == "__main__":
    model = Model()

    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)




