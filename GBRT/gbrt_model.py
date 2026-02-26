import torch.nn as nn

class GBRTModel(nn.Module):
    def __init__(self, input_dim, n_layers=2, n_neurons=64, dropout_rate=0.1):
        super(GBRTModel, self). __init__()
        layers = []
        in_dim = input_dim

        # สร้างเลเยอร์ตามที่ GBRT Tuner สุ่มมาให้
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, n_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = n_neurons

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)