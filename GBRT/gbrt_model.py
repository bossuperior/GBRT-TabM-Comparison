import torch.nn as nn


class FlexibleMLP(nn.Module):
    def __init__(self, n_layers, n_neurons, input_dim=8):  # California Housing มี 8 features
        super().__init__()
        layers = []
        in_dim = input_dim

        # สร้าง Hidden Layers ตามจำนวนที่ GBRT แนะนำ
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, n_neurons))
            layers.append(nn.ReLU())
            in_dim = n_neurons

        layers.append(nn.Linear(in_dim, 1))  # Output 1 ตัวสำหรับพยากรณ์ราคาบ้าน
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)