import torch.nn as nn

class FlexibleMLP(nn.Module):
    # เพิ่ม dropout_rate มารับค่า
    def __init__(self, n_layers, n_neurons, dropout_rate=0.0, input_dim=8):
        super().__init__()
        layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, n_neurons))
            layers.append(nn.ReLU())
            # แทรก Dropout เข้าไปหลัง Activation Function
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = n_neurons

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)