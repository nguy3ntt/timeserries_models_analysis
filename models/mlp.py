import torch.nn as nn

# MLP as baseline model for time series forecasting,
#  basically treating the time series as a tabular data,
#  and flattening the input before feeding it to the model
class MLP(nn.Module):
    def __init__(self, input_size, seq_len, hidden_dim=128):
        super().__init__()

        self.flatten = nn.Flatten()

        self.model = nn.Sequential(
            nn.Linear(input_size * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.flatten(x)
        return self.model(x)