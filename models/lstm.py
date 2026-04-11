import torch.nn as nn

# RNN model for time series forecasting, using a simple RNN layer followed by a fully connected layer.
#  Add memory cell + gates. Remember important past info for long time
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)

        # last timestep
        out = out[:, -1, :]
        return self.fc(out)