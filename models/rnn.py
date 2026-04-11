import torch.nn as nn

# Improved from the previous MLP, now processing data step by step over time,
#  but still simple without memory cell and gates,
#  so it may struggle with long-term dependencies
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)

        # take last timestep
        out = out[:, -1, :]
        return self.fc(out)