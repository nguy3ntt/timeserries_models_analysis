import torch.nn as nn

# Transformer model for time series forecasting,
#  using self-attention mechanism to capture long-term dependencies + No vanishing gradient issue
class TransformerModel(nn.Module):
    def __init__(self, input_size, seq_len, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        x = self.input_projection(x)
        x = self.transformer(x)

        # last timestep
        x = x[:, -1, :]
        return self.fc(x)