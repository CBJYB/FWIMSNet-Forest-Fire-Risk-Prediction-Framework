"""
author:CBJ
"""

import torch
import torch.nn as nn


class MultiScale1DCNN(nn.Module):
    """
    Multi-Scale 1D Convolutional Neural Network

    Contains three parallel paths:
- Short-term path: kernel_size=7 (corresponds to FFMC, rapid weather-driven response)
- Medium-term path: kernel_size=30 (corresponds to DMC, medium-term eco-hydrological trends)
- Long-term path: kernel_size=90 (corresponds to DC, climate-scale drought accumulation)
    """
    def __init__(self, input_dims, kernel_sizes=[7, 30, 90], out_channels=32):
        """
        **Args:**
    **input_dims: Input feature dimensions for each path [short-term, medium-term, long-term]**
    **kernel_sizes: List of convolution kernel sizes**
    **out_channels: Number of output channels for each path**
        """
        super(MultiScale1DCNN, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels

        # Short-term Path - Corresponds to FFMC and Climate-Driven Group
        self.short_term_conv = nn.Conv1d(
            in_channels=input_dims[0],
            out_channels=out_channels,
            kernel_size=kernel_sizes[0],
            padding='same'
        )

        # Medium-term Path - Corresponds to DMC and Hydrological-Driven Group
        self.mid_term_conv = nn.Conv1d(
            in_channels=input_dims[1],
            out_channels=out_channels,
            kernel_size=kernel_sizes[1],
            padding='same'
        )

        # Long-term Path - Corresponds to DC and Soil-Driven Group
        self.long_term_conv = nn.Conv1d(
            in_channels=input_dims[2],
            out_channels=out_channels,
            kernel_size=kernel_sizes[2],
            padding='same'
        )

        # Normalization
        self.bn_short = nn.BatchNorm1d(out_channels)
        self.bn_mid = nn.BatchNorm1d(out_channels)
        self.bn_long = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        for conv in [self.short_term_conv, self.mid_term_conv, self.long_term_conv]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

    def forward(self, x_short, x_mid, x_long):

        x_short = x_short.permute(0, 2, 1)
        x_mid = x_mid.permute(0, 2, 1)
        x_long = x_long.permute(0, 2, 1)

        # Multi-Scale Convolution
        short_out = self.relu(self.bn_short(self.short_term_conv(x_short)))
        mid_out = self.relu(self.bn_mid(self.mid_term_conv(x_mid)))
        long_out = self.relu(self.bn_long(self.long_term_conv(x_long)))

        short_out = short_out.permute(0, 2, 1)
        mid_out = mid_out.permute(0, 2, 1)
        long_out = long_out.permute(0, 2, 1)

        # Feature Concatenation
        fused_features = torch.cat([short_out, mid_out, long_out], dim=-1)
        fused_features = self.dropout(fused_features)

        return fused_features


class GRUTransformerBlock(nn.Module):
    """
    GRU-Transformer Collaborative Module
    Structure:
    1. GRU Layer: Captures temporal evolution patterns and medium-term memory effects
    2. Transformer Layer: Mines global contextual correlations
    """
    def __init__(self, input_dim=96, hidden_dim=128, nhead=8, num_layers=2, dropout=0.1):

        super(GRUTransformerBlock, self).__init__()

        # GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Normalization
        self.norm_gru = nn.LayerNorm(hidden_dim)
        self.norm_trans = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        """
        Args:
            x: Input Features [batch_size, seq_len, 96]
        Returns:
            Encoded Features [batch_size, seq_len, 128]
        """
        # GRU
        gru_out, _ = self.gru(x)
        gru_out = self.norm_gru(gru_out)
        gru_out = self.dropout(gru_out)

        # Transformer
        trans_out = self.transformer(gru_out)
        trans_out = self.norm_trans(trans_out)

        return trans_out


class FWIMSNet(nn.Module):
    """
    FWI-MSNet:

    1. Multi-Scale Input Feature Construction (Paper Table 1: Three Groups: Short-term/Medium-term/Long-term)
    2. Multi-Scale Temporal Feature Extraction Module (Parallel 1D-CNN)
    3. GRU-Transformer Collaborative Mechanism
    4. Fully Connected Output Layer

    """
    def __init__(self,
                 short_term_dim=10,   # Short-term Path Feature Dimension (Climate-Driven Group)
                 mid_term_dim=6,      # Medium-term Path Feature Dimension (Hydrological-Driven Group)
                 long_term_dim=5,     # Long-term Path Feature Dimension (Soil-Driven Group)
                 kernel_sizes=[7, 30, 90],
                 cnn_out_channels=32,
                 gru_hidden_dim=128,
                 transformer_nhead=8,
                 transformer_layers=2,
                 dropout=0.1):

        super(FWIMSNet, self).__init__()

        # 1: Multi-Scale Feature Extraction Module
        self.multiscale_cnn = MultiScale1DCNN(
            input_dims=[short_term_dim, mid_term_dim, long_term_dim],
            kernel_sizes=kernel_sizes,
            out_channels=cnn_out_channels
        )

        # 2: GRU-Transformer Collaborative Module
        self.gru_transformer = GRUTransformerBlock(
            input_dim=cnn_out_channels * 3,  # Concatenation of Three Path Outputs: 32*3=96
            hidden_dim=gru_hidden_dim,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dropout=dropout
        )

        # 3: Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.output_layer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x_short, x_mid, x_long):
        """
        Args:
            x_short: [batch_size, seq_len, short_term_dim]
            x_mid: [batch_size, seq_len, mid_term_dim]
            x_long: [batch_size, seq_len, long_term_dim]
        Returns:
            Predicted FWI Value [batch_size, 1]
        """
        # 1: Multi-Scale Feature Extraction Module
        fused_features = self.multiscale_cnn(x_short, x_mid, x_long)

        # 2: GRU-Transformer Collaborative Module
        encoded_features = self.gru_transformer(fused_features)

        # 3: Take the output of the last time step
        last_step_features = encoded_features[:, -1, :]

        # 4: output Layer
        output = self.output_layer(last_step_features)

        return output


# ========== Model Variants for Ablation Experiments ==========

class FWIMSNetNoMultiScale(nn.Module):
    """
    Variant without multi-scale convolution
Directly concatenates features and inputs them into GRU-Transformer
    """
    def __init__(self, total_dim=21, gru_hidden_dim=128, transformer_nhead=8,
                 transformer_layers=2, dropout=0.1):
        super(FWIMSNetNoMultiScale, self).__init__()

        self.fc = nn.Linear(total_dim, 96)

        self.gru_transformer = GRUTransformerBlock(
            input_dim=96,
            hidden_dim=gru_hidden_dim,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dropout=dropout
        )

        self.output_layer = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x_short, x_mid, x_long):
        x = torch.cat([x_short, x_mid, x_long], dim=-1)
        x = self.fc(x)
        x = self.gru_transformer(x)
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x


class FWIMSNetNoTransformer(nn.Module):
    """
    Variant without Transformer
    Uses only multi-scale CNN + GRU
    """
    def __init__(self, short_term_dim=10, mid_term_dim=6, long_term_dim=5,
                 kernel_sizes=[7, 30, 90], cnn_out_channels=32,
                 gru_hidden_dim=128, dropout=0.1):
        super(FWIMSNetNoTransformer, self).__init__()

        self.multiscale_cnn = MultiScale1DCNN(
            input_dims=[short_term_dim, mid_term_dim, long_term_dim],
            kernel_sizes=kernel_sizes,
            out_channels=cnn_out_channels
        )

        self.gru = nn.GRU(
            input_size=cnn_out_channels * 3,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x_short, x_mid, x_long):
        x = self.multiscale_cnn(x_short, x_mid, x_long)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x


class FWIMSNetNoGRU(nn.Module):
    """
    Variant without GRU
    Uses only multi-scale CNN + Transformer
    """
    def __init__(self, short_term_dim=10, mid_term_dim=6, long_term_dim=5,
                 kernel_sizes=[7, 30, 90], cnn_out_channels=32,
                 transformer_nhead=8, transformer_layers=2, dropout=0.1):
        super(FWIMSNetNoGRU, self).__init__()

        self.multiscale_cnn = MultiScale1DCNN(
            input_dims=[short_term_dim, mid_term_dim, long_term_dim],
            kernel_sizes=kernel_sizes,
            out_channels=cnn_out_channels
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_channels * 3,
            nhead=transformer_nhead,
            dim_feedforward=cnn_out_channels * 3 * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(cnn_out_channels * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x_short, x_mid, x_long):
        x = self.multiscale_cnn(x_short, x_mid, x_long)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x


# ========== Model Factory Function ==========

def create_model(model_type='full', **kwargs):

    if model_type == 'full':
        return FWIMSNet(
            short_term_dim=kwargs.get('short_term_dim', 10),
            mid_term_dim=kwargs.get('mid_term_dim', 6),
            long_term_dim=kwargs.get('long_term_dim', 5),
            kernel_sizes=kwargs.get('kernel_sizes', [7, 30, 90]),
            cnn_out_channels=kwargs.get('cnn_out_channels', 32),
            gru_hidden_dim=kwargs.get('gru_hidden_dim', 128),
            transformer_nhead=kwargs.get('transformer_nhead', 8),
            transformer_layers=kwargs.get('transformer_layers', 2),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif model_type == 'no_multiscale':
        return FWIMSNetNoMultiScale(
            total_dim=kwargs.get('total_dim', 21),
            gru_hidden_dim=kwargs.get('gru_hidden_dim', 128),
            transformer_nhead=kwargs.get('transformer_nhead', 8),
            transformer_layers=kwargs.get('transformer_layers', 2),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif model_type == 'no_transformer':
        return FWIMSNetNoTransformer(
            short_term_dim=kwargs.get('short_term_dim', 10),
            mid_term_dim=kwargs.get('mid_term_dim', 6),
            long_term_dim=kwargs.get('long_term_dim', 5),
            kernel_sizes=kwargs.get('kernel_sizes', [7, 30, 90]),
            cnn_out_channels=kwargs.get('cnn_out_channels', 32),
            gru_hidden_dim=kwargs.get('gru_hidden_dim', 128),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif model_type == 'no_gru':
        return FWIMSNetNoGRU(
            short_term_dim=kwargs.get('short_term_dim', 10),
            mid_term_dim=kwargs.get('mid_term_dim', 6),
            long_term_dim=kwargs.get('long_term_dim', 5),
            kernel_sizes=kwargs.get('kernel_sizes', [7, 30, 90]),
            cnn_out_channels=kwargs.get('cnn_out_channels', 32),
            transformer_nhead=kwargs.get('transformer_nhead', 8),
            transformer_layers=kwargs.get('transformer_layers', 2),
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Calculate the Number of Model Parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)