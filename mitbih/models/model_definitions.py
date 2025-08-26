"""
model_definitions.py

This module defines and initializes neural network architectures for ECG arrhythmia classification.
Models include:
- AcharyaCNN: Based on the architecture from Acharya et al. (2017)
- ECGCNN: A tunable 1D CNN used as the baseline model
- iTransformer: A simple Transformer-based model with a classification token
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_models(num_classes=5):
    """
    Returns a dictionary of model instances with given number of output classes.

    Args:
        num_classes (int): Number of classification categories.

    Returns:
        dict: Keys are model names, values are instantiated model objects.
    """
    return {
        'AcharyaCNN': AcharyaCNN(num_classes=num_classes),
        'ECGCNN':     ECGCNN(num_classes=num_classes),
        'iTransformer': iTransformer(num_classes=num_classes)
    }

class AcharyaCNN(nn.Module):
    """
    CNN architecture inspired by Acharya et al. (2017) for 1D ECG classification.

    Structure:
        - 3 convolutional + pooling layers
        - 3 fully connected layers
        - ReLU activations

    Args:
        num_classes (int): Number of output classes for classification.
    """
    def __init__(self, num_classes=5):
        super(AcharyaCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, stride=1)  
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)                              

        self.conv2 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=4, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)                             

        self.conv3 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=4, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)                              

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 260)
            out = self.pool1(F.relu(self.conv1(dummy)))
            out = self.pool2(F.relu(self.conv2(out)))
            out = self.pool3(F.relu(self.conv3(out)))
            self.flatten_dim = out.view(1, -1).shape[1]
            # print("Calculated flatten_dim =", self.flatten_dim)
        
        # Flatten for fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, num_classes)

    def forward(self, x):
        assert x.shape[2] == 260, f"Expected input length 260, got {x.shape[2]}"
        x = self.pool1(F.relu(self.conv1(x)))   # (B, 5, 129)
        x = self.pool2(F.relu(self.conv2(x)))   # (B, 10, 63)
        x = self.pool3(F.relu(self.conv3(x)))   # (B, 20, 30)

        # print("Shape before flatten:", x.shape) 
        x = x.view(-1, self.flatten_dim)        # dynamic flattening
        x = F.relu(self.fc1(x))                 # (B, 30)
        x = F.relu(self.fc2(x))                 # (B, 20)
        x = self.fc3(x)                         # (B, 5)
        return x

# the baseline mode of 1DCNN
class ECGCNN(nn.Module):
    """
    A 1D CNN model with tunable hyperparameters and optional third convolutional layer.

    Args:
        kernel_size (int): Size of convolutional filters.
        dropout (float): Dropout rate.
        filters1 (int): Number of filters in first conv layer.
        filters2 (int): Number of filters in second conv layer.
        fc1_size (int): Size of the first fully connected layer.
        num_classes (int): Number of output classes.
        filters3 (int or None): Number of filters in third conv layer if used.
        use_third_conv (bool): Whether to include a third conv layer.
    """
    def __init__(self, kernel_size=5, dropout=0.3, filters1=32, filters2=64, fc1_size=130, num_classes=5, filters3=None, use_third_conv=False):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, filters1, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(filters1)
        self.conv2 = nn.Conv1d(filters1, filters2, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(filters2)
        self.pool = nn.MaxPool1d(2)

        self.use_third_conv = use_third_conv
        if self.use_third_conv and filters3:
            self.conv3 = nn.Conv1d(filters2, filters3, kernel_size=kernel_size, padding=kernel_size//2)
            self.bn3 = nn.BatchNorm1d(filters3)
        
        self.dropout = nn.Dropout(dropout)

        # Determine output size after conv+pool layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 260)
            out = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            out = self.pool(F.relu(self.bn2(self.conv2(out))))
            if use_third_conv:
                out = self.pool(F.relu(self.bn3(self.conv3(out))))
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        if self.use_third_conv:
            x = self.pool(F.relu(self.bn3(self.conv3(x))))        
        x = x.view(x.size(0), -1)                      
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding for 1D transformer inputs.

    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum sequence length.
    """
    def __init__(self, d_model, max_len=260):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class iTransformer(nn.Module):
    """
    A simplified Transformer-based model for classifying ECG signals.

    Architecture:
        - Linear projection of 1D input into embedding space
        - Optional learnable positional embedding
        - CLS token prepended for sequence classification
        - Multi-layer TransformerEncoder
        - Linear classification head

    Args:
        input_len (int): Length of input signal (default: 260).
        num_classes (int): Number of output classes.
        emb_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward layer size.
        num_layers (int): Number of Transformer encoder layers.
        dropout (float): Dropout rate in Transformer layers.
    """
    def __init__(self, input_len=260, num_classes=5, emb_dim=128, num_heads=4, ff_dim=256, num_layers=4, dropout=0.1):
        super(iTransformer, self).__init__()

        # Input projection (1D to emb_dim)
        self.input_proj = nn.Linear(input_len, emb_dim)

        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, emb_dim))
        #self.pos_embed = nn.Parameter(torch.randn(1, 261, emb_dim))
        
        # Transformer Encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))  # Instance token
        self.fc = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, 260)
        #x = x.transpose(0,2,1)
        B = x.size(0)

        # Flatten time into feature vector per instance: (B, emb_dim)
        x = self.input_proj(x.squeeze(1))  # (B, emb_dim)

        # Add a class token: (B, 1, emb_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_dim]
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # [B, 2, emb_dim]

        # Add positional encoding (simple fixed vector here)
        x = x + self.pos_embed

        # Pass through transformer
        x = self.transformer_encoder(x)  # (B, 2, emb_dim)

        # Take the CLS token output
        cls_out = x[:, 0, :]  # (B, emb_dim)
        return self.fc(cls_out)  # (B, num_classes)
