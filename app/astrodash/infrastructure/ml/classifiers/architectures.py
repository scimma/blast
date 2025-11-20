import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
import math

# MLPs
class singlelayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(singlelayerMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = [64,64]):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_dim)):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim[i]))
            else:
                layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# Additional positional encoding classes
class learnable_fourier_encoding(nn.Module):
    def __init__(self, dim = 64):
        '''
        Learnable Fourier encoding for position,
        MLP([sin(fc(x)), cos(fc(x))])
        Args:
            dim: dimension
        '''
        super(learnable_fourier_encoding, self).__init__()
        self.freq = nn.Linear(1, dim, bias=False)
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = x[:, :, None]
        encoding = torch.cat([torch.sin(self.freq(x)),
                              torch.cos(self.freq(x))], dim=-1)
        encoding = F.relu( self.fc1(encoding) )
        encoding = self.fc2(encoding)
        return encoding

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        '''
        The usual sinusoidal positional encoding
        args:
            dim: the dimension
        '''
        super().__init__()
        self.dim = dim
        self.div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        # Create the positional encoding matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        sine = torch.sin(x[:,:,None] * self.div_term[None,None,:].to(x.device))
        cosine = torch.cos(x[:,:,None] * self.div_term[None,None,:].to(x.device))
        return torch.cat([sine, cosine], dim=-1)

# Relative position encoding
class RelativePosition(nn.Module):
    '''
    relative positional encoding for discrete distances
    '''
    def __init__(self, num_units: int, max_relative_position: int):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q: int, length_k: int) -> torch.Tensor:
        range_vec_q = torch.arange(length_q, device=self.embeddings_table.device)
        range_vec_k = torch.arange(length_k, device=self.embeddings_table.device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings

# Multi-head attention with relative positioning
class MultiHeadAttentionLayer_relative(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: torch.device):
        '''
        Multiheaded attention with relative positional encoding
        '''
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.register_buffer('scale', scale)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, query len, hid dim]

        return x

# Image processing components
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)  # (B, dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        return x

# Generic transformer model
class TransformerModel(nn.Module):
    '''
    A minimal transformer model
    '''
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1, selfattn = True):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, selfattn)
            for _ in range(num_layers)
        ])

    def forward(self, x, context=None):
        for layer in self.layers:
            x = layer(x, context)
        return x

# Dash CNN architecture
class AstroDashPyTorchNet(nn.Module):
    """
    PyTorch implementation of the AstroDash CNN.
    """
    def __init__(self, n_types, im_width=32):
        super().__init__()
        self.im_width = im_width

        # Layer1: Conv(1→32) + ReLU + MaxPool
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Match TF behavior
        )

        # Layer2: Conv(32→64) + ReLU + MaxPool
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Match TF behavior
        )

        # FC layers (only use the main path, ignore the parallel branch)
        self.fc1 = nn.Linear(4096, 1024)  # 64 * 8 * 8 = 4096
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(1024, n_types)

    def forward(self, x):
        x = x.view(-1, 1, self.im_width, self.im_width)

        # Layer1
        h_pool1 = self.layer1(x)

        # Layer2
        h_pool2 = self.layer2(h_pool1)

        # Reshape to match TensorFlow's flattening
        # PyTorch: (batch, channels, height, width) -> (batch, channels * height * width)
        # TensorFlow: (batch, height, width, channels) -> (batch, height * width * channels)
        # Both should result in (batch, 64 * 8 * 8) = (batch, 4096)
        # But we need to transpose to match TensorFlow's channel ordering
        h_pool2_transposed = h_pool2.permute(0, 2, 3, 1)  # NCHW -> NHWC
        h_pool2_flat = h_pool2_transposed.reshape(h_pool2.size(0), -1)
        h_fc1 = F.relu(self.fc1(h_pool2_flat))

        # Readout: dropout on h_fc1, then output (ignore the parallel branch)
        h_fc_drop = self.dropout(h_fc1)
        output = self.output(h_fc_drop)

        return F.softmax(output, dim=1)

# Transformer blocks
class SinusoidalMLPPositionalEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.div_term = torch.exp(torch.arange(0, dim).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sine = torch.sin(x[:,:,None] * self.div_term[None,None,:].to(x.device))
        cosine = torch.cos(x[:,:,None] * self.div_term[None,None,:].to(x.device))
        encoding = torch.cat([sine, cosine], dim=-1)
        encoding = F.relu(self.fc1(encoding))
        encoding = self.fc2(encoding)
        return encoding

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1,
                 context_self_attn: bool = False):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                                dropout=dropout, batch_first=True)
        if context_self_attn:
            self.context_self_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                                dropout=dropout, batch_first=True)
            self.layernorm_context = nn.LayerNorm(embed_dim)
        else:
            self.context_self_attn = nn.Identity()
            self.layernorm_context = nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        if context is not None:
            if not isinstance(self.context_self_attn, nn.Identity):
                context_attn_output, _ = self.context_self_attn(context, context, context, key_padding_mask=context_mask)
                context = self.layernorm_context(context + self.dropout(context_attn_output))
            cross_attn_output, _ = self.cross_attn(x, context, context, key_padding_mask=context_mask)
            x = self.layernorm2(x + self.dropout(cross_attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout(ffn_output))
        return x

# Transformer Classifier Architecture
class spectraTransformerEncoder(nn.Module):
    def __init__(self,
                 bottleneck_length: int,
                 model_dim: int,
                 num_heads: int,
                 num_layers: int,
                 num_classes: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 selfattn: bool = False):
        super().__init__()
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        self.redshift_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)
        self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)
        self.flux_embd = nn.Linear(1, model_dim)
        self.transformerblocks = nn.ModuleList([
            TransformerBlock(model_dim, num_heads, ff_dim, dropout, selfattn)
            for _ in range(num_layers)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, num_classes)
        )
    def forward(self, wavelength, flux, redshift, mask=None):
        flux_embd = self.flux_embd(flux[:, :, None]) + self.wavelength_embd_layer(wavelength)
        redshift_embd = self.redshift_embd_layer(redshift[:, None])
        if redshift_embd.dim() == 4 and redshift_embd.shape[2] == 1:
            redshift_embd = redshift_embd.squeeze(2)
        context = torch.cat([flux_embd, redshift_embd], dim=1)
        if mask is not None:
            mask = torch.cat([mask, torch.zeros(mask.shape[0], 1).to(torch.bool).to(mask.device)], dim=1)
        x = self.initbottleneck[None, :, :].repeat(context.shape[0], 1, 1)
        h = x
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, context, context_mask=mask)
        final_bottleneck = x + h
        pooled_features = self.pooling(final_bottleneck.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(pooled_features)
        return logits
