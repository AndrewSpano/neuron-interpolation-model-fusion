import torch
import torch.nn as nn

from models.base_model import BaseModel, WeightRepresentation, LevelRepresentation


class PatchEmbedding(nn.Module):
    """Simple patch embedding for ViT-style model."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
    

class InputEmbedding(nn.Module):
    """Layer to handle the initial inputs of the ViT model."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim, dropout):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self._init_weights()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x.permute(0, 2, 1)  # [B, embed_dim, num_patches + 1]
        return x
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)


class PermutedTransformerEncoder(nn.TransformerEncoderLayer):
    """Transformer encoder layer with permuted input and output."""

    def __init__(self, is_last_encoder=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = None
        if is_last_encoder:
            self.norm = nn.LayerNorm(kwargs['d_model'])
        self.is_last_encoder = is_last_encoder

    def forward(self, x):
        """x has shape: [B, hidden_dim, seqlen]"""
        x = x.permute(0, 2, 1)  # [B, seqlen, hidden_dim]
        x = super().forward(x)  # [B, seqlen, hidden_dim]
        if self.is_last_encoder:
            x = self.norm(x)
            x = x[:, 0]  # [B, hidden_dim]  <-- Keep only the CLS token from the last encoder
        else:
            x = x.permute(0, 2, 1)  # [B, hidden_dim, seqlen]
        return x


class SmallViT(BaseModel):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=128, depth=4, num_heads=4, mlp_ratio=2.0, dropout=0.1,
                 embedder_separare_level=False, train_indices=None, val_indices=None):
        super().__init__(train_indices, val_indices)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dim_feedforward = int(embed_dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.dropout_p = dropout
        self.embedder_separare_level = embedder_separare_level

        self.blocks = nn.ModuleList()

        embedder = InputEmbedding(img_size, patch_size, in_chans, embed_dim, dropout=dropout)
        first_encoder = PermutedTransformerEncoder(
            is_last_encoder=depth == 1,
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=nn.GELU()
        )
        if not embedder_separare_level:
            self.blocks.append(nn.Sequential(embedder, first_encoder))
        else:
            self.blocks.append(embedder)
            self.blocks.append(first_encoder)

        if depth > 1:
            for idx in range(depth - 1):
                is_last_encoder = idx == depth - 2
                encoder = PermutedTransformerEncoder(
                    is_last_encoder=is_last_encoder,
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=self.dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    activation=nn.GELU()
                )
                self.blocks.append(encoder)

        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

    def forward_until_level(self, x: torch.Tensor, level_idx: int = -1) -> torch.Tensor:
        if level_idx == -1:
            return self.forward(x)
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == level_idx:
                return x

        return self.head(x)

    def copy_model(self) -> 'SmallViT':
        model_copy = SmallViT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout_p,
            embedder_separare_level=self.embedder_separare_level
        )
        model_copy.load_state_dict(self.state_dict())
        return model_copy.to(self.device)

    @property
    def get_ordered_trainable_named_layers(self) -> list[WeightRepresentation]:
        return []

    @property
    def get_ordered_trainable_named_levels(self) -> list[LevelRepresentation]:
        levels = []

        # add initial embedding layer (if separate)
        encoder_start_idx = 0
        if self.embedder_separare_level:
            levels.append(
                LevelRepresentation(
                    name='embedder',
                    output_width=self.embed_dim,
                    network=self.blocks[0]
                )
            )
            encoder_start_idx = 1

        # add transformer encoder layers
        for i, blk in enumerate(self.blocks[encoder_start_idx:]):
            levels.append(LevelRepresentation(
                name=f'block_{i}',
                output_width=self.embed_dim,
                network=blk
            ))

        # final classifier
        levels.append(LevelRepresentation(
            name='classifier',
            output_width=self.num_classes,
            network=self.head
        ))

        return levels
