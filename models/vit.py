"""Adapted from: https://github.com/omihub777/ViT-CIFAR."""
import torch
import torchsummary
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel, WeightRepresentation, LevelRepresentation


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, head: int, dropout: float):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.sqrt_d = self.hidden_dim ** 0.5

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, seqlen, _ = x.size()
        q = self.q(x).view(b, seqlen, self.head, self.hidden_dim // self.head).transpose(1,2)
        k = self.k(x).view(b, seqlen, self.head, self.hidden_dim // self.head).transpose(1,2)
        v = self.v(x).view(b, seqlen, self.head, self.hidden_dim // self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d, dim=-1)  # [batch, head, seqlen, n]
        attn = torch.einsum("bhij, bhjf->bihf", score, v)  # [batch, head, seqlen, hidden_dim // head]
        o = self.dropout(self.o(attn.flatten(2)))
        return o
    

class InputEmbedder(nn.Module):

    def __init__(self, img_size: int, patch: int, in_c: int, hidden_dim: int):
        super(InputEmbedder, self).__init__()
        self.img_size = img_size
        self.patch = patch
        self.in_c = in_c
        self.hidden_dim = hidden_dim
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * in_c  # for CIFAR: (32 / 8) ** 2 * 3 = 48 patch vec length
        num_tokens = (self.patch ** 2) + 1  # +1 for cls token

        self.emb = nn.Linear(f, hidden_dim)  # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)  # CLS token
        out = out + self.pos_emb
        out = out.permute(0, 2, 1)  # [batch, hidden_dim, seqlen]  <-- for neuron importance scores
        return out

    def _to_words(self, x):
        """(b, c, h, w) -> (b, n, f)"""
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0, 2, 3, 4, 5, 1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, hidden_dim: int, mlp_hidden: int, head: int, is_last_encoder: bool, dropout: float = 0.0):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(hidden_dim)
        self.msa = MultiHeadSelfAttention(hidden_dim, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.is_last_encoder = is_last_encoder

    def forward(self, x):
        """x: [batch, hidden_dim, seqlen]. This is because for neuron importance scores,
            the captum library requires the "output neurons" to be in the first dimension.
            output: [batch, hidden_dim, seqlen]."""
        x = x.permute(0, 2, 1)  # [batch, seqlen, hidden_dim]
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        if self.is_last_encoder:
            out = out[:, 0]
        else:
            out = out.permute(0, 2, 1)  # [batch, hidden_dim, seqlen]
        return out


class ViT(BaseModel):

    def __init__(
            self,
            in_c: int,
            num_classes: int = 10,
            img_size: int = 32,
            patch: int = 8,
            dropout: float = 0.0,
            depth: int = 7,
            hidden: int = 384,
            mlp_hidden: int = 384*4,
            head: int = 8,
            embedder_separare_level: bool = False,
            train_indices=None,
            val_indices=None
    ):
        super(ViT, self).__init__(train_indices, val_indices)
        self.in_c = in_c
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch = patch
        self.dropout_p = dropout
        self.depth = depth
        self.hidden = hidden
        self.mlp_hidden = mlp_hidden
        self.head = head
        self.embedder_separare_level = embedder_separare_level

        embedder = InputEmbedder(img_size, patch, in_c, hidden)
        encodder_list = [
            TransformerEncoder(
                hidden,
                mlp_hidden=mlp_hidden,
                dropout=dropout,
                head=head,
                is_last_encoder=(i == depth - 1)  # `True` for the last encoder
            )
            for i in range(depth)
        ]

        block_list = []
        if embedder_separare_level:
            block_list.append(embedder)
            block_list.extend(encodder_list)
        else:
            level0 = nn.Sequential(embedder, encodder_list[0])
            block_list.append(level0)
            block_list.extend(encodder_list[1:])

        self.blocks = nn.ModuleList(block_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        for enc in self.blocks:
            x = enc(x)
        return self.fc(x)
    
    def forward_until_level(self, x: torch.Tensor, level_idx: int = -1) -> torch.Tensor:
        if level_idx == -1:
            return self.forward(x)
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == level_idx:
                return x

        return self.fc(x)
    
    def copy_model(self) -> 'ViT':
        """Creates a copy of the model with the same architecture,
            and the same weights as the current model."""
        model_copy = ViT(
            in_c=self.in_c,
            num_classes=self.num_classes,
            img_size=self.img_size,
            patch=self.patch,
            dropout=self.dropout_p,
            depth=self.depth,
            hidden=self.hidden,
            mlp_hidden=self.mlp_hidden,
            head=self.head,
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
                    output_width=self.hidden,
                    network=self.blocks[0]
                )
            )
            encoder_start_idx = 1

        # add transformer encoder layers
        for i, blk in enumerate(self.blocks[encoder_start_idx:]):
            levels.append(LevelRepresentation(
                name=f'block_{i}',
                output_width=self.hidden,
                network=blk
            ))

        # final classifier
        levels.append(LevelRepresentation(
            name='classifier',
            output_width=self.num_classes,
            network=self.fc
        ))

        return levels



if __name__=="__main__":
    b, n, f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)
