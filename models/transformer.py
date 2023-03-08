import torch
import torch.nn as nn

from collections import OrderedDict


class TransformerMlp(nn.Module):
    def __init__(self, dim, dropout_prob, fc_dims):
        super(TransformerMlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, fc_dims),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_dims, dim),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim, dropout_prob, attn_heads, fc_dims):
        super(EncoderBlock, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=attn_heads, dropout=dropout_prob, bias=True
        )

        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.mlp = TransformerMlp(dim, dropout_prob, fc_dims)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        h = self.dropout(h)
        h = self.layer_norm1(h + x)

        h2 = self.mlp(h)
        h2 = self.dropout(h2)
        h2 = self.layer_norm2(h2 + h)

        return h2


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_heads: int,
        dropout_prob: float,
        n_enc_blocks: int,
        use_global: bool,
    ):
        """

        @param dim: embedding dimension size of Transformer encoding block
        @param attn_heads: number of multi-head attention units
        @param dropout_prob: dropout probability
        @param n_enc_blocks: number of stacked encoder blocks
        @param use_global: flag to get global embedding

        """
        super(Transformer, self).__init__()

        self.encoder = self._build_encoders(n_enc_blocks, dim, dropout_prob, attn_heads)
        self.dropout = nn.Dropout(dropout_prob)
        self.use_global = use_global
        if self.use_global:
            self.h_global = nn.Parameter(torch.rand(1, dim))
            self.register_parameter(name="global", param=self.h_global)

    def _build_encoders(self, n_enc_blocks, dim, dropout_prob, attn_heads):
        """
        Build stacked encoders
        @param n_enc_blocks:
        @param dropout_prob:
        @param attn_heads:
        @return:
        """
        enc_blocks = []
        for i in range(n_enc_blocks):
            enc_blocks.append(
                (f"encoder_{i}", EncoderBlock(dim, dropout_prob, attn_heads, dim))
            )
        encoder = nn.Sequential(OrderedDict(enc_blocks))
        return encoder

    def forward(self, x):
        """
        @param x: (batch size, n objects, embedding dim)
        @return:
        """
        if self.use_global:
            h_global = self.h_global.expand(x.shape[0], x.shape[-1])
            h_global = h_global.unsqueeze(1)
            x = torch.cat([h_global, x], dim=1)
        x = x.transpose(0, 1)
        # needs (sequence length, batch size, embedding dimension)
        # print("before update: ", x[0, 0, :])
        h = self.encoder(x)
        # print("after update", h[0, 0, :])
        return h.transpose(0, 1)


class TransformerClassifier(nn.Module):
    def __init__(self, in_dim: int, dim: int, max_seq: int, transformer: Transformer):
        super(TransformerClassifier, self).__init__()
        self.activation = nn.ReLU()
        self.pos_embedding = nn.Embedding(num_embeddings=max_seq, embedding_dim=dim)
        self.linear = nn.Linear(in_dim, dim)
        self.transformer = transformer
        self.output = nn.Linear(dim, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, p):
        h = self.activation(self.linear(x))
        h_pos = self.pos_embedding(p)
        h = h + h_pos
        h = self.transformer(h)
        y_hat = self.output(h[:, 0, :])

        if not self.training:
            y_hat = self.softmax(y_hat)

        return y_hat

    @classmethod
    def build(cls, in_dim, h_dim, attn_heads, encoder_blocks, max_seq_len):
        transformer = Transformer(
            dim=h_dim,
            attn_heads=attn_heads,
            n_enc_blocks=encoder_blocks,
            dropout_prob=0.0,
            use_global=True,
        )
        model = TransformerClassifier(
            in_dim, dim=h_dim, transformer=transformer, max_seq=max_seq_len
        )
        return model


if __name__ == "__main__":
    # needs (batch size, sequence length, channels)
    x = torch.randn(4, 5, 40)
    p = torch.arange(0, 5, dtype=torch.long)
    d = 128
    t = Transformer(
        dim=d, attn_heads=8, n_enc_blocks=1, dropout_prob=0.0, use_global=True
    )
    model = TransformerClassifier(40, dim=d, transformer=t, max_seq=5)
    y = model(x, p)
    print(y.shape)
