import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # add batch dim
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads # num of heads
        assert d_model % heads == 0, "d_model must be divisible by heads"

        self.d_keys = d_model // heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask, dropout: nn.Dropout) -> torch.Tensor:
        # mask can be encoder mask or decoder mask
        d_k = queries.shape[-1]

        #  transpose the last two dimensions of the keys tensor, which switches the rows and columns.
        #  computes the dot product and scales the attention scores
        attention_scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # used in decoder
            attention_scores.masked_fill_(mask == 0, -1e9)
            # fill with a small number wherever mask = 0
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ values), attention_scores # (batch, heads, seq_len)  -> (batch, heads, seq_len, d_k)


    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask = None):
        
        queries = self.w_q(queries) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        keys = self.w_k(keys) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        values = self.w_v(values) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, heads, d_k) -> (batch, heads, seq_len, d_k)
        queries = queries.view(queries.shape[0], queries.shape[1], self.heads, self.d_keys).transpose(1, 2)
        keys = keys.view(keys.shape[0], keys.shape[1], self.heads, self.d_keys).transpose(1, 2)
        values = values.view(values.shape[0], values.shape[1], self.heads, self.d_keys).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttention.attention(queries, keys, values, mask, self.dropout)

        # Combine heads together
        # (batch, heads, seq_len, d_k) ->  (batch, seq_len, heads, d_k) -> (batch, seq_len, h*d_k or d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_keys)
        # continuous memory for the data
        return self.w_out(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward_block: PositionwiseFeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention,
                 feed_forward_block: PositionwiseFeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (bs, seq, d_model) -> (bs, seq, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = proj

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, num_heads: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # Embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos_embed = PositionalEncoding(src_seq_len, d_model, dropout)
    tgt_pos_embed = PositionalEncoding(tgt_seq_len, d_model, dropout)

    # Create encoder blocks
    encoder_blks = []
    for _ in range(N):
        enc_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        enc_feed_forward_blk = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoder_blk = EncoderBlock(enc_self_attention, enc_feed_forward_blk, dropout)
        encoder_blks.append(encoder_blk)

    encoder = Encoder(nn.ModuleList(encoder_blks))

    decoder_blks = []
    for _ in range(N):
        dec_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        dec_cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        dec_feed_forward_blk = PositionwiseFeedForward(d_model, d_ff, dropout)
        decoder_blk = DecoderBlock(dec_self_attention, dec_cross_attention, dec_feed_forward_blk, dropout)
        decoder_blks.append(decoder_blk)

    decoder = Decoder(nn.ModuleList(decoder_blks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos_embed, tgt_pos_embed, projection_layer)

    # Init parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
