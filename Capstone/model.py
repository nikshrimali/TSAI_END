# model.py - File contains code for BERT model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.legacy.data import Field, BucketIterator, TabularDataset, Example, Dataset

# Encoder

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_Layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 100):

        
        super().__init__()

        self.device = device

        # below we are breaking down the embedding into input and positional embedding
        self.tok_embd = nn.Embedding(num_embeddings=input_dim, embedding_dim=hid_dim)
        self.pos_embd = nn.Embedding(max_length, hid_dim)

        # We also add layers for multi-headed processing

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  pf_dim,
                                                  n_heads,
                                                  dropout,
                                                  device)
                                    for _ in range(n_Layers)])

        # When we add two embeddings, we multiply our embeddings with a scale parameter, which helps us to maintain
        # our values in a certain range

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        # Creating linear layer to reduce the dimension of the vector


        # We also add a dropout value for regularization

        self.dropout = nn.Dropout(dropout)

    
    def forward(self, input_src, src_mask):

        # input_src = [batch_size, src_len]
        batch_size = input_src.shape[0]
        src_len = input_src.shape[1]

        # Is src_len same in all the cases?

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size,1).to(self.device)
        # print(f'Pos shape {pos.shape}, {input_src.shape}')
        #pos = [batch_size, src_len]

        input_embd = self.tok_embd(input_src)
        pos_embd = self.pos_embd(pos)

        # input_embd = pos_embd = [batch_size, src_len, embedding_dim]
        src = self.dropout(input_embd*self.scale + pos_embd)
        # src = [batch_size, src_len, hid_dim]

        # what does encoder returns?
        # 
        for layer in self.layers:
            src = layer(src, src_mask)

        return src

# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 pf_dim,
                 n_heads,
                 dropout,
                 device):
        
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim) # Layer norm after attention layer
        self.self_ff_layer_norm = nn.LayerNorm(hid_dim) # Layer norm after feed forward layer
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device) # Multi-head attention layer
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, src, src_mask):
        # what is src mask?????
        # src = [batch_size, src_len, hid_dim]
        # Why source has the hidden dim?
        # src_mask = [batch_size, 1, 1, src_len]
        
        _src, _ = self.self_attention(src, src, src, src_mask) # Self attention layer
        src = self.self_attn_layer_norm(self.dropout(_src) + src) # Add and Norm layer with residual connection

        # src = [batch_size, src_len, hid_dim]
        # Pointwise feedforward
        _src = self.positionwise_feedforward(src)
        src = self.self_ff_layer_norm(self.dropout(_src) + src)
        # src = [batch_size, src len, hid_dim]
        return src

# Multi headed attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 dropout,
                 device):
        
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim//n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]
        # query = [batch size, query_len, hid_dim]
        # key = [batch size, key_len, hid_dim]
        # value = [batch size, value_len, hid_dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query_len, hid_dim]
        # K = [batch size, key_len, hid_dim]
        # V = [batch size, value_len, hid_dim]

        # How hid_dim is divided into n_heads, head_dim?
        # print('Attention batch, heads, head_dim', batch_size, -1, self.n_heads, self.head_dim)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q=K=V = [batch_size, n_heads, query/key/value_len, head_dim]

        # Transposing the input embedding and then doing matrix multiplication

        # [batch_size, src_len, src_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))/ self.scale
        # print('Shape of energy', energy.shape)
        # energy = [batch_size, n_heads, query_len, key_length]
        # As you can notice the head dim is not there in the energy vector. only query_len and key_len are there

        # This energy is then multiplied with the value to calculate attention

        if mask is not None:
            # If mask values are close to zero, set it to very small values, we do this because?
            energy = energy.masked_fill(mask == 0, -1e10)

        # After matrix mul of query and key, and scaling we will apply softmax to get the output in a distribution of 0 to +1.
        # This value will act as an attention vector for us

        attention = torch.softmax(energy, dim=-1)

        # Attention is then further multiplied by the values to get the contextual embeddings

        x = torch.matmul(self.dropout(attention), V)

        #x = [batch_size, query_length, n_heads, hid_dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        #x = [batch_size, n_heads, query_length, hid_dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch_size, query_len, hid_dim]

        x = self.fc_o(x)

        #x = [batch_size, query_len, hid_dim]

        return x, attention

# Positionwise feed forward layer
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, seq_len, hid_dim] 
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        # x = [batch_size, seq_len, hid_dim]

        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=10000):

        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)

        self.tok_embd = nn.Embedding(output_dim, hid_dim)
        self.pos_embd = nn.Embedding(output_dim, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.fc_out = nn.Linear(hid_dim, output_dim)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):

        # src = [batch_size, trg_len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size,1).to(device)

        # pos = [batch_size, trg_len]

        trg = self.dropout(self.tok_embd(trg)* self.scale + self.pos_embd(pos))

        for layer in self.layers:
            # Why src_mask and trg_mask
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
        # output = [batch_size, trg_len, output_dim, ]
        return output, attention


# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_lyr_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_lyr_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch_size, trg_len, hid_dim]
        # enc_src = [batch_size, src_len, hid_dim]
        # trg_mask = [batch_size, 1, trg_len, trg_len]
        # src_mask = [batch_size, 1, 1, src_len]

        # Self 
        # print('Target shape and mask', trg.shape, trg_mask.shape)
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # Layer Norm - Dropout, Relu, residual connection
        trg = self.self_attn_lyr_norm(self.dropout(_trg) + trg)

        # query, key, value
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_lyr_norm(self.dropout(_trg) + trg)

        trg = self.positionwise_feedforward(trg)
        # trg = [batch_size, trg_len, hid_dim]
        # attention = [batch_size, n_heads, trg_len, src_len]

        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention

# Seq2Seq

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):

        # src = [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch_size, 1, 1, src_len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch_size, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # print(f'Make target mask {trg.shape}, {trg_pad_mask.shape}')
        # trg_mask = [batch_size, 1, 1, trg_len]
    
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        
        #trg_mask = [batch_size, 1, trg_len, trg_len]
        
        trg_mask = trg_pad_mask & trg_sub_mask # What is this & operator???

        # trg_mask = [batch_size, 1, trg_len, trg_len]
        # print(f'Target mask shape make_trg_mask {trg_mask.shape}')

        return trg_mask

    def forward(self, src, trg):
        # src = [batch_size, src_len]
        # trg = [batch_size, trg_len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch_size, src_len, hid_dim, output_dim]
        
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch_size, trg_len, output_dim]
        # attention = [batch_size, n_heads, trg_len, src_len]

        return output, attention