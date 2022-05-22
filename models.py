import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from IPython import display

import imageio
import pickle
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from utils import snr_db2sigma, errors_ber, errors_bitwise_ber, errors_bler, min_sum_log_sum_exp, moving_average, extract_block_errors, extract_block_nonerrors

from polar import *
from pac_code import *

from sklearn.manifold import TSNE
import math
import random
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import sys
import csv





class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None,causal=False):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            # if args.model == 'gpt':
            #     attn = attn.masked_fill(mask == 0, -1e9)
            # else:
            mask=mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class ScalarMult(nn.Module):
    '''scalar multiplication layer'''

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(1e-10*torch.ones(1))

    def forward(self, x):
        out = self.alpha*x
        return out

        
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scalar = ScalarMult()


    def forward(self, q, k, v, mask=None,causal=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # if len(list(q.size()))==4:
        #     q = q.view(q.size(0)*sz_b,q.size(2),q.size(3),q.size(4)).transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # else:
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        #q = self.scalar(q)
        q += residual
        q = self.layer_norm(q)
        

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.scalar = ScalarMult()

    def forward(self, x):

        residual = x

        x = self.w_2(F.gelu(self.w_1(x))) #F.gelu
        x = self.dropout(x)
        #x = self.scalar(x)
        x += residual

        x = self.layer_norm(x)

        return x
        

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None, cross_attend=True):
        dec_enc_attn=[]
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        if cross_attend:
            dec_output, dec_enc_attn = self.enc_attn(
                dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200,num=10000):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid,num))

    def _get_sinusoid_encoding_table(self, n_position, d_hid,num):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position,num):
            return [position / np.power(num, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i,num) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class XFormerEncoder(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(XFormerEncoder,self).__init__()
        self.embed_dim = config.embed_dim
        self.block_len = config.max_len
        self.pos_emb = nn.Embedding(config.N+1, config.embed_dim,padding_idx=0)
        self.position_enc = PositionalEncoding(self.embed_dim, n_position=self.block_len)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(config.embed_dim, config.embed_dim*4, config.n_head, config.embed_dim//config.n_head, config.embed_dim//config.n_head, dropout=config.dropout)
            for _ in range(config.n_layers)])
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)

    def forward(self,noisy_enc,src_mask,device,return_attns=False):
        position_indices = torch.arange(1,self.block_len+1, device=device)
        pos_enc = self.pos_emb(position_indices)
        enc_output = noisy_enc*pos_enc   #<---- addition instead of multiplication?
        enc_output = self.position_enc(enc_output)

        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)
        enc_slf_attn_list = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output # [b_size,block_len,embed_dim]

class XFormerDecoder(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(XFormerDecoder,self).__init__()
        self.embed_dim = config.embed_dim
        self.block_len = config.max_len
        self.emb_auto = nn.Embedding(config.N+1, config.embed_dim,padding_idx=0)
        self.emb_cross = nn.Embedding(config.N+1, config.embed_dim,padding_idx=0)
        self.emb_inputs = nn.Embedding(4, config.embed_dim,padding_idx=3)
        self.position_enc_auto = PositionalEncoding(self.embed_dim, n_position=self.block_len)
        self.position_enc_cross = PositionalEncoding(self.embed_dim, n_position=self.block_len,num=5000)
        self.dropout = nn.Dropout(p=config.dropout)
        self.dropout_cross = nn.Dropout(p=config.dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(config.embed_dim, config.embed_dim*4, config.n_head, config.embed_dim//config.n_head, config.embed_dim//config.n_head, dropout=config.dropout)
            for _ in range(config.n_layers)])
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.layer_norm_cross = nn.LayerNorm(config.embed_dim, eps=1e-6)


    def forward(self,noisy_enc,src_mask,trg_seq,trg_mask,device,return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []
        position_indices = torch.arange(1,self.block_len+1, device=device)
        emb_self = self.emb_auto(position_indices)
        emb_cross = self.emb_cross(position_indices)
        enc_output = noisy_enc*emb_cross   #<---- addition instead of multiplication?
        dec_output = self.emb_inputs(trg_seq)
        enc_output = self.position_enc_cross(enc_output)
        dec_output = self.position_enc_auto(dec_output)

        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm(dec_output)

        enc_output = self.dropout_cross(enc_output)
        enc_output = self.layer_norm_cross(enc_output)

        cross_attend = [False for _ in self.layer_stack]
        cross_attend[0] = True
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list
        return dec_output # [b_size,block_len,embed_dim]

class XFormerGPT(nn.Module):
    def __init__(self, config, layer_idx=None):
        super(XFormerGPT,self).__init__()
        self.embed_dim = config.embed_dim
        self.block_len = config.max_len
        self.position_enc_auto = PositionalEncoding(self.embed_dim, n_position=self.block_len)
        self.dropout = nn.Dropout(p=config.dropout)
        #self.pos_emb = nn.Embedding(config.N, config.embed_dim)
        #self.dropout_cross = nn.Dropout(p=config.dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(config.embed_dim, config.embed_dim*4, config.n_head, config.embed_dim//config.n_head, config.embed_dim//config.n_head, dropout=config.dropout)
            for _ in range(config.n_layers)])
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.layer_norm_cross = nn.LayerNorm(config.embed_dim, eps=1e-6)


    def forward(self,trg_seq,trg_mask,device,return_attns=False,return_layer=None):
        #position_indices = torch.arange(1,self.block_len+1, device=device)
        #pos_enc = self.pos_emb(position_indices)
        dec_slf_attn_list, dec_enc_attn_list = [], []
        dec_output = self.position_enc_auto(trg_seq)
        dec_output = self.dropout(dec_output)
        #dec_output = self.layer_norm(dec_output)
        layer=1
        intermediate_layer_out = None
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, slf_attn_mask=trg_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            if return_layer is not None:
                if layer == return_layer:
                    intermediate_layer_out = dec_output
            layer += 1
        if return_attns:
            return dec_output, dec_slf_attn_list
        if return_layer is not None:
            return dec_output, intermediate_layer_out
        return dec_output # [b_size,block_len,embed_dim]



class XFormerEndToEndGPT(nn.Module):
    def __init__(self,config):
        super(XFormerEndToEndGPT,self).__init__()
        self.embed_dim = config.embed_dim
        self.block_len = config.max_len
        self.trg_pad_idx = 2
        self.start_embed_layer = nn.Sequential(
            nn.Linear(config.N,self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim,self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim,self.embed_dim),
            )
        self.learnt_pos = True
        if not self.learnt_pos:
            self.emb_inputs = nn.Embedding(2, self.embed_dim)
            #self.emb_inputs = nn.Embedding(4, self.embed_dim,padding_idx=3)
        else:
            self.pos_emb = nn.Embedding(self.block_len, config.embed_dim)
        self.layer_norm_inp = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm_out = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.Decoder = XFormerGPT(config)
        self.Lin_Decoder = nn.Linear(config.embed_dim,1)

    def forward(self,noisy_enc,mask,trg_seq,device,return_layer = None):
        src_mask = mask
        trg_seq = trg_seq[:,:-1]
        if not self.learnt_pos:
            trg_seq = torch.cat((torch.ones((trg_seq.size(0),1),device=device).long(),(trg_seq==-1).long()),-1) # shift inputs forward by one token
            trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq) # batch_size x max_len x max_len
            trg_seq = self.emb_inputs(trg_seq)
        else:
            trg_seq = torch.cat((torch.ones((trg_seq.size(0),1),device=device),trg_seq),-1) # shift inputs forward by one token
            trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq) # batch_size x max_len x max_len
            trg_seq = torch.ones(self.embed_dim,device=device)*trg_seq.unsqueeze(-1)
            position_indices = torch.arange(self.block_len, device=device)
            pos_enc = self.pos_emb(position_indices)
            trg_seq = trg_seq*pos_enc

        start_emb = self.start_embed_layer(noisy_enc)
        trg_seq[:,0] = start_emb
        if return_layer is not None:
            output,intermediate_layer_out = self.Decoder(trg_seq,trg_mask,device,return_layer=return_layer)
        else:
            output = self.Decoder(trg_seq,trg_mask,device)
        logits = self.Lin_Decoder(output)

        decoded_msg_bits = logits.sign()
        output = torch.sigmoid(logits)
        output = torch.cat((1-output,output),-1)
        out_mask = mask

        if return_layer is not None:
            return output,decoded_msg_bits,out_mask,logits,intermediate_layer_out

        return output,decoded_msg_bits,out_mask,logits # [b_size,block_len,2]

    def decode(self,noisy_enc,info_positions,mask,device):
        start_emb = self.start_embed_layer(noisy_enc)
        inp_seq = torch.ones((noisy_enc.size(0),self.block_len,self.embed_dim),device=device)
        inp_seq[:,0] = start_emb
        inp_mask = mask.unsqueeze(1) & get_subsequent_mask(noisy_enc)
        output_bits = torch.ones((noisy_enc.size(0),self.block_len),device=device)
        for i in range(noisy_enc.size(1)):
            if i in info_positions:
                mask_i = inp_mask[:,i,:].unsqueeze(1)
                output = self.Decoder(inp_seq,mask_i,device)
                output = self.Lin_Decoder(output)
                next_bit = output[:,i].sign()
            else:
                next_bit = torch.ones((noisy_enc.size(0),1),device=device)
            output_bits[:,i] = next_bit[:,0]
            #print(next_bit)
            if i < noisy_enc.size(1)-1:
                if not self.learnt_pos:
                    embed_next_bit = self.emb_inputs((next_bit==1).long())
                    inp_seq[:,i+1] = embed_next_bit[:,0]
                else:
                    embed_next_bit = next_bit*self.pos_emb(torch.tensor(i+1,device=device)).unsqueeze(0)
                    inp_seq[:,i+1] = embed_next_bit

        out_mask = mask
        return output_bits,out_mask
        
class StartEmbedder(nn.Module):
    def __init__(self,inp_dim,hidden_dim,num_layers):
        super(StartEmbedder,self).__init__()
        self.inp_dim = inp_dim 
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([nn.Linear(self.inp_dim,self.hidden_dim)]+[nn.Linear(hidden_dim,hidden_dim) for i in range(num_layers-1)])
        
    def forward(self,x):
        out = self.layers[0](x)
        res = out
        out = F.gelu(out)
        for layer in self.layers[1:-1]:
            out = layer(out)
            out = F.gelu(out)
        out = self.layers[-1](out)
        out = out + res
        return out
        
class rnnAttn(nn.Module):
    def __init__(self, args):
        super(rnnAttn, self).__init__()
        
        #self.vocab_size = params['vocab_size']
        self.d_emb = 1#args.embed_dim#params['d_emb']
        self.d_hid = args.embed_dim#params['d_hid']
        self.block_len = args.N
        self.n_layer = 2
        self.btz = args.batch_size
        self.feature1 = multiplyFeature(args.mat)
        #self.encoder = nn.Embedding(self.vocab_size, self.d_emb)
        self.attn = Attention(self.d_hid)
        self.rnn = nn.GRU(self.d_emb, self.d_hid, self.n_layer, batch_first=True)
        self.startEmbedder1 = StartEmbedder(args.N,self.d_hid,3)
        self.startEmbedder2 = StartEmbedder(args.N,self.d_hid,3)
        # the combined_W maps the combined hidden states and context vectors to d_hid 
        self.combined_W = nn.Linear(self.d_hid * 3, self.d_hid)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_hid,self.d_hid),
            nn.GELU(),
            nn.Linear(self.d_hid,self.d_hid),
            nn.GELU(),
            nn.Linear(self.d_hid,1),
            )
        

    def forward(self,noisy_enc,mask,trg_seq,device,return_layer = None, return_attn_weights=False):
        
        """
            IMPLEMENT ME!
            Copy your implementation of RNNLM, make sure it passes the RNNLM check
            In addition to that, you need to add the following 3 things
            1. pass rnn output to attention module, get context vectors and attention weights
            2. concatenate the context vec and rnn output, pass the combined
               vector to the layer dealing with the combined vectors (self.combined_W)
            3. if return_attn_weights, instead of return the [N, L, V]
               matrix, return the attention weight matrix
               of dimension [N, L, L] which returned from the forrward function of Attnetion module
        """
        batch_size, seq_len= noisy_enc.shape
        #multFeat = self.feature1(noisy_enc,device)
        trg_seq = trg_seq[:,:-1]
        trg_seq = torch.cat((torch.ones((trg_seq.size(0),1),device=device).long(),trg_seq),-1)
        start_hidden = self.startEmbedder1(noisy_enc)
        #dumb_decode = self.startEmbedder2(multFeat)
        hidden = torch.cat((start_hidden.unsqueeze(1),start_hidden.unsqueeze(1)),1)
        hidden = torch.transpose(hidden,0,1)
        hidden = hidden.contiguous()
        start_hidden = (torch.ones((batch_size,seq_len,1),device=device)*start_hidden.unsqueeze(1))
        #dumb_decode = (torch.ones((batch_size,seq_len,1),device=device)*dumb_decode.unsqueeze(1))
        #init=torch.zeros(self.n_layer, batch_size, self.d_hid).to(device)
        #wordvecs = self.encoder(batch)
        #print(hidden.size())
        outs,last_hidden = self.rnn(trg_seq.unsqueeze(-1),hidden)
        context_vec,attn_weights = self.attn(outs)
        
        cat_vec = torch.cat((context_vec,outs,start_hidden),dim = -1)
        dec = self.combined_W(cat_vec)
        logits = self.decoder(torch.tanh(dec))
        
        decoded_msg_bits = logits.sign()
        output = torch.sigmoid(logits)
        output = torch.cat((1-output,output),-1)
        out_mask = mask
        
        return output,decoded_msg_bits,out_mask,logits
    
    def decode(self,noisy_enc,info_positions,mask,device):
        batch_size, seq_len= noisy_enc.shape
        inp_seq = torch.ones((noisy_enc.size(0),self.block_len),device=device)
        inp_seq[:,0] = 1
        output_bits = torch.ones((noisy_enc.size(0),self.block_len),device=device)
        
        #multFeat = self.feature1(noisy_enc,device)
        start_hidden = self.startEmbedder1(noisy_enc)
        #dumb_decode = self.startEmbedder2(multFeat)
        
        hidden = torch.cat((start_hidden.unsqueeze(1),start_hidden.unsqueeze(1)),1)
        hidden = torch.transpose(hidden,0,1)
        hidden = hidden.contiguous()
        
        outs_arr = torch.ones((noisy_enc.size(0),self.block_len,self.d_hid),device=device)
        
        start_hidden = (torch.ones((batch_size,seq_len,1),device=device)*start_hidden.unsqueeze(1))
        #dumb_decode = (torch.ones((batch_size,seq_len,1),device=device)*dumb_decode.unsqueeze(1))
        for i in range(noisy_enc.size(1)):
            if i in info_positions:
                outs,last_hidden = self.rnn(inp_seq[:,i].unsqueeze(-1).unsqueeze(-1),hidden)
                outs_arr[:,i,:] = outs.squeeze()
                context_vec,_ = self.attn(outs_arr)
                
                cat_vec = torch.cat((context_vec,outs_arr,start_hidden),dim = -1)
                dec = self.combined_W(cat_vec)
                logits = self.decoder(torch.tanh(dec))
                hidden = last_hidden
                next_bit = logits[:,i].sign().squeeze()
            else:
                outs,last_hidden = self.rnn(inp_seq[:,i].unsqueeze(-1).unsqueeze(-1),hidden)
                outs_arr[:,i,:] = outs.squeeze()
                hidden = last_hidden
                next_bit = torch.ones((noisy_enc.size(0)),device=device)
            #print(output_bits[:,i].size())
            #print(next_bit.size())
            output_bits[:,i] = next_bit
            #print(next_bit)
            if i < noisy_enc.size(1)-1:
                inp_seq[:,i+1] = next_bit
        out_mask = mask
        return output_bits,out_mask
        
class Attention(nn.Module):
    def __init__(self, d_hidden):
        super(Attention, self).__init__()
        self.linear_w1 = nn.Linear(d_hidden, d_hidden)
        self.linear_w2 = nn.Linear(d_hidden, 1)
        
    
    def forward(self, x):
      
        """
            IMPLEMENT ME!
            For each time step t
                1. Obtain attention scores for step 0 to (t-1)
                   This should be a dot product between current hidden state (x[:,t:t+1,:])
                   and all previous states x[:, :t, :]. While t=0, since there is not
                   previous context, the context vector and attention weights should be of zeros.
                   You might find torch.bmm useful for computing over the whole batch.
                2. Turn the scores you get for 0 to (t-1) steps to a distribution.
                   You might find F.softmax to be helpful.
                3. Obtain the sum of hidden states weighted by the attention distribution
            Concat the context vector you get in step 3. to a matrix.
            
            Also remember to store the attention weights, the attention matrix 
            for each training instance should be a lower triangular matrix. Specifically,
            each row, element 0 to t-1 should sum to 1, the rest should be padded with 0.
            e.g. 
            [ [0.0000, 0.0000, 0.0000, 0.0000],
              [1.0000, 0.0000, 0.0000, 0.0000],
              [0.4246, 0.5754, 0.0000, 0.0000],
              [0.2798, 0.3792, 0.3409, 0.0000] ]
            
            Return the context vector matrix and the attention weight matrix
            
        """
        batch_seq_len = x.shape[1]
        modif_hidden = self.linear_w1(x)
        attn_logits = torch.bmm(x,modif_hidden.transpose(1,2))
        mask = torch.triu(-10000000000000000.0*torch.ones((batch_seq_len,batch_seq_len),device=x.device))
        attn_weights = nn.functional.softmax(attn_logits + mask,-1)
        mult_mask = torch.ones(attn_weights.shape,device=x.device)
        mult_mask[:,0,:]=0
        attn_weights = attn_weights*mult_mask
        context_vecs = torch.bmm(attn_weights,x)
        return context_vecs, attn_weights
        
class XFormerEndToEndDecoder(nn.Module):
    def __init__(self,config):
        super(XFormerEndToEndDecoder,self).__init__()
        self.embed_dim = config.embed_dim
        self.block_len = config.max_len
        self.trg_pad_idx = 3
        self.start_idx = 2
        self.Decoder = XFormerDecoder(config)
        self.Lin_Decoder = nn.Linear(config.embed_dim,1)

    def forward(self,noisy_enc,mask,trg_seq,device):
        src_mask = mask
        trg_seq = trg_seq[:,:-1]
        trg_seq = torch.cat((2*torch.ones((trg_seq.size(0),1),device=device).long(),(trg_seq==1).long()),-1)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        batch_size = trg_mask.size(0)
        max_len = trg_mask.size(1)
        trg_mask = trg_mask.view(batch_size*max_len,max_len)
        trg_maskh = torch.cat((trg_mask[:,1:],torch.zeros((trg_mask.size(0),1),device=device)),-1).float()
        trg_seq = (trg_seq*torch.ones((max_len,batch_size,max_len),device=device)).long().permute((1,0,2)).reshape(batch_size*max_len,max_len)
        noisy_enc = (noisy_enc*torch.ones((max_len,batch_size,max_len),device=device)).permute((1,0,2)).reshape(batch_size*max_len,max_len)
        src_mask = (src_mask*torch.ones((max_len,batch_size,max_len),device=device)).permute((1,0,2)).reshape(batch_size*max_len,max_len)

        #noisy_enc : [b_size, block_len]
        output = torch.ones(self.embed_dim,device=device)*noisy_enc.unsqueeze(-1)
        #print(trg_mask.size())
        #noisy_enc : [b_size,block_len,embed_dim]

        output = self.Decoder(output,src_mask,trg_seq,trg_mask,device)
        logits = self.Lin_Decoder(output)
        decoded_msg_bits = logits.sign()
        output = torch.sigmoid(logits)
        output = torch.cat((1-output,output),-1)
        out_mask = trg_mask.float() - trg_maskh
        return output,decoded_msg_bits,out_mask,logits # [b_size,block_len,2]

    def decode(self,noisy_enc,info_positions, mask,device):
        enc_input = torch.ones(self.embed_dim,device=device)*noisy_enc.unsqueeze(-1)
        inp_seq = torch.ones((noisy_enc.size(0),self.block_len),device=device).long()
        inp_seq[:,0] = 2
        inp_mask = mask.unsqueeze(1) & get_subsequent_mask(noisy_enc)
        output_bits = torch.ones((noisy_enc.size(0),self.block_len),device=device)
        for i in range(noisy_enc.size(1)):
            if i in info_positions:
                mask_i = inp_mask[:,i,:]
                output = self.Decoder(enc_input,mask,inp_seq,mask_i,device)
                output = self.Lin_Decoder(output)
                next_bit = output[:,i].sign()
            else:
                next_bit = torch.ones((noisy_enc.size(0),1),device=device)
            output_bits[:,i] = next_bit[:,0]
            embed_next_bit = ((next_bit==1).long())
            if i < noisy_enc.size(1)-1:
                inp_seq[:,i+1] = embed_next_bit[:,0]
        out_mask = mask
        return output_bits,out_mask

    

        
    


class XFormerEndToEndEncoder(nn.Module):
    def __init__(self,config):
        super(XFormerEndToEndEncoder,self).__init__()
        self.embed_dim = config.embed_dim
        self.block_len = config.max_len
        self.Encoder = XFormerEncoder(config)
        self.Lin_Decoder = nn.Linear(config.embed_dim,1)

    def forward(self,noisy_enc,mask,trg_seq,device):
        #noisy_enc : [b_size, block_len]
        output = torch.ones(self.embed_dim,device=device)*noisy_enc.unsqueeze(-1)
        #noisy_enc : [b_size,block_len,embed_dim]
        output = self.Encoder(output,mask,device)
        logits = self.Lin_Decoder(output)
        decoded_msg_bits = logits.sign()
        output = torch.sigmoid(logits)
        output = torch.cat((1-output,output),-1)
        out_mask = mask
        return output,decoded_msg_bits,out_mask,logits # [b_size,block_len,2]

    def decode(self,noisy_enc,info_positions,mask,device,trg_seq=None):
        _,decoded_msg_bits,out_mask,_ = self.forward(noisy_enc,mask,trg_seq,device)
        #decoded_msg_bits = (decoded_msg_bits==1).long()
        return decoded_msg_bits,out_mask




class convNet(nn.Module):
    def __init__(self,config):
        super(convNet,self).__init__()
        self.hidden_dim = config.embed_dim
        self.input_len = config.max_len
        self.output_len = config.N
        bias = not config.dont_use_bias
        self.kernel = 7
        self.padding = int((self.kernel-1)/2)
        
        self.layers1 = nn.Sequential(
            nn.Conv1d(1,int(self.hidden_dim/2),self.kernel,padding=self.padding,bias=bias),
            nn.GELU(),
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=2*self.padding,dilation=2,bias=bias),
            nn.GELU(),
            )
        self.layers2 = nn.Sequential(
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=4*self.padding,dilation=4,bias=bias),
            nn.GELU(),
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=self.padding,bias=bias),
            nn.GELU(),
            )
        self.layers3 = nn.Sequential(
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=2*self.padding,dilation=2,bias=bias),
            nn.GELU(),
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=4*self.padding,dilation=4,bias=bias),
            nn.GELU(),
            )
        self.layers4 = nn.Sequential(
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=self.padding,bias=bias),
            nn.GELU(),
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim/2),self.kernel,padding=2*self.padding,dilation=2,bias=bias),
            nn.GELU(),
            )
        self.layers5 = nn.Sequential(
            nn.Conv1d(int(self.hidden_dim/2),int(self.hidden_dim),self.kernel,padding=4*self.padding,dilation=4,bias=bias),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim,self.hidden_dim,self.kernel,padding=self.padding,bias=bias),
            nn.GELU(),
            )
        self.layersFin = nn.Sequential(
            nn.Linear(self.hidden_dim*self.output_len , 4*self.output_len),
            nn.GELU(),
            nn.Linear(4*self.output_len , self.output_len),
            nn.GELU(),
            nn.Linear(self.output_len , self.output_len)
            )
            
        self.layer_norm = nn.LayerNorm(self.output_len, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self,noisy_enc,mask,trg_seq,device):
        input1 = noisy_enc.unsqueeze(1)
        
        input2 = self.layers1(input1)
        
        residual2 = input2
        input3 = self.layers2(input2) + residual2

        residual3 = input3
        input4 =  self.layers3(input3)+ residual3
        
        residual4 = input4
        input5 =  self.layers4(input4) + residual4
        
        residual5 = input5
        input6 =  self.layers5(input5)
        
        
        output = self.layer_norm(self.dropout(self.layersFin(torch.flatten(input6,start_dim=1))))
        #print(output.size())
        logits = output.squeeze().unsqueeze(-1)
        decoded_msg_bits = logits.sign()
        output = torch.sigmoid(logits)
        output = torch.cat((1-output,output),-1)
        out_mask = mask
        return output,decoded_msg_bits,out_mask,logits,input4 # [b_size,block_len,2]

    def decode(self,noisy_enc,info_positions,mask,device,trg_seq=None):
        _,decoded_msg_bits,out_mask,_,_ = self.forward(noisy_enc,mask,trg_seq,device)
        #decoded_msg_bits = (decoded_msg_bits==1).long()
        return decoded_msg_bits,out_mask

