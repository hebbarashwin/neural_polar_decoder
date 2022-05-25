from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import pickle
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import csv

from utils import snr_db2sigma, errors_ber, errors_bler, errors_bitwise_ber, log_sum_exp, moving_average, get_epos, get_minD, get_pairwiseD
from pac_code import *
from polar import *

import math
import random
import numpy as np
from tqdm import tqdm
from collections import namedtuple, Counter
import sys

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='PAC codes')

    parser.add_argument('--id', type=str, default=None, help='ID: optional, to run multiple runs of same hyperparameters') #Will make a folder like init_932 , etc.

    parser.add_argument('--N', type=int, default=32)#, choices=[4, 8, 16, 32, 64, 128], help='Polar code parameter N')

    parser.add_argument('--K', type=int, default=12)#, choices= [3, 4, 8,  16, 32, 64], help='Polar code parameter K')

    parser.add_argument('--target_K', type=int, default=None)#, choices= [3, 4, 8,  16, 32, 64], help='Polar code parameter K')

    parser.add_argument('--test', dest = 'test', default=False, action='store_true', help='Testing?')

    parser.add_argument('--code', type=str, default='PAC', choices=['PAC', 'Polar'], help='PAC or Polar?')

    parser.add_argument('--rate_profile', type=str, default='RM', choices=['RM', 'rev_RM', 'polar', 'sorted', 'sorted_last', 'rev_polar','custom', 'random'], help='PAC rate profiling')

    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--info_ind', type=int, default=63)#, choices= [3, 4, 8,  16, 32, 64], help='Polar code parameter K')

    parser.add_argument('--rnn_type', type=str, default='GRU', choices=['GRU', 'LSTM'], help='RNN method')

    parser.add_argument("--bidirectional", type=str2bool, nargs='?', const=True, default=False, help="Bidirectional RNN?")

    parser.add_argument('--decoding_type', type=str, default='y_h0', choices=['y_h0', 'y_input', 'y_h0_out'], help='RNN method')

    parser.add_argument('--target', type=str, default='gt', choices=['gt', 'llr'], help='training target')

    parser.add_argument("--onehot", type=str2bool, nargs='?', const=True, default=False, help="input one-hot?")

    parser.add_argument('--mult', type=int, default=1)#, multiplying factor to increase effective batch size

    parser.add_argument('--print_freq', type=int, default=100)#, multiplying factor to increase effective batch size

    parser.add_argument('--rnn_feature_size', type=int, default=256)#, choices=[32, 64, 128, 256, 512, 1024], help='num_iters')

    parser.add_argument('--rnn_pool_type', type=str, default='last', choices=['last', 'average'], help='How to pool hidden states??')

    parser.add_argument('--rnn_depth', type=int, default=2)#, choices=[32, 64, 128, 256, 512, 1024], help='num_iters')

    parser.add_argument('--y_depth', type=int, default=3)#, choices=[3,4,5,6], help='num_iters')

    parser.add_argument('--y_hidden_size', type=int, default=128)#, choices=[3,4,5,6], help='num_iters')

    parser.add_argument('--out_linear_depth', type=int, default=1)#, choices=[3,4,5,6], help='num_iters')

    parser.add_argument('--dropout', type=float, default=0.)#, choices=[64, 128, 256, 1024], help='number of blocks')

    parser.add_argument("--use_skip", type=str2bool, nargs='?', const=True, default=False, help="use skip connection?")

    parser.add_argument("--use_layernorm", type=str2bool, nargs='?', const=True, default=False, help="use skip connection?")

    parser.add_argument('--weight0', type=float, default=None, help='weigh loss at bit 0 ')

    parser.add_argument("--test_codes", type=str2bool, nargs='?', const=True, default=False, help="test_codes?")

    parser.add_argument("--test_bitwise", type=str2bool, nargs='?', const=True, default=False, help="test_bitwise?")

    # num_episodes = 50000
    parser.add_argument('--num_steps', type=int, default=200000)#, choices=[100, 20000, 40000], help='number of blocks')

    parser.add_argument('--batch_size', type=int, default=4096)#, choices=[64, 128, 256, 1024], help='number of blocks')

    parser.add_argument('--activation', type=str, default='selu', choices=['selu', 'relu', 'elu', 'tanh', 'sigmoid'], help='activation function')

    # TRAINING parameters
    parser.add_argument('--initialization', type=str, default='He', choices=['Dontknow', 'He', 'Xavier'], help='initialization')

    parser.add_argument('--optimizer_type', type=str, default='AdamW', choices=['Adam', 'RMS', 'AdamW'], help='optimizer type')

    parser.add_argument('--scheduler', type=str, default=None, choices=['cosine','step'], help='optimizer type')

    parser.add_argument('--loss', type=str, default='MSE', choices=['Huber', 'MSE', 'BCE'], help='loss function')

    parser.add_argument('--loss_on_all', dest = 'loss_on_all', default=False, action='store_true', help='loss on all bits or only info bits')

    parser.add_argument('--loss_only', type=int, default=None, help='loss only on x bits')

    parser.add_argument('--split_batch', dest = 'split_batch', default=False, action='store_true', help='split batch - for teacher forcing')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--lr_decay', type=int, default=None, help='learning rate decay frequency (in episodes)')

    parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='learning rate decay factor')

    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping factor')

    parser.add_argument('--no_detach', dest = 'no_detach', default=False, action='store_true', help='detach previous output during rnn training?')

    # TEACHER forcing
    # if only tfr_max is given assume no annealing
    parser.add_argument('--tfr_min', type=float, default=None, help='teacher forcing ratio minimum')

    parser.add_argument('--tfr_max', type=float, default=0., help='teacher forcing ratio maximum')

    parser.add_argument('--tfr_decay', type=float, default=10000, help='teacher forcing ratio decay parameter')

    parser.add_argument('--teacher_steps', type=int, default=-10000, help='initial number of steps to do teacher forcing only')

    # TESTING parameters
    parser.add_argument('--dec_train_snr', type=float, default=-1., help='SNR at which decoder is trained')

    parser.add_argument('--validation_snr', type=float, default=None, help='SNR at which decoder is validated')

    parser.add_argument('--testing_snr', type=float, default=None, help='SNR at which decoder is validated')

    parser.add_argument("--do_range_training", type=str2bool, nargs='?', const=True, default=False, help="Train on range of SNRs")

    parser.add_argument('--model_save_per', type=int, default=10000, help='num of episodes after which model is saved')

    parser.add_argument('--test_snr_start', type=float, default=-2., help='testing snr start')

    parser.add_argument('--test_snr_end', type=float, default=4., help='testing snr end')

    parser.add_argument('--snr_points', type=int, default=7, help='testing snr num points')

    parser.add_argument('--test_batch_size', type=int, default=10000, help='number of blocks')

    parser.add_argument('--test_size', type=int, default=100000, help='size of the batches')


    parser.add_argument('--noise_type', type=str, choices=['awgn', 'fading', 'radar', 't-dist'], default='awgn')
    parser.add_argument('--vv',type=float, default=5, help ='only for t distribution channel : degrees of freedom')
    parser.add_argument('--radar_prob',type=float, default=0.05, help ='only for radar distribution channel')
    parser.add_argument('--radar_power',type=float, default=5.0, help ='only for radar distribution channel')


    parser.add_argument('--model_iters', type=int, default=None, help='by default load final model, option to load a model of x episodes')

    parser.add_argument('--test_load_path', type=str, default=None, help='load test model given path')

    parser.add_argument('--list_size', type=int, default=None)#, choices=[100, 20000, 40000], help='number of blocks')

    parser.add_argument('--run_fano', dest = 'run_fano', default=False, action='store_true', help='run fano decoding')

    parser.add_argument('--random_test', dest = 'random_test', default=False, action='store_true', help='run test on random data (default action is to test on same samples as Fano did)')

    parser.add_argument('--save_path', type=str, default=None, help='save name')

    parser.add_argument('--progressive_path', type=str, default=None, help='save name')

    parser.add_argument('--load_path', type=str, default=None, help='load name')

    parser.add_argument("--run_dumer", type=str2bool, nargs='?', const=True, default=True, help="run dumer during test?")

    parser.add_argument("--run_ML", type=str2bool, nargs='?', const=True, default=False, help="run ML during test?")

    # parser.add_argument('-id', type=int, default=100000)
    parser.add_argument('--hard_decision', dest = 'hard_decision', default=False, action='store_true', help='polar code sc decoding hard decision?')

    parser.add_argument('--gpu', type=int, default=-2, help='gpus used for training - e.g 0,1,3')

    parser.add_argument('--anomaly', dest = 'anomaly', default=False, action='store_true', help='enable anomaly detection')

    parser.add_argument('--only_args', dest = 'only_args', default=False, action='store_true')

    parser.add_argument('--use_ynn', dest = 'use_ynn', default=False, action='store_true')

    parser.add_argument('--reverse_order', dest = 'reverse_order', default=False, action='store_true')

    parser.add_argument('--print_cust', dest = 'print_cust', default=False, action='store_true')

    parser.add_argument('--fresh', dest = 'fresh', default=False, action='store_true')

    args = parser.parse_args()

    if args.target_K is None:
        args.target_K = args.N // 2 if args.K <= args.N // 2 else args.K
    if args.N == 4:
        args.g = 7 # Convolutional coefficients are [1,1, 0, 1]
        # args.M = 2 # log N


    elif args.N == 8:
        args.g = 13 # Convolutional coefficients are [1, 0, 1, 1]
        # args.M = 3 # log N

    elif args.N == 16:
        args.g =  21    # [1, 0, 1, 0, 1]


    elif args.N == 32:
        args.g =  53    # [1, 1, 0, 1, 0, 1]

    else:
        args.g = 91


    args.M = int(math.log(args.N, 2))

    args.are_we_doing_ML = True if args.K <=0 else False
    if args.run_ML:
        args.are_we_doing_ML = True

    # if args.N == args.K:
    #     args.are_we_doing_ML = True
    # args.hard_decision = True # use hard-SC

    if args.tfr_min is None:
        args.tfr_min = args.tfr_max

    if args.decoding_type == 'y_input' and not args.use_ynn:
        args.y_depth = 0
        if not args.out_linear_depth > 1:
            args.y_hidden_size = 0
    return args


def get_onehot(actions):
    inds = (0.5 + 0.5*actions).long()
    return torch.eye(2, device = inds.device)[inds].reshape(actions.shape[0], -1)

def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class RNN_Model(nn.Module):
    def __init__(self, rnn_type, input_size, feature_size, output_size, num_rnn_layers, y_size, y_hidden_size, y_depth, activation = 'relu', dropout = 0., skip=False, out_linear_depth=1, y_output_size = None, bidirectional = False, use_layernorm = False):
        super(RNN_Model, self).__init__()

        assert rnn_type in ['GRU', 'LSTM']
        self.input_size  = input_size
        self.activation = activation
        self.feature_size = feature_size
        self.output_size = output_size
        self.skip = skip

        self.num_rnn_layers = num_rnn_layers
        self.bidirectional = bidirectional
        self.rnn = getattr(nn, rnn_type)(self.input_size, self.feature_size, self.num_rnn_layers, bidirectional = self.bidirectional, batch_first = True)
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)

        self.y_depth = y_depth
        self.y_size = y_size
        self.y_hidden_size = y_hidden_size
        self.out_linear_depth = out_linear_depth
        self.y_output_size = (int(self.bidirectional) + 1)*self.num_rnn_layers*self.feature_size if y_output_size is None else y_output_size
        if use_layernorm:
            self.layernorm = nn.LayerNorm(self.feature_size)
        else:
            self.layernorm = nn.Identity()

        #try:
        if self.y_hidden_size > 0 and self.y_depth > 0:
            self.y_linears = nn.ModuleList([nn.Linear(self.y_size, self.y_hidden_size, bias=True)])
            self.y_linears.extend([nn.Linear(self.y_hidden_size, self.y_hidden_size, bias=True) for ii in range(1, self.y_depth-1)])
            if (not hasattr(self, 'skip')) or (not self.skip):
                self.y_linears.append(nn.Linear(self.y_hidden_size, self.y_output_size, bias=True))
            else:
                self.y_linears.append(nn.Linear(self.y_hidden_size, self.y_output_size - self.y_size, bias=True))

        #except:
        #    pass
        if self.out_linear_depth == 1:
            self.linear = nn.Linear((int(self.bidirectional) + 1)*self.feature_size, self.output_size)
        else:
            layers = []
            layers.append(nn.Linear((int(self.bidirectional) + 1)*self.feature_size, self.y_hidden_size))
            for ii in range(1, self.out_linear_depth-1):
                layers.append(nn.SELU())
                layers.append(nn.Linear(self.y_hidden_size, self.y_hidden_size))
            layers.append(nn.SELU())
            layers.append(nn.Linear(self.y_hidden_size, self.output_size))
            self.linear = nn.Sequential(*layers)


    def act(self, inputs):
        if self.activation == 'tanh':
            return  F.tanh(inputs)
        elif self.activation == 'elu':
            return F.elu(inputs)
        elif self.activation == 'relu':
            return F.relu(inputs)
        elif self.activation == 'selu':
            return F.selu(inputs)
        elif self.activation == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.activation == 'linear':
            return inputs
        else:
            return inputs

    def get_h0(self, y):
        x = y.clone()
        for ii, layer in enumerate(self.y_linears):
            if ii != self.y_depth:
                x = self.act(layer(x))
            else:
                x = layer(x)
        if self.skip:
            x = torch.cat([y, x], 1)
        x = x.reshape(-1, self.feature_size, (int(self.bidirectional) + 1)*self.num_rnn_layers).permute(2, 0, 1).contiguous()

        if self.rnn_type == 'GRU':
            return x
        else:
            return (x,x)

    def get_Fy(self, y):
        Fy = y.clone()
        for ii, layer in enumerate(self.y_linears):
            if ii != self.y_depth:
                Fy = self.act(layer(Fy))
            else:
                Fy = layer(Fy)
        return Fy

    def forward(self, input, hidden, Fy=None):

        out, hidden = self.rnn(input, hidden)
        out = self.drop(out)
        out = self.layernorm(out)

        if Fy is None:
            decoded = self.linear(out)
        else:
            decoded = self.linear(torch.cat([Fy, out], -1))
        decoded = decoded.view(-1, self.output_size)
        return decoded, hidden

class RNN_decoder:
    def __init__(self, decoding_type, N, info_inds, onehot = False, reverse_order = False):
        self.decoding_type = decoding_type
        self.N = N
        self.info_inds = info_inds
        self.onehot = onehot#
        self.reverse_order = reverse_order

    def decode(self, net, train, y, gt = None, teacher_forcing_ratio = 0., loss_inds = None):

        if not self.onehot:
            onehot_fn = lambda x:x
        else:
            onehot_fn = get_onehot

        if self.reverse_order:
            iter_range = list(range(self.N-1, -1, -1))
            if gt is not None:
                gt = gt.flip(1)
        else:
            iter_range = list(range(0, self.N))

        if train: #training
            net.train()
            decoded = torch.ones(y.shape[0], self.N, device = y.device)
            if random.random() < teacher_forcing_ratio: # do teacher forcing
                assert gt is not None
                assert gt.shape[1] == self.N
                if self.decoding_type == 'y_h0':
                    hidden = net.get_h0(y)
                    for ii, jj in enumerate(iter_range): # don't assume first bit is always frozen
                        if ii == 0:
                            out, hidden = net(onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size), hidden)
                        else:
                            out, hidden = net(onehot_fn(gt[:, ii-1]).view(-1, 1, net.input_size), hidden)
                        decoded[:, ii] = out.squeeze()
                elif self.decoding_type == 'y_input':
                    if net.y_depth == 0:
                        Fy = y
                    else:
                        Fy = net.get_Fy(y)

                    hidden = torch.zeros((int(net.bidirectional) + 1)*net.num_rnn_layers, y.shape[0], net.feature_size, device = y.device)
                    if net.rnn_type == 'LSTM':
                        hidden = (hidden, hidden)
                    for ii, jj in enumerate(iter_range): # don't assume first bit is always frozen
                        if ii == 0:
                            out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size - self.N)], 2), hidden)
                        else:
                            out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(gt[:, ii-1]).view(-1, 1, net.input_size - self.N)], 2), hidden)
                        decoded[:, ii] = out.squeeze()
                elif self.decoding_type == 'y_h0_out':
                    hidden = net.get_h0(y)
                    Fy = hidden.clone().permute(1, 0, 2).contiguous().reshape(-1, 1, net.num_rnn_layers*net.feature_size)
                    for ii, jj in enumerate(iter_range): # don't assume first bit is always frozen
                        if ii == 0:
                            out, hidden = net(onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size), hidden, Fy)
                        else:
                            out, hidden = net(onehot_fn(gt[:, ii-1]).view(-1, 1, net.input_size), hidden, Fy)
                        decoded[:, ii] = out.squeeze()

            else: # student forcing
                if self.decoding_type == 'y_h0':
                    hidden = net.get_h0(y)
                    for ii, jj in enumerate(iter_range): # don't assume first bit is always frozen
                        if ii == 0:
                            out, hidden = net(onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size), hidden)
                        else:
                            if not args.no_detach:
                                out, hidden = net(onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size).detach().clone(), hidden)
                            else:
                                out, hidden = net(onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size).clone(), hidden)

                        if ii in self.info_inds:
                            decoded[:, ii] = out.squeeze()
                elif self.decoding_type == 'y_input':

                    if net.y_depth == 0:
                        Fy = y
                    else:
                        Fy = net.get_Fy(y)

                    hidden = torch.zeros((int(net.bidirectional) + 1)*net.num_rnn_layers, y.shape[0], net.feature_size, device = y.device)
                    if net.rnn_type == 'LSTM':
                        hidden = (hidden, hidden)
                    for ii, jj in enumerate(iter_range): # don't assume first bit is always frozen
                        if ii == 0:
                            out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size - self.N)], 2), hidden)
                        else:
                            if not args.no_detach:
                                out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size - self.N).detach().clone()], 2), hidden)
                            else:
                                out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size - self.N).clone()], 2), hidden)
                        if ii in self.info_inds:
                            decoded[:, ii] = out.squeeze()
                elif self.decoding_type == 'y_h0_out':
                    hidden = net.get_h0(y)
                    Fy = hidden.clone().permute(1, 0, 2).contiguous().reshape(-1, 1, net.num_rnn_layers*net.feature_size)
                    for ii, jj in enumerate(iter_range): # don't assume first bit is always frozen
                        if ii == 0:
                            out, hidden = net(onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size), hidden, Fy)
                        else:
                            if not args.no_detach:
                                out, hidden = net(onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size).detach().clone().sign(), hidden, Fy)
                            else:
                                out, hidden = net(onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size).clone().sign(), hidden, Fy)
                        # out, hidden = net(decoded[:, ii-1].view(-1, 1, 1).clone(), hidden)
                        if ii in self.info_inds:
                            decoded[:, ii] = out.squeeze()
            if not self.reverse_order:
                return decoded
            else:
                return decoded.flip(1)

        else: #test
            net.eval()
            if loss_inds is None:
                loss_inds = self.info_inds
            with torch.no_grad():
                if gt is None:
                    decoded = torch.ones(y.shape[0], self.N, device = y.device)
                else:
                    decoded = gt.clone()
                if self.decoding_type == 'y_h0':
                    hidden = net.get_h0(y)
                    for ii, jj in enumerate(iter_range): # don't assume first bit is always frozen
                        if ii == 0:
                            out, hidden = net(onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size),  hidden)
                        else:
                            out, hidden = net(onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size), hidden)
                        if jj in loss_inds:
                            decoded[:, ii] = out.squeeze().sign()
                elif self.decoding_type == 'y_input':
                    if net.y_depth == 0:
                        Fy = y
                    else:
                        Fy = net.get_Fy(y)

                    hidden = torch.zeros((int(net.bidirectional) + 1)*net.num_rnn_layers, y.shape[0], net.feature_size, device = y.device)
                    if net.rnn_type == 'LSTM':
                        hidden = (hidden, hidden)
                    for ii, jj in enumerate(iter_range): # don't assume first bit is always frozen
                        if ii == 0:
                            out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size - self.N)], 2), hidden)
                        else:
                            out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size - self.N).detach().clone()], 2), hidden)
                        if jj in loss_inds:
                            decoded[:, ii] = out.squeeze().sign()
                elif self.decoding_type == 'y_h0_out':
                    hidden = net.get_h0(y)
                    Fy = hidden.clone().permute(1, 0, 2).contiguous().reshape(-1, 1, net.num_rnn_layers*net.feature_size)
                    for ii, jj in enumerate(iter_range): # don't assume first bit is always frozen
                        if ii == 0:
                            out, hidden = net(onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size),  hidden, Fy)
                        else:
                            out, hidden = net(onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size), hidden, Fy)
                        if jj in loss_inds:
                            decoded[:, ii] = out.squeeze().sign()
            if not self.reverse_order:
                return decoded
            else:
                return decoded.flip(1)

    def pruneLists(self, hidden_list, decoded_list, metric_list, L):
        _, inds = torch.topk(-1*metric_list, L, 0) # select L gratest indices in every row
        sorted_inds, _ = torch.sort(inds, 0)
        batch_size = decoded_list.shape[1]

        # llr_array_list = torch.gather(llr_array_list, 0, sorted_inds.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, llr_array_list.shape[2], llr_array_list.shape[3]))
        # partial_llrs_list = torch.gather(partial_llrs_list, 0, sorted_inds.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, partial_llrs_list.shape[2], partial_llrs_list.shape[3]))
        # metric_list = torch.gather(metric_list, 0, sorted_inds)
        # u_hat_list = torch.gather(u_hat_list, 0, sorted_inds.unsqueeze(-1).repeat(1, 1, u_hat_list.shape[2]))
        hshape = hidden_list.shape
        h_list = hidden_list.permute(0, 2, 1, 3)
        hidden_list = h_list[sorted_inds, torch.arange(batch_size)].permute(0, 2, 1, 3)

        metric_list = metric_list[sorted_inds, torch.arange(batch_size)]
        decoded_list = decoded_list[sorted_inds, torch.arange(batch_size)]
        return hidden_list.contiguous(), decoded_list, metric_list

    def list_decode(self, net, y, code, L = 1):

        if not self.onehot:
            onehot_fn = lambda x:x
        else:
            onehot_fn = get_onehot
        loss_inds = self.info_inds
        batch_size = y.shape[0]

        net.eval()
        with torch.no_grad():
            decoded = torch.ones(y.shape[0], self.N, device = y.device)
            if self.decoding_type == 'y_h0':
                hidden = net.get_h0(y)
                for ii in range(0, self.N): # don't assume first bit is always frozen

                    if ii in loss_inds:
                        decoded[:, ii] = out.squeeze().sign()
            elif self.decoding_type == 'y_input':
                if net.y_depth == 0:
                    Fy = y
                else:
                    Fy = net.get_Fy(y)

            elif self.decoding_type == 'y_h0_out':
                hidden = net.get_h0(y)
                Fy = hidden.clone().permute(1, 0, 2).contiguous().reshape(-1, 1, net.num_rnn_layers*net.feature_size)

            hidden = torch.zeros(net.num_rnn_layers, y.shape[0], net.feature_size, device = y.device)
            store_device = y.device #torch.device('cpu')
            hidden_list = hidden.unsqueeze(0).cpu()
            decoded_list = decoded.unsqueeze(0).cpu()
            metric_list = torch.zeros(1, y.shape[0]).cpu()

            for ii in range(self.N): # don't assume first bit is always frozen
                list_size = hidden_list.shape[0]
                if ii in self.info_inds:
                    metric_list = torch.vstack([metric_list, metric_list])
                    decoded_list = torch.vstack([decoded_list, decoded_list])
                    hidden_list = torch.vstack([hidden_list, hidden_list])

                for list_index in range(list_size):

                    if self.decoding_type  == 'y_h0':
                        if ii == 0:
                            out, hidden = net(onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size),  hidden)
                        else:
                            out, hidden = net(onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size), hidden)
                    elif self.decoding_type == 'y_input':
                        if ii == 0:
                            out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size - self.N)], 2), hidden_list[list_index].to(y.device))
                        else:
                            out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(decoded_list[list_index, :, ii-1].to(y.device).sign()).view(-1, 1, net.input_size - self.N).detach().clone()], 2), hidden_list[list_index].to(y.device))
                    elif self.decoding_type == 'y_h0_out':
                        if ii == 0:
                            out, hidden = net(onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size),  hidden, Fy)
                        else:
                            out, hidden = net(onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size), hidden, Fy)
                    if ii in self.info_inds: # not frozen
                        decoded_list[list_index, :, ii] = out.squeeze().sign()
                        decoded_list[list_size+list_index, :, ii] = -1*out.squeeze().sign()

                        metric = torch.abs(out).cpu()
                        metric_list[list_size+list_index, :] += metric.squeeze()

                        hidden_list[list_index] = hidden
                        hidden_list[list_size+list_index] = hidden
                    else: # frozen
                        # decoded_list[list_index, :, ii] is already ones
                        # metric = torch.abs(out.cpu())*(out.sign().cpu() != 1*torch.ones_like(out).cpu()).float().cpu()
                        # metric_list[list_index, :] += metric.squeeze()
                        hidden_list[list_index] = hidden
                if ii in self.info_inds:
                    if hidden_list.shape[0] > L:
                        hidden_list, decoded_list, metric_list = self.pruneLists(hidden_list, decoded_list, metric_list, L)


            list_size = hidden_list.shape[0]
            decoded = decoded_list[:, :, self.info_inds].detach().cpu()
            codeword_list = code.encode_plotkin(decoded.reshape(-1, code.K)).reshape(list_size, batch_size, self.N)
            inds = ((codeword_list - y.cpu().unsqueeze(0))**2).sum(2).argmin(0)
            # get ML decision for each sample.
            decoded = decoded[inds, torch.arange(batch_size)]

            return decoded

def PAC_MAP_decode(noisy_codes, b_codebook):

    b_noisy = noisy_codes.unsqueeze(1).repeat(1, 2**args.K, 1)

    diff = (b_noisy - b_codebook).pow(2).sum(dim=2)

    idx = diff.argmin(dim=1)

    MAP_decoded_bits = all_message_bits[idx, :]

    return MAP_decoded_bits


def test_RNN_and_Dumer_batch(net, msg_bits, corrupted_codewords, snr, run_dumer=True):

    state = corrupted_codewords

    start =time.time()
    decoded_vhat = decoder.decode(net, False, corrupted_codewords)
    decoded_msg_bits = decoded_vhat[:, code.info_inds].sign()
    print('RNN : {}'.format(time.time() - start))
    ber_RNN = errors_ber(msg_bits, decoded_msg_bits).item()
    bler_RNN = errors_bler(msg_bits, decoded_msg_bits).item()

    if run_dumer:
        start = time.time()
        _, decoded_Dumer_msg_bits, _ = code.pac_sc_decode(corrupted_codewords, snr)
        print('SC : {}'.format(time.time() - start))

        ber_Dumer = errors_ber(msg_bits, decoded_Dumer_msg_bits.sign()).item()
        bler_Dumer = errors_bler(msg_bits, decoded_Dumer_msg_bits.sign()).item()
    else:
        ber_Dumer = 0.
        bler_Dumer = 0.

    if args.are_we_doing_ML:
        MAP_decoded_bits = PAC_MAP_decode(corrupted_codewords.detach().cpu(), b_codebook)

        ber_ML = errors_ber(msg_bits.cpu(), MAP_decoded_bits).item()
        bler_ML = errors_bler(msg_bits.cpu(), MAP_decoded_bits).item()

        return ber_RNN, bler_RNN, ber_Dumer, bler_Dumer, ber_ML, bler_ML

    else:

        return ber_RNN, bler_RNN, ber_Dumer, bler_Dumer

def test_fano(msg_bits, noisy_code, snr):

    msg_bits = msg_bits.to('cpu') # run fano on cpu. required?
    sigma = snr_db2sigma(snr)
    noisy_code = noisy_code.to('cpu')
    llrs = (2/sigma**2)*noisy_code

    decoded_bits = torch.empty_like(msg_bits)
    for ii, vv in enumerate(llrs):
        v_hat, pm = pac.fano_decode(vv.unsqueeze(0), delta = 2, verbose = 0, maxDiversions = 1000, bias_type = 'p_e')
        decoded_bits[ii] = pac.extract(v_hat)

    ber_fano = errors_ber(msg_bits, decoded_bits).item()
    bler_fano = errors_bler(msg_bits, decoded_bits).item()

    return ber_fano, bler_fano

def test_full_data(net, code, snr_range, Test_Data_Generator, run_fano = False, run_dumer = True):

    num_test_batches = len(Test_Data_Generator)

    bers_RNN_test = [0. for ii in snr_range]
    blers_RNN_test = [0. for ii in snr_range]

    bers_Dumer_test = [0. for ii in snr_range]
    blers_Dumer_test = [0. for ii in snr_range]

    bers_ML_test = [0. for ii in snr_range]
    blers_ML_test = [0. for ii in snr_range]

    bers_fano_test = [0. for ii in snr_range]
    blers_fano_test = [0. for ii in snr_range]

    for (k, msg_bits) in enumerate(Test_Data_Generator):

        msg_bits = msg_bits.to(device)
        pac_code = code.encode(msg_bits)

        for snr_ind, snr in enumerate(snr_range):
            noisy_code = code.channel(pac_code, snr, args.noise_type, args.vv, args.radar_power, args.radar_prob)

            if args.are_we_doing_ML:

                ber_RNN, bler_RNN, ber_Dumer, bler_Dumer, ber_ML, bler_ML  = test_RNN_and_Dumer_batch(net, msg_bits, noisy_code, snr)
            else:

                ber_RNN, bler_RNN, ber_Dumer, bler_Dumer  = test_RNN_and_Dumer_batch(net, msg_bits, noisy_code, snr, run_dumer)

            bers_RNN_test[snr_ind] += ber_RNN/num_test_batches
            bers_Dumer_test[snr_ind] += ber_Dumer/num_test_batches

            blers_RNN_test[snr_ind] += bler_RNN/num_test_batches
            blers_Dumer_test[snr_ind] += bler_Dumer/num_test_batches

            if args.are_we_doing_ML:
                bers_ML_test[snr_ind] += ber_ML/num_test_batches
                blers_ML_test[snr_ind] += bler_ML/num_test_batches

            if run_fano:
                ber_fano, bler_fano = test_fano(msg_bits, noisy_code, snr)
                bers_fano_test[snr_ind] += ber_fano/num_test_batches
                blers_fano_test[snr_ind] += bler_fano/num_test_batches

    return bers_RNN_test, blers_RNN_test, bers_Dumer_test, blers_Dumer_test, bers_ML_test, blers_ML_test, bers_fano_test, blers_fano_test

def test_standard(net, msg_bits, received, run_fano = False, run_dumer = True):

    snr_range = list(received.keys())
    bers_RNN_test = [0. for ii in snr_range]
    blers_RNN_test = [0. for ii in snr_range]

    bers_Dumer_test = [0. for ii in snr_range]
    blers_Dumer_test = [0. for ii in snr_range]

    bers_ML_test = [0. for ii in snr_range]
    blers_ML_test = [0. for ii in snr_range]

    bers_fano_test = [0. for ii in snr_range]
    blers_fano_test = [0. for ii in snr_range]

    msg_bits = msg_bits.to(device)
    for snr_ind, (snr, noisy_code) in enumerate(received.items()):
        noisy_code = noisy_code.to(device)
        if args.are_we_doing_ML:

            ber_RNN, bler_RNN, ber_Dumer, bler_Dumer, ber_ML, bler_ML  = test_RNN_and_Dumer_batch(net, msg_bits, noisy_code, snr)

        else:

            ber_RNN, bler_RNN, ber_Dumer, bler_Dumer  = test_RNN_and_Dumer_batch(net, msg_bits, noisy_code, snr, run_dumer)

        bers_RNN_test[snr_ind] += ber_RNN
        bers_Dumer_test[snr_ind] += ber_Dumer

        blers_RNN_test[snr_ind] += bler_RNN
        blers_Dumer_test[snr_ind] += bler_Dumer

        if args.are_we_doing_ML:
            bers_ML_test[snr_ind] += ber_ML
            blers_ML_test[snr_ind] += bler_ML

        if run_fano:
            ber_fano, bler_fano = test_fano(msg_bits, noisy_code, snr)
            bers_fano_test[snr_ind] += ber_fano
            blers_fano_test[snr_ind] += bler_fano

    return bers_RNN_test, blers_RNN_test, bers_Dumer_test, blers_Dumer_test, bers_ML_test, blers_ML_test, bers_fano_test, blers_fano_test

def polar_RNN_full_test(net, polar, snr_range, Test_Data_Generator, run_ML=False, run_SCL = False, run_RNNL = False):

    num_test_batches = len(Test_Data_Generator)

    bers_RNN_test = [0. for ii in snr_range]
    blers_RNN_test = [0. for ii in snr_range]

    bers_RNNL_test = [0. for ii in snr_range]
    blers_RNNL_test = [0. for ii in snr_range]

    bers_SC_test = [0. for ii in snr_range]
    blers_SC_test = [0. for ii in snr_range]

    bers_SCL_test = [0. for ii in snr_range]
    blers_SCL_test = [0. for ii in snr_range]

    bers_ML_test = [0. for ii in snr_range]
    blers_ML_test = [0. for ii in snr_range]

    for (k, msg_bits) in enumerate(Test_Data_Generator):

        msg_bits = msg_bits.to(device)
        polar_code = polar.encode_plotkin(msg_bits)
        gt = torch.ones(msg_bits.shape[0], args.N, device = msg_bits.device)
        gt[:, code.info_inds] = msg_bits
        for snr_ind, snr in enumerate(snr_range):
            noisy_code = polar.channel(polar_code, snr, args.noise_type, args.vv, args.radar_power, args.radar_prob)
            noise = noisy_code - polar_code

            if args.loss_only is None:

                # start = time.time()
                SC_llrs, decoded_SC_msg_bits = polar.sc_decode_new(noisy_code, snr)
                # print('SC : {}'.format(time.time() - start))
                ber_SC = errors_ber(msg_bits, decoded_SC_msg_bits.sign()).item()
                bler_SC = errors_bler(msg_bits, decoded_SC_msg_bits.sign()).item()
                if run_SCL:
                    SCL_llrs, decoded_SCL_msg_bits = polar.scl_decode(noisy_code.cpu(), snr, args.list_size, False)
                    SCL_llrs, decoded_SCL_msg_bits = SCL_llrs.to(msg_bits.device), decoded_SCL_msg_bits.to(msg_bits.device)
                    ber_SCL = errors_ber(msg_bits, decoded_SCL_msg_bits.sign()).item()
                    bler_SCL = errors_bler(msg_bits, decoded_SCL_msg_bits.sign()).item()
            else:
                SC_llrs, decoded_SC_msg_bits = polar.sc_decode_new(noisy_code, snr, gt)
                ber_SC = errors_ber(msg_bits[:, polar.msg_indices], decoded_SC_msg_bits[:, polar.msg_indices].sign()).item()
                bler_SC = errors_bler(msg_bits[:, polar.msg_indices], decoded_SC_msg_bits[:, polar.msg_indices].sign()).item()

                if run_SCL:
                    SCL_llrs, decoded_SCL_msg_bits = polar.scl_decode(noisy_code, snr, args.list_size, False)
                    ber_SCL = errors_ber(msg_bits[:, polar.msg_indices], decoded_SCL_msg_bits[:, polar.msg_indices].sign()).item()
                    bler_SCL = errors_bler(msg_bits[:, polar.msg_indices], decoded_SCL_msg_bits[:, polar.msg_indices].sign()).item()

            if args.loss_only is None:
                # start = time.time()
                decoded_bits = decoder.decode(net, False, noisy_code)
                decoded_RNN_msg_bits = decoded_bits[:, polar.info_positions].sign()
                # print('RNN : {}'.format(time.time() - start))

                ber_RNN = errors_ber(msg_bits, decoded_RNN_msg_bits.sign()).item()
                bler_RNN = errors_bler(msg_bits, decoded_RNN_msg_bits.sign()).item()

                if run_RNNL:
                    decoded_RNNL_msg_bits = decoder.list_decode(net, noisy_code, polar, args.list_size)

                    ber_RNNL = errors_ber(msg_bits.cpu(), decoded_RNNL_msg_bits.sign()).item()
                    bler_RNNL = errors_bler(msg_bits.cpu(), decoded_RNNL_msg_bits.sign()).item()
            else:
                decoded_bits = decoder.decode(net, False, noisy_code, gt, loss_inds=code.loss_inds)
                decoded_RNN_msg_bits = decoded_bits[:, polar.info_positions].sign()

                ber_RNN = errors_ber(msg_bits[:, polar.msg_indices], decoded_RNN_msg_bits[:, polar.msg_indices].sign()).item()
                bler_RNN = errors_bler(msg_bits[:, polar.msg_indices], decoded_RNN_msg_bits[:, polar.msg_indices].sign()).item()
            if run_ML:
                if args.loss_only is not None:
                    all_msg_bits = []

                    for i in range(2**args.loss_only):
                        d = dec2bitarray(i, args.loss_only)
                        all_msg_bits.append(d)
                    all_message_bits = torch.from_numpy(np.array(all_msg_bits))
                    all_message_bits = 1 - 2*all_message_bits.float()

                    decoded = torch.zeros(noisy_code.shape[0], args.K)
                    for jj in range(noisy_code.shape[0]):
                        #if jj%100 == 0:
                        #    print(jj)
                        test_msg_bits = msg_bits[jj:jj+1].repeat(all_message_bits.shape[0], 1).cpu()
                        test_msg_bits[:, code.msg_indices] = all_message_bits

                        codebook = 0.5*code.encode(test_msg_bits)+0.5
                        b_codebook = codebook.unsqueeze(0)
                        b_noisy = noisy_code[jj].cpu()
                        diff = (b_noisy - b_codebook).pow(2).sum(dim=2)
                        idx1 = diff.argmin(dim=1)
                        decoded[jj] = test_msg_bits[idx1, :]
                    ber_ML = errors_ber(msg_bits[:, polar.msg_indices].to(decoded.device), decoded[:, polar.msg_indices].sign()).item()
                    bler_ML = errors_bler(msg_bits[:, polar.msg_indices].to(decoded.device), decoded[:, polar.msg_indices].sign()).item()
                else:
                    all_msg_bits = []

                    for i in range(2**args.K):
                        d = dec2bitarray(i, args.K)
                        all_msg_bits.append(d)
                    all_message_bits = torch.from_numpy(np.array(all_msg_bits))
                    all_message_bits = 1 - 2*all_message_bits.float()
                    codebook = code.encode(all_message_bits)
                    b_codebook = codebook.unsqueeze(0)
                    if args.N != args.K:

                        if args.K > 10:
                            idx = np.zeros(noisy_code.shape[0])
                            for jj in range(noisy_code.shape[0]):
                                b_noisy = noisy_code[jj].cpu()
                                diff = (b_noisy - b_codebook).pow(2).sum(dim=2)
                                idx1 = diff.argmin(dim=1)
                                idx[jj] = idx1
                            decoded = all_message_bits[idx, :]
                        else:
                            b_noisy = noisy_code.cpu().unsqueeze(1).repeat(1, 2**args.K, 1)
                            diff = (b_noisy - b_codebook).pow(2).sum(dim=2)
                            idx = diff.argmin(dim=1)
                            decoded = all_message_bits[idx, :]
                    elif args.N == args.K:
                        decoded_codeword = noisy_code.sign()
                        decoded = polar.encode_plotkin(decoded_codeword)
                    ber_ML = errors_ber(msg_bits[:, polar.msg_indices].to(decoded.device), decoded[:, polar.msg_indices].sign()).item()
                    bler_ML = errors_bler(msg_bits[:, polar.msg_indices].to(decoded.device), decoded[:, polar.msg_indices].sign()).item()
                    bers_ML_test[snr_ind] += ber_ML/num_test_batches
                    blers_ML_test[snr_ind] += bler_ML/num_test_batches

            bers_RNN_test[snr_ind] += ber_RNN/num_test_batches
            bers_SC_test[snr_ind] += ber_SC/num_test_batches


            blers_RNN_test[snr_ind] += bler_RNN/num_test_batches
            blers_SC_test[snr_ind] += bler_SC/num_test_batches

            if run_SCL:
                bers_SCL_test[snr_ind] += ber_SCL/num_test_batches
                blers_SCL_test[snr_ind] += bler_SCL/num_test_batches

            if run_RNNL:
                bers_RNNL_test[snr_ind] += ber_RNNL/num_test_batches
                blers_RNNL_test[snr_ind] += bler_RNNL/num_test_batches

            # print(ber_SC, ber_ML)
    return bers_RNN_test, blers_RNN_test, bers_SC_test, blers_SC_test, bers_SCL_test, blers_SCL_test, bers_RNNL_test, blers_RNNL_test, bers_ML_test, blers_ML_test



def test_model(net, code, snr, bitwise = False, tf = False, data = None):
    if data is not None:
        msg_bits, noisy_code = data
        msg_bits, noisy_code = msg_bits.to(device), noisy_code.to(device)
        encoded = code.encode(msg_bits)
    else:
        msg_bits = 1 - 2 * (torch.rand(args.test_batch_size, code.K, device=device) < 0.5).float()
        encoded = code.encode(msg_bits)
        noisy_code = code.channel(encoded, snr, args.noise_type, args.vv, args.radar_power, args.radar_prob)
    gt = torch.ones(msg_bits.shape[0], code.N, device = msg_bits.device)
    gt[:, code.info_inds] = msg_bits

    if tf:
        with torch.no_grad():
            decoded_bits = decoder.decode(net, True, noisy_code, gt, 1)
    else:
        decoded_bits = decoder.decode(net, False, noisy_code)
    decoded_RNN_msg_bits = decoded_bits[:, code.info_inds].sign()

    ber_RNN = errors_ber(msg_bits, decoded_RNN_msg_bits.sign()).item()
    bler_RNN = errors_bler(msg_bits, decoded_RNN_msg_bits.sign()).item()

    if not bitwise:
        return ber_RNN, bler_RNN
    else:
        ber_bitwise = errors_bitwise_ber(msg_bits, decoded_RNN_msg_bits.sign())
        return ber_RNN, bler_RNN, ber_bitwise

def test_SC(code, snr, bitwise = False):
    msg_bits = 1 - 2 * (torch.rand(args.test_batch_size, code.K, device=device) < 0.5).float()
    encoded = code.encode(msg_bits)
    noisy_code = code.channel(encoded, snr, args.noise_type, args.vv, args.radar_power, args.radar_prob)

    _, decoded_bits = code.sc_decode_new(noisy_code, snr)
    #decoded_RNN_msg_bits = decoded_bits[:, code.info_positions].sign()

    ber_RNN = errors_ber(msg_bits, decoded_bits.sign()).item()
    bler_RNN = errors_bler(msg_bits, decoded_bits.sign()).item()

    if not bitwise:
        return ber_RNN, bler_RNN
    else:
        ber_bitwise = errors_bitwise_ber(msg_bits, decoded_bits.sign())
        return ber_RNN, bler_RNN, ber_bitwise

def get_code(code_type, rate_profile, N, K, g=None):
    n = int(np.log2(N))
    ### Encoder
    if code_type == 'PAC':
        code = PAC(args, N, K, g, rate_profile = rate_profile)
        pac = code
        frozen_levels = (code.rate_profiler(-torch.ones(1, K), scheme = rate_profile) == 1.)[0].numpy()

        # print("Current frozen levels are:{0}".format(frozen_levels))

        code.info_inds = code.B

        # print("Current information indices are:{0}".format(info_inds))

        code.frozen_inds = np.array(list(set(np.arange(N))^set(code.B)))
        code.encode = code.pac_encode
        code.rate_profile = rate_profile

    elif code_type == 'Polar':
        if rate_profile == 'polar':
            # computed for SNR = 0
            if n == 5:
                rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

            elif n == 4:
                rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
            elif n == 3:
                rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
            elif n == 2:
                rs = np.array([3, 2, 1, 0])

            rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1
            rs = rs[rs<N]

            ###############
            ### Polar code
            ##############

            ### Encoder

            code = PolarCode(n, K, args, rs=rs)
        elif rate_profile == 'RM':
            rmweight = np.array([countSetBits(i) for i in range(N)])
            Fr = np.argsort(rmweight)[:-K]
            Fr.sort()
            code = PolarCode(n, K, args, F=Fr)

        elif rate_profile == 'rev_RM':
            rmweight = np.array([countSetBits(i) for i in range(N)])
            wts = np.argsort(rmweight)
            Fr = np.concatenate([wts[:-args.target_K], wts[N-args.target_K+K:]])
            Fr.sort()
            code = PolarCode(n, K, args, F=Fr)

        elif rate_profile == 'custom':
            Fr = np.array(list(set(np.arange(N))^set([args.info_ind])))
            Fr.sort()
            code = PolarCode(n, K, args, F=Fr)
            print("One bit info: bit {}".format(code.info_positions))

        elif rate_profile == 'sorted':
            if n == 5:
                rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

            elif n == 4:
                rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
            elif n == 3:
                rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
            elif n == 2:
                rs = np.array([3, 2, 1, 0])

            rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

            rs = rs[rs<N]
            first_inds = rs[:args.target_K].copy()
            first_inds.sort()
            rs[:args.target_K] = first_inds
            code = PolarCode(n, K, args, rs=rs)

        elif rate_profile == 'sorted_last':
            if n == 5:
                rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

            elif n == 4:
                rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
            elif n == 3:
                rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
            elif n == 2:
                rs = np.array([3, 2, 1, 0])

            rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

            rs = rs[rs<N]
            first_inds = rs[:args.target_K].copy()
            first_inds.sort()
            rs[:args.target_K] = first_inds[::-1]
            code = PolarCode(n, K, args, rs=rs)

        elif rate_profile == 'rev_polar':
            if n == 5:
                rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

            elif n == 4:
                rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
            elif n == 3:
                rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
            elif n == 2:
                rs = np.array([3, 2, 1, 0])

            rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

            rs = rs[rs<N]
            first_inds = rs[:args.target_K].copy()
            rs[:args.target_K] = first_inds[::-1]
            code = PolarCode(n, K, args, rs=rs)

            if n == 5:
                rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

            elif n == 4:
                rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
            elif n == 3:
                rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
            elif n == 2:
                rs = np.array([3, 2, 1, 0])

            rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

            rs = rs[rs<N]
            first_inds = rs[:args.target_K].copy()
            rs[:args.target_K] = first_inds[::-1]
            code = PolarCode(n, K, args, rs=rs)

        elif rate_profile == 'random':
            if n == 5:
                rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

            elif n == 4:
                rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
            elif n == 3:
                rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
            elif n == 2:
                rs = np.array([3, 2, 1, 0])

            rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

            rs = rs[rs<N]
            first_inds = rs[:args.target_K].copy()
            rs[:args.target_K] = first_inds[::-1]
            code = PolarCode(n, K, args, rs=rs)

            if n == 5:
                rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

            elif n == 4:
                rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
            elif n == 3:
                rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
            elif n == 2:
                rs = np.array([3, 2, 1, 0])

            rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

            rs = rs[rs<N]
            first_inds = rs[:args.target_K].copy()
            rs[:args.target_K] = np.random.RandomState(seed=args.random_seed).permutation(first_inds)
            code = PolarCode(n, K, args, rs=rs)

        polar = code
        code.info_inds = code.info_positions
        code.frozen_inds = code.frozen_positions
        code.rate_profile = rate_profile
        code.encode = code.encode_plotkin

        if args.loss_only is not None:
            code.loss_inds = rs[:args.loss_only].copy()
            code.loss_inds.sort()
            code.msg_indices = np.where(np.in1d(code.info_inds, code.loss_inds))[0]
        else:
            code.msg_indices = np.arange(args.K)

    return code

if __name__ == '__main__':

    args = get_args()
    if torch.cuda.is_available() and args.gpu != -1:
        if args.gpu == -2:
            device = torch.device("cuda")
        else:
            device = torch.device("cuda:{0}".format(args.gpu))
    else:
        device = torch.device("cpu")

    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)

    # torch.manual_seed(37)
    kwargs = {'num_workers': 4, 'pin_memory': False} if torch.cuda.is_available() else {}

    ID = '' if args.id is None else args.id
    lr_ = args.lr if args.scheduler is None else str(args.lr)+'_decay_{}_{}'.format(args.lr_decay, args.scheduler)
    tfr_ = 'tfr_min_{}_max_{}_decay_{}_init_{}'.format(args.tfr_max, args.tfr_min, args.tfr_decay, args.teacher_steps) if args.tfr_min != args.tfr_max else 'tfr_{}'.format(args.tfr_min)
    if args.code == 'PAC':
        g_ = '_g_{}'.format(args.g)
    else:
        g_ = ''
    if args.loss_only is None:
        lo_ = ''
    else:
        lo_ = '_lo{}'.format(args.loss_only)
    y_ = 'y_depth_{}_hsize_{}{}{}'.format(args.y_depth, args.y_hidden_size, '_out'+str(args.out_linear_depth) if args.out_linear_depth > 1 else '', '_LN' if args.use_layernorm else '')
    rnn_type = args.rnn_type if not args.bidirectional else 'Bi-'+args.rnn_type

    if args.rate_profile == 'random' and args.random_seed != 42:
        rate_profile1 = 'random{}'.format(args.random_seed)
    else:
        rate_profile1 = args.rate_profile
    if args.target_K == args.N//2:
        rate_profile = rate_profile1
    else:
        rate_profile = '{}_{}'.format(rate_profile1, args.target_K)
    if args.use_skip:
        y_ = y_ + '_skip'
    results_save_path = './Supervised_RNN_{18}_Results/{18}_{0}_{1}{19}/Scheme_{2}{3}/{4}/{5}_depth_{6}_fsize_{7}/{8}/Dec_snr_{9}_bs_{10}/{17}/Activ_{11}_Init_{12}/Optim_{13}_LR_{14}_loss_{15}/{16}'\
                                .format(args.K, args.N, rate_profile, g_, args.decoding_type if not args.onehot else args.decoding_type + '_onehot', rnn_type, args.rnn_depth, args.rnn_feature_size, y_, args.dec_train_snr, args.batch_size*args.mult, \
                                    args.activation, args.initialization, args.optimizer_type, lr_, args.loss, ID, tfr_, args.code, lo_)
    if args.save_path is None:
        final_save_path = './Supervised_RNN_{18}_Results/final_nets/Scheme_{2}/N{1}_K{0}{19}{3}_{4}_{5}_depth_{6}_fsize_{7}_{8}_snr_{9}_bs_{10}_{17}_activ_{11}_init_{12}_optim_{13}_lr_{14}_loss_{15}_{16}.pt'\
                                .format(args.K, args.N, rate_profile, g_, args.decoding_type if not args.onehot else args.decoding_type + '_onehot', rnn_type, args.rnn_depth, args.rnn_feature_size, y_, args.dec_train_snr, args.batch_size*args.mult, \
                                    args.activation, args.initialization, args.optimizer_type, lr_, args.loss, ID, tfr_, args.code, lo_)
        os.makedirs('./Supervised_RNN_{0}_Results/final_nets/Scheme_{1}'.format(args.code, rate_profile), exist_ok = True)
    else:
        final_save_path = args.save_path

    if args.progressive_path is not None:
        os.makedirs(args.progressive_path, exist_ok=True)
        save_prog = os.path.join(args.progressive_path, 'K{}'.format(args.K))
        os.makedirs(save_prog, exist_ok=True)

    ############
    ## PAC Code parameters
    ############
    K = args.K
    N = args.N
    g = args.g
    M = args.M
    n = int(np.log2(args.N))

    n_actions = 2 # either 0 or 1.

    # dec_train_snr = 8. # For training the Q network

    ###############
    ### PAC code
    ##############

    if args.validation_snr is not None:
        valid_snr = args.validation_snr
    else:
        valid_snr = args.dec_train_snr

    ### Encoder
    code = get_code(args.code, args.rate_profile, args.N, args.K, args.g)
    code_N2 = get_code(args.code, args.rate_profile, args.N, args.target_K, args.g)
    if args.test_codes:
        test_bers = {ii:[] for ii in np.arange(min(8, args.target_K//2), args.target_K +1)}
        test_blers = {ii:[] for ii in np.arange(min(8, args.target_K//2), args.target_K +1)}
        test_bers_tf = {ii:[] for ii in np.arange(min(8, args.target_K//2), args.target_K +1)}
        test_blers_tf = {ii:[] for ii in np.arange(min(8, args.target_K//2), args.target_K +1)}
        if args.testing_snr is not None:
            testing_snrs = [args.testing_snr for iii in range(len(test_bers))]
        else:
            if args.N == 64:
                testing_snrs = [2 for iii in range(len(test_bers))]
            elif args.N == 32:
                testing_snrs = [1 for iii in range(len(test_bers))]
            else:
                testing_snrs = [valid_snr for iii in range(len(test_bers))]
    if args.test_bitwise:
        bitwise_bers = []
        bitwise_bers_tf = []
    decoder = RNN_decoder(args.decoding_type, args.N, code.info_inds, args.onehot, args.reverse_order)

    std_data_path = './data/polar/test/test_N{0}_K{1}.p'.format(args.N, args.N//2)
    if os.path.exists(std_data_path) and args.target_K == args.N//2:
        datas = torch.load(std_data_path)
        snr_test = datas['snr']
        msg_bits_d = datas['msg']
        noisy_d = datas['rec']
        std_data = (msg_bits_d, noisy_d)
    else:
        std_data = None

    if args.only_args:
        print("Loaded args. Exiting")
        sys.exit()
    ##############
    ### Neural networks
    ##############

    if args.decoding_type == 'y_h0':
        net = RNN_Model(args.rnn_type, 1+int(args.onehot), args.rnn_feature_size, 1, args.rnn_depth, args.N, args.y_hidden_size, args.y_depth, args.activation, args.dropout, args.use_skip, bidirectional = args.bidirectional, use_layernorm = args.use_layernorm).to(device)
    elif args.decoding_type == 'y_input':
        if args.use_ynn:
            net = RNN_Model(args.rnn_type, args.N+1+int(args.onehot), args.rnn_feature_size, 1, args.rnn_depth, args.N, args.y_hidden_size, args.y_depth, args.activation, args.dropout, args.use_skip, y_output_size = args.N, out_linear_depth = args.out_linear_depth, bidirectional = args.bidirectional, use_layernorm = args.use_layernorm).to(device)
        else:
            net = RNN_Model(args.rnn_type, args.N+1+int(args.onehot), args.rnn_feature_size, 1, args.rnn_depth, args.N, args.y_hidden_size if args.out_linear_depth > 1 else 0, 0, args.activation, args.dropout, args.use_skip, out_linear_depth = args.out_linear_depth, bidirectional = args.bidirectional, use_layernorm = args.use_layernorm).to(device)
    elif args.decoding_type == 'y_h0_out':
        net = RNN_Model(args.rnn_type, 1+int(args.onehot), args.rnn_feature_size, 1, args.rnn_depth, args.N, args.y_hidden_size, args.y_depth, args.activation, args.dropout, args.use_skip, 3, (1+args.rnn_depth)*args.rnn_feature_size, bidirectional = args.bidirectional, use_layernorm = args.use_layernorm).to(device)

    if args.load_path is not None:
        checkpoint_train = torch.load(args.load_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint_train['net'])
        print("Pretrained model loaded")

    if not args.test:
        if not args.fresh:
            os.makedirs(results_save_path)
            os.makedirs(results_save_path +'/Models')
            print("Save path already exists! Forgot --test? Else, use --fresh flag")
        else:
            os.makedirs(results_save_path, exist_ok=True)
            os.makedirs(results_save_path +'/Models', exist_ok=True)

        ##############
        ### Optimizers
        ##############
        if args.optimizer_type == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr = args.lr)
        elif args.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(net.parameters(), lr = args.lr)
        elif args.optimizer_type == 'RMS':
            optimizer = optim.RMSprop(net.parameters(), lr = args.lr)
        elif args.optimizer_type == 'SGD':
            optimizer = optim.RMSprop(net.parameters(), lr = args.lr)
        else:
            raise Exception("Optimizer not supported yet!")

        if args.scheduler is None:
            scheduler = None
        elif args.scheduler == 'cosine':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,args.lr_decay,args.num_steps,num_cycles=args.num_steps//args.lr_decay)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay, args.lr_decay_gamma)

        if args.loss == 'Huber':
            loss_fn = F.smooth_l1_loss
        elif args.loss == 'MSE':
            loss_fn = nn.MSELoss()

        elif args.loss == 'BCE':
            if args.weight0 is None:
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        training_losses = []
        training_bers = []

        mavg_steps = 25


        print("Training ({}, {}). Need to save for: {} \n Save path: {}".format( args.K, args.N, args.model_save_per, results_save_path))
        if args.validation_snr is not None:
            valid_snr = args.validation_snr
        else:
            valid_snr = args.dec_train_snr

        range_snr = [args.dec_train_snr,args.dec_train_snr+1,args.dec_train_snr+2]
        try:
            for i_step in range(args.num_steps):  ## Each episode is like a sample now until memory size is reached.

                start_time = time.time()
                if args.do_range_training:
                    train_snr = range_snr[i_step%3]
                else:
                    train_snr = args.dec_train_snr
                msg_bits = 1 - 2 * (torch.rand(args.batch_size, args.K, device=device) < 0.5).float()
                gt = torch.ones(args.batch_size, args.N, device = device)
                gt[:, code.info_inds] = msg_bits

                encoded = code.encode(msg_bits)
                corrupted_codewords = code.channel(encoded, train_snr, args.noise_type, args.vv, args.radar_power, args.radar_prob)
                teacher_forcing_ratio = args.tfr_min + (args.tfr_max - args.tfr_min) * math.exp(-1 * (i_step - args.teacher_steps)/args.tfr_decay) if i_step > args.teacher_steps else args.tfr_max
                decoded_vhat = decoder.decode(net, True, corrupted_codewords, gt, teacher_forcing_ratio)
                decoded_msg_bits = decoded_vhat[:, code.info_inds]

                # OLD LOSS: on all bits
                if args.loss_on_all:
                    loss = loss_fn(decoded_vhat, gt)
                else:
                    if args.loss_only is None:
                        # NEW LOSS : only on info bits
                        if args.loss == 'BCE':
                            loss = loss_fn(decoded_msg_bits, 0.5+0.5*msg_bits)
                        else:
                            if args.target == 'gt':
                                loss = loss_fn(decoded_msg_bits, msg_bits)
                            elif args.target == 'llr':
                                SC_llrs, _ = code.sc_decode_new(corrupted_codewords, train_snr, gt)
                                loss = loss_fn(decoded_msg_bits, SC_llrs[:, code.info_inds])
                        ber = errors_ber(msg_bits, decoded_msg_bits.sign()).item()
                    else:
                        # loss only on last args.loss_only bits
                        if args.loss == 'BCE':
                            loss = loss_fn(decoded_msg_bits[:, code.msg_indices], 0.5+0.5*msg_bits[:, code.msg_indices])
                        else:
                            if args.target == 'gt':
                                loss = loss_fn(decoded_msg_bits[:, code.msg_indices], msg_bits[:, code.msg_indices])
                            elif args.target == 'llr':
                                SC_llrs, _ = code.sc_decode_new(corrupted_codewords, train_snr, gt)
                                loss = loss_fn(decoded_msg_bits[:, code.msg_indices], SC_llrs[:, code.info_inds][:, code.msg_indices])
                        ber = errors_ber(msg_bits[:, code.msg_indices], decoded_msg_bits[:, code.msg_indices].sign()).item()


                (loss/args.mult).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip) # gradient clipping to avoid exploding gradients

                if i_step % args.mult == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                training_losses.append(round(loss.item(),5))
                training_bers.append(round(ber, 5))
                if i_step % args.print_freq == 0:
                    ber_val, bler_val = test_model(net, code, valid_snr)
                    if args.test_bitwise:
                        _, _, ber_bitwise = test_model(net, code_N2, valid_snr, bitwise = True, data = std_data)
                        _, _, ber_bitwise_tf = test_model(net, code_N2, valid_snr, bitwise = True, tf = True, data = std_data)

                        bitwise_bers.append(ber_bitwise.detach().cpu())
                        bitwise_bers_tf.append(ber_bitwise_tf.detach().cpu())

                    print('[%d/%d] At %d dB, Loss: %.7f, BER: %.7f'
                                % (i_step, args.num_steps,  valid_snr, loss, ber_val))
                    if args.print_cust:
                        _, _, ber_bitwise_tf_100 = test_model(net, code, 100, bitwise = True, tf = True)
                        print('Bitwise 100dB: ', *ber_bitwise_tf_100.cpu().numpy(), '\n')

                    if args.test_codes:
                        for test_ii, KK in enumerate(test_bers.keys()):
                            test_code = get_code(args.code, args.rate_profile, args.N, KK, args.g)
                            ber_val, bler_val = test_model(net, test_code, testing_snrs[test_ii])
                            ber_val_tf, bler_val_tf = test_model(net, test_code, testing_snrs[test_ii], tf = True)
                            test_bers[KK].append(ber_val)
                            test_blers[KK].append(bler_val)
                            test_bers_tf[KK].append(ber_val_tf)
                            test_blers_tf[KK].append(bler_val_tf)
                if i_step == 10:
                    print("Time for one step is {0:.4f} minutes".format((time.time() - start_time)/60))

                # Save the model for safety

                if ((i_step+1) % args.model_save_per == 0) or (i_step+1) == args.num_steps:

                    # print(i_episode +1 )
                    torch.save({'net': net.state_dict(), 'step':i_step+1, 'args':args} ,\
                                    results_save_path+'/Models/model_{0}.pt'.format(i_step+1))
                    torch.save({'net': net.state_dict(), 'step':i_step+1, 'args':args} ,\
                                    results_save_path+'/Models/model_final.pt')
                    torch.save({'net': net.state_dict(), 'step':i_step+1, 'args':args} ,\
                                    final_save_path)


                    episode_x = np.arange(1, 1+len(training_losses))
                    episode_x_mavg = np.arange(1+len(training_losses)-len(moving_average(training_losses, n=mavg_steps)), 1+len(training_losses))

                    plt.figure()
                    plt.plot(episode_x, training_losses)
                    plt.plot(episode_x_mavg, moving_average(training_losses, n=mavg_steps))
                    plt.savefig(results_save_path +'/training_losses.png')
                    plt.close()

                    plt.figure()
                    plt.plot(episode_x, training_losses)
                    plt.plot(episode_x_mavg, moving_average(training_losses, n=mavg_steps))
                    plt.yscale('log')
                    plt.savefig(results_save_path +'/training_losses_log.png')
                    plt.close()

                    plt.figure()
                    plt.plot(episode_x, training_bers)
                    plt.plot(episode_x_mavg, moving_average(training_bers, n=mavg_steps))
                    plt.savefig(results_save_path +'/training_bers.png')
                    plt.close()

                    plt.figure()
                    plt.plot(episode_x, training_bers)
                    plt.plot(episode_x_mavg, moving_average(training_bers, n=mavg_steps))
                    plt.yscale('log')
                    plt.savefig(results_save_path +'/training_bers_log.png')
                    plt.close()

                    with open(os.path.join(results_save_path, 'values_training.csv'), 'w') as f:

                        # using csv.writer method from CSV package
                        write = csv.writer(f)

                        write.writerow(episode_x)
                        write.writerow(training_losses)
                        write.writerow(training_bers)


                    if args.test_codes:
                        with open(os.path.join(results_save_path, 'tested_codes.csv'), 'w') as f:

                            # using csv.writer method from CSV package
                            episode_tt = np.arange(0, len(test_bers[list(test_bers.keys())[0]])*100, 100)
                            write = csv.writer(f)

                            write.writerow(episode_tt)
                            for KK in test_bers.keys():
                                write.writerow(test_bers[KK])
                                write.writerow(test_blers[KK])
                        with open(os.path.join(results_save_path, 'tested_codes_tf.csv'), 'w') as f:

                            # using csv.writer method from CSV package
                            episode_tt = np.arange(0, len(test_bers_tf[list(test_bers.keys())[0]])*100, 100)
                            write = csv.writer(f)

                            write.writerow(episode_tt)
                            for KK in test_bers.keys():
                                write.writerow(test_bers_tf[KK])
                                write.writerow(test_blers_tf[KK])

                        if args.progressive_path is not None:
                            with open(os.path.join(save_prog, 'tested_codes.csv'), 'w') as f:

                                # using csv.writer method from CSV package
                                episode_tt = np.arange(0, len(test_bers[list(test_bers.keys())[0]])*100, 100)
                                write = csv.writer(f)

                                write.writerow(episode_tt)
                                for KK in test_bers.keys():
                                    write.writerow(test_bers[KK])
                                    write.writerow(test_blers[KK])
                            with open(os.path.join(save_prog, 'tested_codes_tf.csv'), 'w') as f:

                                # using csv.writer method from CSV package
                                episode_tt = np.arange(0, len(test_bers_tf[list(test_bers.keys())[0]])*100, 100)
                                write = csv.writer(f)

                                write.writerow(episode_tt)
                                for KK in test_bers.keys():
                                    write.writerow(test_bers_tf[KK])
                                    write.writerow(test_blers_tf[KK])
                    if args.test_bitwise:
                        with open(os.path.join(results_save_path, 'tested_bitwise.csv'), 'w') as f:

                            # using csv.writer method from CSV package
                            episode_tt = np.arange(0, len(bitwise_bers)*100, 100)
                            write = csv.writer(f)

                            write.writerow(episode_tt)
                            bers_bit = torch.stack(bitwise_bers).squeeze(2)
                            for ii in range(bers_bit.shape[1]):
                                write.writerow(bers_bit[:, ii].numpy())
                        with open(os.path.join(results_save_path, 'tested_bitwise_tf.csv'), 'w') as f:

                            # using csv.writer method from CSV package
                            episode_tt = np.arange(0, len(bitwise_bers_tf)*100, 100)
                            write = csv.writer(f)

                            write.writerow(episode_tt)
                            bers_bit = torch.stack(bitwise_bers_tf).squeeze(2)
                            for ii in range(bers_bit.shape[1]):
                                write.writerow(bers_bit[:, ii].numpy())
                        if args.progressive_path is not None:
                            with open(os.path.join(save_prog, 'tested_bitwise.csv'), 'w') as f:

                                # using csv.writer method from CSV package
                                episode_tt = np.arange(0, len(bitwise_bers)*100, 100)
                                write = csv.writer(f)

                                write.writerow(episode_tt)
                                bers_bit = torch.stack(bitwise_bers).squeeze(2)
                                for ii in range(bers_bit.shape[1]):
                                    write.writerow(bers_bit[:, ii].numpy())
                            with open(os.path.join(save_prog, 'tested_bitwise_tf.csv'), 'w') as f:

                                # using csv.writer method from CSV package
                                episode_tt = np.arange(0, len(bitwise_bers_tf)*100, 100)
                                write = csv.writer(f)

                                write.writerow(episode_tt)
                                bers_bit = torch.stack(bitwise_bers_tf).squeeze(2)
                                for ii in range(bers_bit.shape[1]):
                                    write.writerow(bers_bit[:, ii].numpy())

            print('Complete')

        except KeyboardInterrupt:

            torch.save({'net': net.state_dict(), 'step':i_step+1, 'args':args} ,\
                            results_save_path+'/Models/model_{0}.pt'.format(i_step+1))
            torch.save({'net': net.state_dict(), 'step':i_step+1, 'args':args} ,\
                            results_save_path+'/Models/model_final.pt')
            torch.save({'net': net.state_dict(), 'step':i_step+1, 'args':args} ,\
                            final_save_path)

            episode_x = np.arange(1, 1+len(training_losses))
            episode_x_mavg = np.arange(1+len(training_losses)-len(moving_average(training_losses, n=mavg_steps)), 1+len(training_losses))

            plt.figure()
            plt.plot(episode_x, training_losses)
            plt.plot(episode_x_mavg, moving_average(training_losses, n=mavg_steps))
            plt.savefig(results_save_path +'/training_losses.png')
            plt.close()

            plt.figure()
            plt.plot(episode_x, training_losses)
            plt.plot(episode_x_mavg, moving_average(training_losses, n=mavg_steps))
            plt.yscale('log')
            plt.savefig(results_save_path +'/training_losses_log.png')
            plt.close()

            plt.figure()
            plt.plot(episode_x, training_bers)
            plt.plot(episode_x_mavg, moving_average(training_bers, n=mavg_steps))
            plt.savefig(results_save_path +'/training_bers.png')
            plt.close()

            plt.figure()
            plt.plot(episode_x, training_bers)
            plt.plot(episode_x_mavg, moving_average(training_bers, n=mavg_steps))
            plt.yscale('log')
            plt.savefig(results_save_path +'/training_bers_log.png')
            plt.close()


            print("Exited and saved")

            with open(os.path.join(results_save_path, 'values_training.csv'), 'w') as f:

                # using csv.writer method from CSV package
                write = csv.writer(f)

                write.writerow(episode_x)
                write.writerow(training_losses)
                write.writerow(training_bers)

            if args.test_codes:
                with open(os.path.join(results_save_path, 'tested_codes.csv'), 'w') as f:

                    # using csv.writer method from CSV package
                    episode_tt = np.arange(0, len(test_bers[list(test_bers.keys())[0]])*100, 100)
                    write = csv.writer(f)

                    write.writerow(episode_tt)
                    for KK in test_bers.keys():
                        write.writerow(test_bers[KK])
                        write.writerow(test_blers[KK])
                with open(os.path.join(results_save_path, 'tested_codes_tf.csv'), 'w') as f:

                    # using csv.writer method from CSV package
                    episode_tt = np.arange(0, len(test_bers_tf[list(test_bers.keys())[0]])*100, 100)
                    write = csv.writer(f)

                    write.writerow(episode_tt)
                    for KK in test_bers.keys():
                        write.writerow(test_bers_tf[KK])
                        write.writerow(test_blers_tf[KK])

                if args.progressive_path is not None:
                    with open(os.path.join(save_prog, 'tested_codes.csv'), 'w') as f:

                        # using csv.writer method from CSV package
                        episode_tt = np.arange(0, len(test_bers[list(test_bers.keys())[0]])*100, 100)
                        write = csv.writer(f)

                        write.writerow(episode_tt)
                        for KK in test_bers.keys():
                            write.writerow(test_bers[KK])
                            write.writerow(test_blers[KK])
                    with open(os.path.join(save_prog, 'tested_codes_tf.csv'), 'w') as f:

                        # using csv.writer method from CSV package
                        episode_tt = np.arange(0, len(test_bers_tf[list(test_bers.keys())[0]])*100, 100)
                        write = csv.writer(f)

                        write.writerow(episode_tt)
                        for KK in test_bers.keys():
                            write.writerow(test_bers_tf[KK])
                            write.writerow(test_blers_tf[KK])
            if args.test_bitwise:
                with open(os.path.join(results_save_path, 'tested_bitwise.csv'), 'w') as f:

                    # using csv.writer method from CSV package
                    episode_tt = np.arange(0, len(bitwise_bers)*100, 100)
                    write = csv.writer(f)

                    write.writerow(episode_tt)
                    bers_bit = torch.stack(bitwise_bers).squeeze(2)
                    for ii in range(bers_bit.shape[1]):
                        write.writerow(bers_bit[:, ii].numpy())
                with open(os.path.join(results_save_path, 'tested_bitwise_tf.csv'), 'w') as f:

                    # using csv.writer method from CSV package
                    episode_tt = np.arange(0, len(bitwise_bers_tf)*100, 100)
                    write = csv.writer(f)

                    write.writerow(episode_tt)
                    bers_bit = torch.stack(bitwise_bers_tf).squeeze(2)
                    for ii in range(bers_bit.shape[1]):
                        write.writerow(bers_bit[:, ii].numpy())
                if args.progressive_path is not None:
                    with open(os.path.join(save_prog, 'tested_bitwise.csv'), 'w') as f:

                        # using csv.writer method from CSV package
                        episode_tt = np.arange(0, len(bitwise_bers)*100, 100)
                        write = csv.writer(f)

                        write.writerow(episode_tt)
                        bers_bit = torch.stack(bitwise_bers).squeeze(2)
                        for ii in range(bers_bit.shape[1]):
                            write.writerow(bers_bit[:, ii].numpy())
                    with open(os.path.join(save_prog, 'tested_bitwise_tf.csv'), 'w') as f:

                        # using csv.writer method from CSV package
                        episode_tt = np.arange(0, len(bitwise_bers_tf)*100, 100)
                        write = csv.writer(f)

                        write.writerow(episode_tt)
                        bers_bit = torch.stack(bitwise_bers_tf).squeeze(2)
                        for ii in range(bers_bit.shape[1]):
                            write.writerow(bers_bit[:, ii].numpy())

    print("TESTING :")
    times = []
    results_load_path = results_save_path

    if args.model_iters is not None:
        checkpoint1 = torch.load(results_load_path +'/Models/model_{0}.pt'.format(args.model_iters), map_location=lambda storage, loc: storage)
    elif args.test_load_path is not None:
        checkpoint1 = torch.load(args.test_load_path , map_location=lambda storage, loc: storage)
    else:
        checkpoint1 = torch.load(results_load_path +'/Models/model_final.pt', map_location=lambda storage, loc: storage)
        try:
            args.model_iters = i_step + 1
        except:
            pass
    loaded_step = checkpoint1['step']
    net.load_state_dict(checkpoint1['net'])
    net.to(device)
    print("Model loaded at step {}".format(loaded_step))


    if args.snr_points == 1 and args.test_snr_start == args.test_snr_end:
        snr_range = [args.test_snr_start]
    else:
        snrs_interval = (args.test_snr_end - args.test_snr_start)* 1.0 /  (args.snr_points-1)
        snr_range = [snrs_interval* item + args.test_snr_start for item in range(args.snr_points)]

    Test_msg_bits = 2 * (torch.rand(args.test_size, args.K) < 0.5).float() - 1

    Test_Data_Generator = torch.utils.data.DataLoader(Test_msg_bits, batch_size=args.test_batch_size , shuffle=False, **kwargs)

    num_test_batches = len(Test_Data_Generator)



    ######
    ### MAP decoding stuff
    ######

    # if args.are_we_doing_ML:

    #     all_msg_bits = []

    #     for i in range(2**args.K):

    #         d = dec2bitarray(i, args.K)
    #         all_msg_bits.append(d)

    #     all_message_bits = torch.from_numpy(np.array(all_msg_bits))
    #     all_message_bits = 1 - 2*all_message_bits.float()

    #     codebook = code.encode(all_message_bits)

    #     global b_codebook
    #     b_codebook = codebook.unsqueeze(0)#codebook.repeat(args.test_batch_size, 1, 1)

    run_fano = args.run_fano
    run_dumer = args.run_dumer

    fano_path = './data/pac/fano/Scheme_{3}/N{0}_K{1}_g{2}.p'.format(args.N, args.K, args.g, rate_profile)
    test_data_path = './data/pac/test/Scheme_{3}/test_N{0}_K{1}_g{2}.p'.format(args.N, args.K, args.g, rate_profile)
    if os.path.exists(fano_path):
        fanos = pickle.load(open(fano_path, 'rb'))
        snr_range_fano = fanos[0]
        bers_fano_test = fanos[1]
        blers_fano_test = fanos[2]
        run_fano = False
    else:
        snr_range_fano = snr_range
        bers_fano_test = []
        blers_fano_test = []


    start_time = time.time()

    if args.code == 'PAC':
        if not args.random_test:
            try:
                test_dict = torch.load(test_data_path)
                random_test = False
            except:
                random_test = True
        else:
            random_test = True
        if random_test:
            print("Testing on random data")
            bers_RNN_test, blers_RNN_test, bers_Dumer_test, blers_Dumer_test, bers_ML_test, blers_ML_test, bers_fano_temp, blers_fano_temp = test_full_data(net, code, snr_range, Test_Data_Generator, run_fano = run_fano, run_dumer = run_dumer)
        else:
            print("Testing on the standard data")
            msg_bits = test_dict['msg']
            received = test_dict['rec']
            snr_range = list(received.keys())
            print(snr_range)
            bers_RNN_test, blers_RNN_test, bers_Dumer_test, blers_Dumer_test, bers_ML_test, blers_ML_test, bers_fano_temp, blers_fano_temp = test_standard(net, msg_bits, received, run_fano = run_fano, run_dumer = run_dumer)

        if not os.path.exists(fano_path):
            bers_fano_test = bers_fano_temp
            blers_fano_test = blers_fano_temp
            snr_range_fano = snr_range

        if args.run_fano:
            if not os.path.exists(fano_path):
                os.makedirs('./data/pac/fano/Scheme_{}'.format(args.rate_profile), exist_ok=True)
                print("Saving fano error rates at: {}".format(fano_path))
                pickle.dump([snr_range, bers_fano_test, blers_fano_test], open(fano_path, 'wb'))

        print("Test SNRs : ", snr_range)
        print("BERs of RNN: {0}".format(bers_RNN_test))
        print("BERs of SC decoding: {0}".format(bers_Dumer_test))
        print("BERs of ML: {0}".format(bers_ML_test))
        print("BERs of Fano: {0}".format(bers_fano_test))

        print("Time taken = {} seconds".format(time.time() - start_time))
        ## BER
        plt.figure(figsize = (12,8))

        ok = 0
        plt.semilogy(snr_range, bers_RNN_test, label="RNN decoder", marker='*', linewidth=1.5)

        if args.run_dumer:
            plt.semilogy(snr_range, bers_Dumer_test, label="SC decoder", marker='^', linewidth=1.5)

        if args.are_we_doing_ML:
            plt.semilogy(snr_range, bers_ML_test, label="ML decoder", marker='o', linewidth=1.5)
        # if args.run_fano:
        plt.semilogy(snr_range_fano, bers_fano_test, label="Fano decoder", marker='P', linewidth=1.5)

        ## BLER
        plt.semilogy(snr_range, blers_RNN_test, label="RNN decoder (BLER)", marker='*', linewidth=1.5, linestyle='dashed')
        if args.run_dumer:
            plt.semilogy(snr_range, blers_Dumer_test, label="SC decoder (BLER)", marker='^', linewidth=1.5, linestyle='dashed')

        if args.are_we_doing_ML:
            plt.semilogy(snr_range, blers_ML_test, label="ML decoder", marker='o', linewidth=1.5, linestyle='dashed')
        # if args.run_fano:
        plt.semilogy(snr_range_fano, blers_fano_test, label="Fano decoder", marker='P', linewidth=1.5, linestyle='dashed')

        plt.grid()
        plt.xlabel("SNR (dB)", fontsize=16)
        plt.ylabel("Error Rate", fontsize=16)
        plt.title("PAC({1}, {2}): RNN trained at Dec_SNR = {0} dB".format(args.dec_train_snr, args.K,args.N))
        plt.legend(prop={'size': 15})
        if args.test_load_path is not None:
            os.makedirs('RNN_PAC_Results/figures', exist_ok=True)
            fig_save_path = 'RNN_PAC_Results/figures/new_plot.pdf'
        else:
            fig_save_path = results_load_path + "/step_{}.pdf".format(args.model_iters if args.model_iters is not None else '_final')
        plt.savefig(fig_save_path)

        plt.close()

    elif args.code == 'Polar':

        bers_RNN_test, blers_RNN_test, bers_SC_test, blers_SC_test, bers_SCL_test, blers_SCL_test, bers_RNNL_test, blers_RNNL_test, bers_ML_test, blers_ML_test = polar_RNN_full_test(net, code, snr_range, Test_Data_Generator, args.are_we_doing_ML, args.list_size is not None, False)

        print("Test SNRs : ", snr_range)
        print("BERs of RNN: {0}".format(bers_RNN_test))
        print("BERs of SC decoding: {0}".format(bers_SC_test))
        if args.list_size is not None:
            print("BERs of SCL decoding, L={1}: {0}".format(bers_SCL_test, args.list_size))
            print("BERs of RNNL decoding, L={1}: {0}".format(bers_RNNL_test, args.list_size))
        print("BERs of ML: {0}".format(bers_ML_test))

        print("Time taken = {} seconds".format(time.time() - start_time))
        ## BER
        plt.figure(figsize = (12,8))

        ok = 0
        plt.semilogy(snr_range, bers_RNN_test, label="RNN decoder", marker='*', linewidth=1.5)
        plt.semilogy(snr_range, bers_SC_test, label="SC decoder", marker='^', linewidth=1.5)
        if args.list_size is not None:
            plt.semilogy(snr_range, bers_SCL_test, label="SC List decoder, L={}".format(args.list_size), marker='P', linewidth=1.5)
            plt.semilogy(snr_range, bers_RNNL_test, label="RNN List decoder, L={}".format(args.list_size), marker='v', linewidth=1.5)

        if args.are_we_doing_ML:
            plt.semilogy(snr_range, bers_ML_test, label="ML decoder", marker='o', linewidth=1.5)
        # if args.run_fano:
        ## BLER
        plt.semilogy(snr_range, blers_RNN_test, label="RNN decoder (BLER)", marker='*', linewidth=1.5, linestyle='dashed')
        plt.semilogy(snr_range, blers_SC_test, label="SC decoder (BLER)", marker='^', linewidth=1.5, linestyle='dashed')
        if args.list_size is not None:
            plt.semilogy(snr_range, blers_SCL_test, label="SC List decoder, L={}".format(args.list_size), marker='P', linewidth=1.5, linestyle='dashed')
            plt.semilogy(snr_range, blers_RNNL_test, label="RNN List decoder, L={}".format(args.list_size), marker='v', linewidth=1.5, linestyle='dashed')


        if args.are_we_doing_ML:
            plt.semilogy(snr_range, blers_ML_test, label="ML decoder", marker='o', linewidth=1.5, linestyle='dashed')
        # if args.run_fano:

        plt.grid()
        plt.xlabel("SNR (dB)", fontsize=16)
        plt.ylabel("Error Rate", fontsize=16)
        if args.rate_profile == 'polar':
            plt.title("Polar({1}, {2}): RNN trained at Dec_SNR = {0} dB".format(args.dec_train_snr, args.K,args.N))
        elif args.rate_profile == 'RM':
            plt.title("RM({1}, {2}): RNN trained at Dec_SNR = {0} dB".format(args.dec_train_snr, args.K,args.N))

        plt.legend(prop={'size': 15})
        if args.test_load_path is not None:
            os.makedirs('RNN_Polar_Results/figures', exist_ok=True)
            fig_save_path = 'RNN_Polar_Results/figures/new_plot.pdf'
        else:
            fig_save_path = results_load_path + "/step_{}.pdf".format(args.model_iters if args.model_iters is not None else '_final')
        plt.savefig(fig_save_path)

        plt.close()
