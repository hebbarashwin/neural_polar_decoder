import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from IPython import display

import pickle
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from utils import snr_db2sigma, errors_ber, errors_bitwise_ber, errors_bler, min_sum_log_sum_exp, moving_average, extract_block_errors, extract_block_nonerrors
from models import convNet,XFormerEndToEndGPT,XFormerEndToEndDecoder,XFormerEndToEndEncoder,rnnAttn
from polar import *
from pac_code import *

import math
import random
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import sys
import csv


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
    parser = argparse.ArgumentParser(description='Polar/PAC code - decoder')

    parser.add_argument('--id', type=str, default=None, help='ID: optional, to run multiple runs of same hyperparameters') #Will make a folder like init_932 , etc.
    
    parser.add_argument('--previous_id', type=str, default=None, help='ID: optional, to run multiple runs of same hyperparameters') #Will make a folder like init_932 , etc.

    parser.add_argument('--code', type=str, default='pac',choices=['pac', 'polar'], help='code to be tested/trained on')
    
    parser.add_argument('--previous_code', type=str, default=None,choices=[None,'pac', 'polar'], help='code to load model from')

    parser.add_argument('--N', type=int, default=32)#, choices=[4, 8, 16, 32, 64, 128], help='Polar code parameter N')
    
    parser.add_argument('--previous_N', type=int, default=32)#, choices=[4, 8, 16, 32, 64, 128], help='Polar code parameter N')
    
    parser.add_argument('--max_len', type=int, default=32)#, choices=[4, 8, 16, 32, 64, 128], help='Polar code parameter N')

    parser.add_argument('--K', type=int, default=8)#, choices= [3, 4, 8,  16, 32, 64], help='Polar code parameter K')
    
    parser.add_argument('--previous_K', type=int, default=8)#, choices= [3, 4, 8,  16, 32, 64], help='Polar code parameter K')

    parser.add_argument('--test', dest = 'test', default=False, action='store_true', help='Testing?')
    
    parser.add_argument('--plot_progressive', dest = 'plot_progressive', default=False, action='store_true', help='plot merged progressive ber vs time')
    
    parser.add_argument('--do_range_training', dest = 'do_range_training', default=False, action='store_true', help="training on dec_train_snr + 1 and + 2 also?")

    parser.add_argument('--rate_profile', type=str, default='RM', choices=['RM', 'polar', 'sorted', 'last', 'custom'], help='PAC rate profiling')
    
    parser.add_argument('--previous_rate_profile', type=str, default=None, choices=[None,'RM', 'polar', 'sorted', 'last', 'custom'], help='PAC rate profiling')

    parser.add_argument('--embed_dim', type=int, default=64)# embedding size / hidden size of input vectors/hidden outputs between layers

    parser.add_argument('--dropout', type=int, default=0.1)# dropout

    parser.add_argument('--n_head', type=int, default=8)# number of attention heads

    parser.add_argument('--n_layers', type=int, default=6)# number of transformer layers
    
    parser.add_argument('--num_devices', type=int, default=2)# number of transformer layers

    parser.add_argument('--load_previous', dest = 'load_previous', default=False, action='store_true', help='load previous model at step --model_iters')
    
    parser.add_argument('--parallel', dest = 'parallel', default=False, action='store_true', help='gpu parallel')
    
    parser.add_argument('--dont_use_bias', dest = 'dont_use_bias', default=False, action='store_true', help='dont use bias in neural net')# load previous while training?
    
    parser.add_argument('--include_previous_block_errors', dest = 'include_previous_block_errors', default=False, action='store_true', help='train again on block errors of the previous step')
    
    parser.add_argument('--dec_train_snr', type=float, default=-1., help='SNR at which decoder is trained')

    parser.add_argument('--test_snr_start', type=float, default=-2., help='testing snr start')

    parser.add_argument('--test_snr_end', type=float, default=4., help='testing snr end')

    parser.add_argument('--model_iters', type=int, default=None, help='by default load final model, option to load a model of x episodes')
    
    parser.add_argument('--run', type=int, default=None)#, choices= [3, 4, 8,  16, 32, 64], help='Polar code parameter K')
    
    parser.add_argument('--num_steps', type=int, default=400000)#, choices=[100, 20000, 40000], help='number of blocks')

    parser.add_argument('--batch_size', type=int, default=128)#, choices=[64, 128, 256, 1024], help='number of blocks')
    
    parser.add_argument('--mult', type=int, default=1)#, multiplying factor to increase effective batch size
    
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    parser.add_argument('--cosine', dest = 'cosine', default=False, action='store_true', help='cosine annealing')
    
    parser.add_argument('--num_restarts',type=int, default=1, help='number of restarts while cosine annealing')
    
    parser.add_argument('--print_freq', type=int, default=1000, help='validation every x steps')

    parser.add_argument('--activation', type=str, default='selu', choices=['selu', 'relu', 'elu', 'tanh', 'sigmoid'], help='activation function')
    
    parser.add_argument('--curriculum', type=str, default='c2n', choices=['c2n', 'n2c', 'r2l', 'l2r','random'], help='name of curriculum being followed')
    
    parser.add_argument('--target_K', type=int, default=16, help='target K while training a curriculum')

    # TRAINING parameters
    parser.add_argument('--model', type=str, default='gpt', choices=['simple','conv','encoder', 'decoder', 'gpt','denoiser','bigConv','small','multConv','rnnAttn','bitConv'], help='model to be trained')

    parser.add_argument('--initialization', type=str, default='Xavier', choices=['Dontknow', 'He', 'Xavier'], help='initialization')

    parser.add_argument('--optimizer_type', type=str, default='AdamW', choices=['Adam', 'RMS', 'AdamW','SGD'], help='optimizer type')

    parser.add_argument('--loss', type=str, default='MSE', choices=['Huber', 'MSE','NLL','Block'], help='loss function')

    parser.add_argument('--loss_on_all', dest = 'loss_on_all', default=False, action='store_true', help='loss on all bits or only info bits')

    parser.add_argument('--split_batch', dest = 'split_batch', default=False, action='store_true', help='split batch - for teacher forcing')

    

    parser.add_argument('--lr_decay', type=int, default=None, help='learning rate decay frequency (in episodes)')
    
    parser.add_argument('--T_anneal', type=int, default=None, help='Number of iterations to midway in cosine lr')
    

    parser.add_argument('--lr_decay_gamma', type=float, default=None, help='learning rate decay factor')

    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping factor')
    
    parser.add_argument('--validation_snr', type=float, default=None, help='snr at validation')

    parser.add_argument('--no_detach', dest = 'no_detach', default=False, action='store_true', help='detach previous output during rnn training?')

    # TEACHER forcing
    # if only tfr_max is given assume no annealing
    parser.add_argument('--tfr_min', type=float, default=None, help='teacher forcing ratio minimum')

    parser.add_argument('--tfr_max', type=float, default=0., help='teacher forcing ratio maximum')

    parser.add_argument('--tfr_decay', type=float, default=10000, help='teacher forcing ratio decay parameter')

    parser.add_argument('--teacher_steps', type=int, default=-10000, help='initial number of steps to do teacher forcing only')

    # TESTING parameters

    parser.add_argument('--model_save_per', type=int, default=5000, help='num of episodes after which model is saved')

    parser.add_argument('--snr_points', type=int, default=7, help='testing snr num points')

    parser.add_argument('--test_batch_size', type=int, default=1000, help='number of blocks')

    parser.add_argument('--test_size', type=int, default=50000, help='size of the batches')



    parser.add_argument('--test_load_path', type=str, default=None, help='load test model given path')

    parser.add_argument('--run_fano', dest = 'run_fano', default=False, action='store_true', help='run fano decoding')

    parser.add_argument('--random_test', dest = 'random_test', default=False, action='store_true', help='run test on random data (default action is to test on same samples as Fano did)')

    parser.add_argument('--save_path', type=str, default=None, help='save name')

    parser.add_argument('--load_path', type=str, default=None, help='load name')

    parser.add_argument("--run_dumer", type=str2bool, nargs='?', const=True, default=True, help="run dumer during test?")
    # parser.add_argument('-id', type=int, default=100000)
    parser.add_argument('--hard_decision', dest = 'hard_decision', default=False, action='store_true', help='polar code sc decoding hard decision?')

    parser.add_argument('--gpu', type=int, default= -1, help='gpus used for training - e.g 0,1,3') # -1 if run on any available gpu

    parser.add_argument('--anomaly', dest = 'anomaly', default=False, action='store_true', help='enable anomaly detection')

    parser.add_argument('--only_args', dest = 'only_args', default=False, action='store_true')

    args = parser.parse_args()

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

    args.are_we_doing_ML = True if args.K <=16 and args.N <= 32 else False

    # args.hard_decision = True # use hard-SC
    return args

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing
    bits (0 and 1).
    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.
    bit_width : int
        Size of the output bit array.
    Returns
    -------
    bitarray : 1D ndarray of ints
        Array containing the binary representation of the input decimal.
    """

    binary_string = bin(in_number)
    length = len(binary_string)
    bitarray = np.zeros(bit_width, 'int')
    for i in range(length-2):
        bitarray[bit_width-i-1] = int(binary_string[length-i-1])

    return bitarray

def countSetBits(n):

    count = 0
    while (n):
        n &= (n-1)
        count+= 1

    return count

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


def testXformer(net, polar, snr_range, Test_Data_Generator,device,Test_Data_Mask=None, run_ML=False, bitwise_snr_idx = -1):
    num_test_batches = len(Test_Data_Generator)

    bers_Xformer_test = [0. for ii in snr_range]
    bers_bitwise_Xformer_test = torch.zeros((1,polar.K),device=device)
    blers_Xformer_test = [0. for ii in snr_range]

    bers_SC_test = [0. for ii in snr_range]
    blers_SC_test = [0. for ii in snr_range]
    
    bers_SCL_test = [0. for ii in snr_range]
    blers_SCL_test = [0. for ii in snr_range]

    bers_ML_test = [0. for ii in snr_range]
    blers_ML_test = [0. for ii in snr_range]
    
    bers_bitwise_MAP_test = [0. for ii in snr_range]
    blers_bitwise_MAP_test = [0. for ii in snr_range]

    for (k, msg_bits) in tqdm(enumerate(Test_Data_Generator)):

        msg_bits = msg_bits.to(device)
        polar_code = polar.encode_plotkin(msg_bits)


        for snr_ind, snr in enumerate(snr_range):
            noisy_code = polar.channel(polar_code, snr)
            noise = noisy_code - polar_code
            if Test_Data_Mask == None:
                mask = torch.ones(noisy_code.size(),device=device).long()
            SC_llrs, decoded_SC_msg_bits = polar.sc_decode_new(noisy_code, snr)
            if not run_ML:
                SCL_llrs, decoded_SCL_msg_bits = polar.scl_decode(noisy_code.cpu(), snr, 4, use_CRC = False)
                ber_SCL = errors_ber(msg_bits.cpu(), decoded_SCL_msg_bits.sign().cpu()).item()
                bler_SCL = errors_bler(msg_bits.cpu(), decoded_SCL_msg_bits.sign().cpu()).item()
                bers_SCL_test[snr_ind] += ber_SCL/num_test_batches
                blers_SCL_test[snr_ind] += bler_SCL/num_test_batches
            
            ber_SC = errors_ber(msg_bits.cpu(), decoded_SC_msg_bits.sign().cpu()).item()
            bler_SC = errors_bler(msg_bits.cpu(), decoded_SC_msg_bits.sign().cpu()).item()            

            decoded_bits,out_mask = net.decode(noisy_code,polar.info_positions, mask,device)
            decoded_Xformer_msg_bits = decoded_bits[:, polar.info_positions].sign()

            ber_Xformer = errors_ber(msg_bits, decoded_Xformer_msg_bits.sign(), mask = mask[:, polar.info_positions]).item()
            if snr_ind==bitwise_snr_idx:
                ber_bitwise_Xformer = errors_bitwise_ber(msg_bits, decoded_Xformer_msg_bits.sign(), mask = mask[:, polar.info_positions]).squeeze()
                bers_bitwise_Xformer_test += ber_bitwise_Xformer/num_test_batches
                print(ber_bitwise_Xformer)
            bler_Xformer = errors_bler(msg_bits, decoded_Xformer_msg_bits.sign()).item()
            if run_ML:
                b_noisy = noisy_code.unsqueeze(1).repeat(1, 2**args.K, 1)
                diff = (b_noisy - b_codebook).pow(2).sum(dim=2)
                idx = diff.argmin(dim=1)
                decoded = all_message_bits[idx, :]
                decoded_bitwiseMAP_msg_bits = polar.bitwise_MAP(noisy_code,device,snr)

                ber_ML = errors_ber(msg_bits.to(decoded.device), decoded.sign()).item()
                bler_ML = errors_bler(msg_bits.to(decoded.device), decoded.sign()).item()
                ber_bitwiseMAP = errors_ber(msg_bits.cpu(), decoded_bitwiseMAP_msg_bits.sign().cpu()).item()
                bler_bitwiseMAP = errors_bler(msg_bits.cpu(), decoded_bitwiseMAP_msg_bits.sign().cpu()).item()
                bers_ML_test[snr_ind] += ber_ML/num_test_batches
                blers_ML_test[snr_ind] += bler_ML/num_test_batches
                bers_bitwise_MAP_test[snr_ind] += ber_bitwiseMAP/num_test_batches
                blers_bitwise_MAP_test[snr_ind] += bler_bitwiseMAP/num_test_batches

            bers_Xformer_test[snr_ind] += ber_Xformer/num_test_batches
            bers_SC_test[snr_ind] += ber_SC/num_test_batches
            
            blers_Xformer_test[snr_ind] += bler_Xformer/num_test_batches
            blers_SC_test[snr_ind] += bler_SC/num_test_batches
            
            
    print(bers_bitwise_Xformer_test)
    return bers_Xformer_test, blers_Xformer_test, bers_SC_test, blers_SC_test,bers_SCL_test, blers_SCL_test, bers_ML_test, blers_ML_test,bers_bitwise_Xformer_test,bers_bitwise_MAP_test,blers_bitwise_MAP_test

def PAC_MAP_decode(noisy_codes, b_codebook):

    b_noisy = noisy_codes.unsqueeze(1).repeat(1, 2**args.K, 1)

    diff = (b_noisy - b_codebook).pow(2).sum(dim=2)

    idx = diff.argmin(dim=1)

    MAP_decoded_bits = all_message_bits[idx, :]

    return MAP_decoded_bits


def test_RNN_and_Dumer_batch(net, pac, msg_bits, corrupted_codewords, snr, run_dumer=True,Test_Data_Mask =None,bitwise_snr = 1):

    state = corrupted_codewords

    ### DQN decoding
    info_inds = pac.B

    if Test_Data_Mask == None:
        mask = torch.ones(corrupted_codewords.size(),device=device).long()
    else:
        mask = Test_Data_Mask

    decoded_bits,out_mask = net.decode(corrupted_codewords,info_inds, mask,device)
    decoded_Xformer_msg_bits = decoded_bits[:, info_inds].sign()

    ber_Xformer = errors_ber(msg_bits, decoded_Xformer_msg_bits.sign(), mask = mask[:, info_inds]).item()
    bler_Xformer = errors_bler(msg_bits, decoded_Xformer_msg_bits.sign()).item()
    ber_bitwise_Xformer = -1
    if snr==bitwise_snr:
        ber_bitwise_Xformer = errors_bitwise_ber(msg_bits, decoded_Xformer_msg_bits.sign(), mask = mask[:, info_inds]).squeeze()
        

    if run_dumer:
        _, decoded_Dumer_msg_bits, _ = pac.pac_sc_decode(corrupted_codewords, snr)
        ber_Dumer = errors_ber(msg_bits, decoded_Dumer_msg_bits.sign()).item()
        bler_Dumer = errors_bler(msg_bits, decoded_Dumer_msg_bits.sign()).item()
    else:
        ber_Dumer = 0.
        bler_Dumer = 0.

    if args.are_we_doing_ML:
        MAP_decoded_bits = PAC_MAP_decode(corrupted_codewords, b_codebook)

        ber_ML = errors_ber(msg_bits, MAP_decoded_bits).item()
        bler_ML = errors_bler(msg_bits, MAP_decoded_bits).item()

        return ber_Xformer, bler_Xformer, ber_Dumer, bler_Dumer, ber_ML, bler_ML, ber_bitwise_Xformer

    else:

        return ber_Xformer, bler_Xformer, ber_Dumer, bler_Dumer, ber_bitwise_Xformer

def test_fano(pac,msg_bits, noisy_code, snr):

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

def test_full_data(net, pac, snr_range, Test_Data_Generator, run_fano = False, run_dumer = True, Test_Data_Mask=None):

    num_test_batches = len(Test_Data_Generator)

    bers_RNN_test = [0. for ii in snr_range]
    blers_RNN_test = [0. for ii in snr_range]

    bers_Dumer_test = [0. for ii in snr_range]
    blers_Dumer_test = [0. for ii in snr_range]

    bers_ML_test = [0. for ii in snr_range]
    blers_ML_test = [0. for ii in snr_range]

    bers_fano_test = [0. for ii in snr_range]
    blers_fano_test = [0. for ii in snr_range]

    for (k, msg_bits) in tqdm(enumerate(Test_Data_Generator)):

        msg_bits = msg_bits.to(device)
        pac_code = pac.pac_encode(msg_bits, scheme = args.rate_profile)

        for snr_ind, snr in enumerate(snr_range):
            noisy_code = pac.channel(pac_code, snr)
            if Test_Data_Mask == None:
                mask = torch.ones(noisy_code.size(),device=device).long()
            if args.are_we_doing_ML:

                ber_RNN, bler_RNN, ber_Dumer, bler_Dumer, ber_ML, bler_ML  = test_RNN_and_Dumer_batch(net, pac, msg_bits, noisy_code, snr, Test_Data_Mask=mask)

            else:

                ber_RNN, bler_RNN, ber_Dumer, bler_Dumer,_  = test_RNN_and_Dumer_batch(net, pac, msg_bits, noisy_code, snr, run_dumer, Test_Data_Mask=mask)

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

def test_standard(net, pac, msg_bits_all, received, run_fano = False, run_dumer = True, Test_Data_Mask=None,bitwise_snr_idx = 3):

    snr_range = list(received.keys())
    bers_RNN_test = [0. for ii in snr_range]
    blers_RNN_test = [0. for ii in snr_range]

    bers_Dumer_test = [0. for ii in snr_range]
    blers_Dumer_test = [0. for ii in snr_range]

    bers_ML_test = [0. for ii in snr_range]
    blers_ML_test = [0. for ii in snr_range]

    bers_fano_test = [0. for ii in snr_range]
    blers_fano_test = [0. for ii in snr_range]
    
    bers_bitwise_Xformer_test = torch.zeros((1,pac.K),device=device)

    msg_bits_all = msg_bits_all.to(device)
    # quick fix to get this running. need to modify to support other test batch sizes ig
    num_test_batches = msg_bits_all.shape[0]//args.test_batch_size
    for snr_ind, (snr, noisy_code_all) in enumerate(received.items()):
        noisy_code_all = noisy_code_all.to(device)
        if snr_ind == bitwise_snr_idx:
            bitwise_snr = snr
        else:
            bitwise_snr = -100
        for ii in range(num_test_batches):
            msg_bits = msg_bits_all[ii*args.test_batch_size: (ii+1)*args.test_batch_size]
            noisy_code = noisy_code_all[ii*args.test_batch_size: (ii+1)*args.test_batch_size]
            if Test_Data_Mask == None:
                mask = torch.ones(noisy_code.size(),device=device).long()
            if args.are_we_doing_ML:

                ber_RNN, bler_RNN, ber_Dumer, bler_Dumer, ber_ML, bler_ML, ber_bitwise_Xformer  = test_RNN_and_Dumer_batch(net, pac, msg_bits, noisy_code, snr, Test_Data_Mask=mask, bitwise_snr = bitwise_snr)

            else:

                ber_RNN, bler_RNN, ber_Dumer, bler_Dumer, ber_bitwise_Xformer  = test_RNN_and_Dumer_batch(net, pac, msg_bits, noisy_code, snr, Test_Data_Mask=mask, bitwise_snr = bitwise_snr)
            if snr_ind==bitwise_snr_idx:
                #ber_bitwise_Xformer = errors_bitwise_ber(msg_bits, decoded_Xformer_msg_bits.sign(), mask = mask[:, polar.info_positions]).squeeze()
                bers_bitwise_Xformer_test += ber_bitwise_Xformer/num_test_batches
            bers_RNN_test[snr_ind] += ber_RNN/num_test_batches
            bers_Dumer_test[snr_ind] += ber_Dumer/num_test_batches

            blers_RNN_test[snr_ind] += bler_RNN/num_test_batches
            blers_Dumer_test[snr_ind] += bler_Dumer/num_test_batches

            if args.are_we_doing_ML:
                bers_ML_test[snr_ind] += ber_ML/num_test_batches
                blers_ML_test[snr_ind] += bler_ML/num_test_batches

        if run_fano:
            ber_fano, bler_fano = test_fano(msg_bits_all, noisy_code_all, snr)
            bers_fano_test[snr_ind] += ber_fano
            blers_fano_test[snr_ind] += bler_fano

    return bers_RNN_test, blers_RNN_test, bers_Dumer_test, blers_Dumer_test, bers_ML_test, blers_ML_test, bers_fano_test, blers_fano_test,bers_bitwise_Xformer_test

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    args = get_args()
    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)

    if args.gpu == -1: #run on any available device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else: #run on specified gpu
        device = torch.device("cuda:{0}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

    #torch.manual_seed(37)
    kwargs = {'num_workers': 4, 'pin_memory': False} if torch.cuda.is_available() else {}
    if args.previous_code is None:
        args.previous_code = args.code
    if args.previous_rate_profile is None:
        args.previous_rate_profile = args.rate_profile
    ID = '' if args.id is None else args.id
    lr_ = args.lr if args.lr_decay is None else str(args.lr)+'_decay_{}_{}'.format(args.lr_decay, args.lr_decay_gamma)
    if args.code == 'polar':
        results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                    .format(args.K, args.N, args.rate_profile,  args.model, args.n_head,args.n_layers)
        if args.save_path is None:
            final_save_path = './Supervised_Xformer_decoder_Polar_Results/final_nets/Scheme_{2}/N{1}_K{0}_{3}_{4}_depth_{5}.pt'\
                                    .format(args.K, args.N, args.rate_profile,  args.model, args.n_head,args.n_layers)
        else:
            final_save_path = args.save_path
    elif args.code== 'pac':
        results_save_path = './Supervised_Xformer_decoder_PAC_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                    .format(args.K, args.N, args.rate_profile,  args.model, args.n_head,args.n_layers)
        if args.save_path is None:
            final_save_path = './Supervised_Xformer_decoder_PAC_Results/final_nets/Scheme_{2}/N{1}_K{0}_{3}_{4}_depth_{5}.pt'\
                                    .format(args.K, args.N, args.rate_profile,  args.model, args.n_head,args.n_layers)
        else:
            final_save_path = args.save_path
    if ID != '':
        results_save_path = results_save_path + '/' + ID
        final_save_path = final_save_path + '/' + ID
    
    if args.previous_code == 'polar':
        previous_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                    .format(args.previous_K, args.previous_N, args.previous_rate_profile,  args.model, args.n_head,args.n_layers)
    elif args.previous_code == 'pac':
        previous_save_path = './Supervised_Xformer_decoder_PAC_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                    .format(args.previous_K, args.previous_N, args.previous_rate_profile,  args.model, args.n_head,args.n_layers)
    if args.previous_id is not None:
        previous_save_path = previous_save_path + '/' + args.previous_id
    else:
        previous_save_path = previous_save_path #+ '/' + ID
    if args.run is not None:
        results_save_path = results_save_path + '/' + '{0}'.format(args.run)
        final_save_path = final_save_path + '/' + '{0}'.format(args.run)
        previous_save_path = previous_save_path + '/' + '{0}'.format(args.run)
    
    ############
    ## Polar Code parameters
    ############
    K = args.K
    N = args.N
    n = int(np.log2(args.N))
    target_K = args.target_K

    if args.rate_profile == 'polar':
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
        # Multiple SNRs:

        ###############
        ### Polar code
        ##############

        ### Encoder
        if args.code=='polar':
            polar = PolarCode(n, args.K, args, rs=rs)
            polarTarget = PolarCode(n, args.target_K, args, rs=rs)
        elif args.code=='pac':
            polar = PAC(args, args.N, args.K, args.g)
            polarTarget = PAC(args, args.N, args.target_K, args.g)
    elif args.rate_profile == 'RM':
        rmweight = np.array([countSetBits(i) for i in range(args.N)])
        Fr = np.argsort(rmweight)[:-args.K]
        Fr.sort()
        if args.code=='polar':
            polar = PolarCode(n, args.K, args, F=Fr)
            rmweight = np.array([countSetBits(i) for i in range(args.N)])
            Fr = np.argsort(rmweight)[:-args.target_K]
            Fr.sort()
            polarTarget = PolarCode(n, args.target_K, args, F=Fr)
        elif args.code=='pac':
            polar = PAC(args, args.N, args.K, args.g)
            polarTarget = PAC(args, args.N, args.target_K, args.g)
    
    if args.curriculum == 'c2n':
        if args.code == 'polar':
            info_inds = polar.info_positions
            frozen_inds = polar.frozen_positions
        elif args.code == 'pac':
            frozen_levels = (polar.rate_profiler(-torch.ones(1, args.K), scheme = args.rate_profile) == 1.)[0].numpy()
            info_inds = polar.B
            frozen_inds = np.array(list(set(np.arange(args.N))^set(polar.B)))
    elif args.curriculum == 'n2c':
        if args.code == 'polar':
            info_inds = polarTarget.unsorted_info_positions[:args.K].copy()
            frozen_inds = polarTarget.frozen_positions
        elif args.code == 'pac':
            frozen_levels = (polar.rate_profiler(-torch.ones(1, args.K), scheme = args.rate_profile) == 1.)[0].numpy()
            info_inds = polarTarget.unsorted_info_positions[:args.K].copy()
            frozen_inds = np.array(list(set(np.arange(args.N))^set(polar.B)))
    elif args.curriculum == 'l2r':
        if args.code == 'polar':
            info_inds = polarTarget.info_positions[:args.K].copy()
            frozen_inds = polarTarget.frozen_positions
        elif args.code == 'pac':
            frozen_levels = (polar.rate_profiler(-torch.ones(1, args.K), scheme = args.rate_profile) == 1.)[0].numpy()
            info_inds = polarTarget.B[:args.K].copy()
            frozen_inds = np.array(list(set(np.arange(args.N))^set(polar.B)))
    elif args.curriculum == 'r2l':
        if args.code == 'polar':
            info_inds = polarTarget.info_positions[-args.K:].copy()
            frozen_inds = polarTarget.frozen_positions
        elif args.code == 'pac':
            frozen_levels = (polar.rate_profiler(-torch.ones(1, args.K), scheme = args.rate_profile) == 1.)[0].numpy()
            info_inds = polarTarget.B[-args.K:].copy()
            frozen_inds = np.array(list(set(np.arange(args.N))^set(polar.B)))
    elif args.curriculum == 'random':
        if args.code == 'polar':
            random_info = polarTarget.info_positions.copy()
            random.Random(42).shuffle(random_info)
            info_inds = random_info[:args.K].copy()
            frozen_inds = polarTarget.frozen_positions
        elif args.code == 'pac':
            frozen_levels = (polar.rate_profiler(-torch.ones(1, args.K), scheme = args.rate_profile) == 1.)[0].numpy()
            info_inds = polarTarget.B[-args.K:].copy()
            frozen_inds = np.array(list(set(np.arange(args.N))^set(polar.B)))
    
    info_inds.sort()
    if args.code == 'polar':
        target_info_inds = polarTarget.info_positions
    elif args.code == 'pac':
        target_info_inds = polarTarget.B
    target_info_inds.sort()
    print("Info positions : {}".format(info_inds))
    print("Target Info positions : {}".format(target_info_inds))
    print("Frozen positions : {}".format(frozen_inds))
    print("Code : {0} ".format(args.code))
    print("Type of training : {0}".format(args.curriculum))
    print("Rate Profile : {0}".format(args.rate_profile))
    print("Validation SNR : {0}".format(args.validation_snr))

    #___________________Model Definition___________________________________________________#
    
    if args.model == 'gpt':
        xformer = XFormerEndToEndGPT(args)
    elif args.model == 'decoder':
        xformer = XFormerEndToEndDecoder(args)
    elif args.model == 'encoder':
        xformer = XFormerEndToEndEncoder(args)
    elif args.model == 'conv':
        xformer = convNet(args)
    elif args.model == 'rnnAttn':
        xformer = rnnAttn(args)
    


    if not args.test:  # train the model
        os.makedirs(results_save_path, exist_ok=True)
        os.makedirs(results_save_path +'/Models', exist_ok=True)
        os.makedirs(final_save_path , exist_ok=True)
        os.makedirs(final_save_path +'/Models', exist_ok=True)
        
        if args.model_iters is not None and args.load_previous :
            checkpoint1 = torch.load(previous_save_path +'/Models/model_{0}.pt'.format(args.model_iters), map_location=lambda storage, loc: storage)
            #xformer.load_state_dict(torch.load(PATH))
            loaded_step = checkpoint1['step']
            xformer.load_state_dict(checkpoint1['xformer'])
            print("Training Model for {0},{1} loaded at step {2} from previous model {3},{4}".format(args.K,args.N,loaded_step,args.previous_K,args.previous_N))
        else:
            print("Training Model for {0},{1} anew".format(args.K,args.N))
        device_ids = range(args.num_devices)
        if args.parallel:
            xformer = torch.nn.DataParallel(xformer, device_ids=device_ids)
            
        xformer.to(device)
        print("Number of parameters :",count_parameters(xformer))
        
        if args.only_args:
            print("Loaded args. Exiting")
            sys.exit()
        ##############
        ### Optimizers
        ##############
        if args.optimizer_type == 'Adam':
            optimizer = optim.Adam(xformer.parameters(), lr = args.lr)
        elif args.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(xformer.parameters(), lr = args.lr)
        elif args.optimizer_type == 'RMS':
            optimizer = optim.RMSprop(xformer.parameters(), lr = args.lr)
        elif args.optimizer_type == 'SGD':
            optimizer = optim.SGD(xformer.parameters(), lr = args.lr,momentum=1e-4, dampening=0,nesterov = True)
        else:
            raise Exception("Optimizer not supported yet!")

        if args.lr_decay is None:
            scheduler = None
        else:
            if args.T_anneal is None:
                scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay*args.K , args.lr_decay_gamma)
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_anneal, eta_min=5e-5)
        
        if args.cosine:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,2200,args.num_steps,num_cycles=args.num_restarts)

        if args.loss == 'Huber':
            loss_fn = F.smooth_l1_loss
        elif args.loss == 'MSE':
            loss_fn = nn.MSELoss(reduction='mean')
        elif args.loss == 'NLL':
            loss_fn = nn.NLLLoss()
        elif args.loss == 'Block':
            loss_fn = None

        training_losses = []
        training_bers = []
        valid_bers = []
        valid_bitwise_bers= []
        valid_tgt_bers = []
        valid_blers = []
        valid_tgt_blers = []
        valid_steps = []
        
        test_data_path = './data/polar/test/test_N{0}_K{1}.p'.format(args.N, args.K)
        try:
            test_dict = torch.load(test_data_path)
            valid_msg_bits = test_dict['msg']
            valid_received = test_dict['rec']
            print(valid_received.size())
        except:
            print("Did not find standard validation data")
        
        
        mavg_steps = 25
        
        print("Need to save for:", args.model_save_per)
        xformer.train()
        first = [info_inds[0]] 
        range_snr = [args.dec_train_snr,args.dec_train_snr+1,args.dec_train_snr+2]
        if args.validation_snr is not None:
            valid_snr = args.validation_snr
        else:
            valid_snr = args.dec_train_snr
        info_inds = info_inds.copy()
        #acc_error_egs = 
        kernel = 7
        padding = int((kernel-1)/2)
        layersint = nn.Sequential(
            nn.Conv1d(64,1,kernel,padding=padding,dilation=1),
            )
        layersint.to(device)
        try:
            for i_step in range(args.num_steps):  ## Each episode is like a sample now until memory size is reached.
                randperm = torch.randperm(args.batch_size)
                if args.do_range_training:
                    train_snr = args.dec_train_snr
                    if args.code == 'polar':# and (i_step < 2000 or args.model=='multConv'):
                        range_snr = [args.dec_train_snr-1,args.dec_train_snr,args.dec_train_snr+5]
                        train_snr = range_snr[i_step%3]
                    if args.code == 'pac':# and i_step < 15000:
                        range_snr = [args.dec_train_snr,args.dec_train_snr+1,args.dec_train_snr+2]
                        train_snr = range_snr[i_step%3]
                else:
                    train_snr = args.dec_train_snr
                start_time = time.time()
                #torch.cuda.empty_cache()
                msg_bits = 1 - 2 * (torch.rand(args.batch_size, args.K, device=device) < 0.5).float()
                gt = torch.ones(args.batch_size, args.N, device = device)
                gt[:, info_inds] = msg_bits
                gt_valid = gt.clone()
                if args.code == 'polar':
                    polar_code = polar.encode_plotkin(msg_bits,custom_info_positions = info_inds)
                    corrupted_codewords = polar.channel(polar_code, train_snr)#args.dec_train_snr)
                elif args.code == 'pac':
                    polar_code = polar.pac_encode(msg_bits, scheme = args.rate_profile,custom_info_positions = info_inds)
                    corrupted_codewords = polar.channel(polar_code, train_snr)#args.dec_train_snr)
                mask = torch.cat((torch.ones((args.batch_size,args.N),device=device),torch.zeros((args.batch_size,args.max_len-args.N),device=device)),1).long()
                
                if args.include_previous_block_errors and i_step%100 not in [0,1,2,3,4,5,6,7,8]:
                    #print(error_egs_corrupted.size())
                    corrupted_codewords = error_egs_corrupted
                    gt = error_egs_true
                    # corrupted_codewords = corrupted_codewords[randperm]
                    # gt = gt[randperm]
                if args.model == 'conv' or args.model == 'bigConv' or args.model == 'multConv':
                    model_out,decoded_vhat,out_mask,logits,int_layer = xformer(corrupted_codewords,mask,gt,device)
                else:
                    model_out,decoded_vhat,out_mask,logits = xformer(corrupted_codewords,mask,gt,device)

                batch_size = gt.size(0)
                max_len = gt.size(1)

                if args.model == 'gpt' or args.model == 'decoder':
                    pass#gt = (gt*torch.ones((max_len,batch_size,max_len),device=device)).permute((1,0,2)).reshape(batch_size*max_len,max_len)
                elif args.model == 'denoiser':
                    gt = polar_code
                #print(decoded_vhat.size())
                decoded_msg_bits = decoded_vhat[:,info_inds]
                if args.loss == 'NLL':
                    loss = loss_fn(torch.log(model_out[:,info_inds,:]).transpose(1,2),(gt[:, info_inds]==1).long())
                elif args.loss == 'MSE':
                    #out_mask[:,0]=100
                    loss = loss_fn(out_mask[:,info_inds]*logits[:,info_inds,0],out_mask[:,info_inds]*gt[:, info_inds])#+0.5*loss_fn(layersint(int_layer).squeeze(),polar_code)#*args.N
                    #print(logits.size())
                    #loss = loss_fn(out_mask[:,first]*logits[:,first,0],out_mask[:,first]*gt[:, first])#*args.N
                    #out_mask[:,0]=1
                elif args.loss == 'Block':
                    loss = torch.mean(torch.max(out_mask[:,info_inds]*(logits[:,info_inds,0]-gt[:, info_inds])**2,-1).values)
                else:
                    loss = torch.sum(out_mask[:,info_inds]*(model_out[:,info_inds,1]-(gt[:, info_inds]==1).float())**2)/torch.sum(out_mask[:,info_inds])
                # OLD LOSS: on all bits
                # if args.loss_on_all:
                #     loss = loss_fn(decoded_vhat, gt)
                # else:
                #     # NEW LOSS : only on info bits
                #     loss = loss_fn(msg_bits, decoded_msg_bits)
                ber = errors_ber(gt[:,info_inds].cpu(), decoded_msg_bits.cpu(),out_mask[:,info_inds].cpu()).item()
                
                if args.include_previous_block_errors and i_step%100 in [0,1,2,3,4,5,6,7,8]:
                    if i_step == 0:
                        error_egs_corrupted = corrupted_codewords.clone()
                        error_egs_true = gt.clone()
                    error_inds, = extract_block_errors(gt[:,info_inds].cpu(), decoded_msg_bits.cpu(),thresh=5)
                    
                    # print(error_inds.size)
                    _, decoded_SCL_msg_bits = polar.scl_decode(corrupted_codewords[error_inds,:].clone().cpu(), train_snr, 4, use_CRC = False)
                    correct_inds, = extract_block_nonerrors(gt[error_inds,:][:,info_inds].cpu(), decoded_SCL_msg_bits.cpu(),thresh=1)
                    #print(correct_inds.size)
                    error_egs_corrupted = torch.cat((corrupted_codewords[correct_inds,:].clone(),error_egs_corrupted),0)[:args.batch_size,:]
                    error_egs_true = torch.cat((gt[correct_inds,:].clone(),error_egs_true),0)[:args.batch_size,:]
                    
                    # print(error_egs_corrupted.size())
                    # print('\n')
                (loss/args.mult).backward()
                torch.nn.utils.clip_grad_norm_(xformer.parameters(), args.clip) # gradient clipping to avoid exploding gradient
                
                if i_step%args.mult == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                training_losses.append(round(loss.item(),5))
                training_bers.append(round(ber, 5))
		
                if i_step % args.print_freq == 0:
                    xformer.eval()
                    with torch.no_grad():
                        corrupted_codewords_valid = polar.channel(polar_code, valid_snr)
                        decoded_no_noise,_ = xformer.decode(polar_code,info_inds,mask,device)
                        decoded_bits,out_mask = xformer.decode(corrupted_codewords_valid,info_inds,mask,device)
                        decoded_Xformer_msg_bits = decoded_bits[:, info_inds]
                        decoded_Xformer_msg_bits_no_noise = decoded_no_noise[:, info_inds]
                        if args.model == 'denoiser':
                            ber_Xformer = errors_ber(gt_valid[:,info_inds], decoded_Xformer_msg_bits, mask = out_mask[:,info_inds]).item()
                        else:
                            ber_Xformer = errors_ber(gt_valid[:,info_inds], decoded_Xformer_msg_bits, mask = out_mask[:,info_inds]).item()
                            ber_Xformer_noiseless = errors_ber(gt_valid[:,info_inds], decoded_Xformer_msg_bits_no_noise, mask = out_mask[:,info_inds]).item()
                            bler_Xformer = errors_bler(gt_valid[:,info_inds], decoded_Xformer_msg_bits).item()
                            bler_Xformer_noiseless = errors_bler(gt_valid[:,info_inds], decoded_Xformer_msg_bits_no_noise).item()
                            #ber_Xformer = errors_ber(gt[:,first], decoded_bits[:,first], mask = out_mask[:,first]).item()
                        if args.K < args.target_K:
                            msg_bits = 1 - 2 * (torch.rand(args.batch_size, args.target_K, device=device) < 0.5).float()
                            gt = torch.ones(args.batch_size, args.N, device = device)
                            gt[:, target_info_inds] = msg_bits
                            
                            if args.code == 'polar':
                                polar_code = polarTarget.encode_plotkin(msg_bits)
                                corrupted_codewords = polarTarget.channel(polar_code, valid_snr)#args.dec_train_snr)
                            elif args.code == 'pac':
                                polar_code = polarTarget.pac_encode(msg_bits, scheme = args.rate_profile)
                                corrupted_codewords = polarTarget.channel(polar_code, valid_snr)#args.dec_train_snr)
                            decoded_bits,out_mask = xformer.decode(corrupted_codewords,target_info_inds,mask,device)
                            decoded_Xformer_msg_bits = decoded_bits[:, target_info_inds]
                            ber_Xformer_tgt = errors_ber(msg_bits, decoded_Xformer_msg_bits, mask = out_mask[:,target_info_inds]).item()
                            bler_Xformer_tgt = errors_bler(msg_bits, decoded_Xformer_msg_bits).item()
                            bitwise_ber_Xformer_tgt = errors_bitwise_ber(msg_bits, decoded_Xformer_msg_bits, mask = out_mask[:,target_info_inds]).squeeze().cpu().tolist()
                        else:
                            bitwise_ber_Xformer_tgt = errors_bitwise_ber(msg_bits, decoded_Xformer_msg_bits, mask = out_mask[:,target_info_inds]).squeeze().cpu().tolist()
                            bler_Xformer_tgt = errors_bler(msg_bits, decoded_Xformer_msg_bits).item()
                    #print(bitwise_ber_Xformer_tgt)
                    valid_bers.append(round(ber_Xformer, 5))
                    valid_blers.append(round(bler_Xformer, 5))
                    valid_tgt_blers.append(round(bler_Xformer_tgt, 5))
                    if args.K < args.target_K:
                        valid_tgt_bers.append(round(ber_Xformer_tgt, 5))
                    else:
                        valid_tgt_bers.append(round(ber_Xformer, 5))
                    valid_steps.append(i_step)
                    valid_bitwise_bers.append(bitwise_ber_Xformer_tgt)
                    xformer.train()
                    try:
                        print('[%d/%d] At %d dB, Loss: %.7f, Train BER (%d dB) : %.7f, Valid BER: %.7f, Tgt BER: %.7f, Noiseless BER %.7f, Valid BLER : %.7f'
                                    % (i_step, args.num_steps,  valid_snr, loss,train_snr,ber, ber_Xformer,ber_Xformer_tgt,ber_Xformer_noiseless,bler_Xformer))
                    except:
                        print('[%d/%d] At %d dB, Loss: %.7f, Train BER (%d dB) : %.7f, Valid BER: %.7f, Tgt BER: %.7f, Noiseless BER %.7f, Valid BLER : %.7f'
                                    % (i_step, args.num_steps,  valid_snr, loss,train_snr,ber, ber_Xformer,ber_Xformer,ber_Xformer_noiseless,bler_Xformer))
                if i_step == 10:
                    print("Time for one step is {0:.4f} minutes".format((time.time() - start_time)/60))

                # Save the model for safety

                if ((i_step+1) % args.model_save_per == 0) or (i_step+1 == 10) or ((i_step+1) % args.num_steps == 0):

                    # print(i_episode +1 )
                    torch.save({'xformer': xformer.state_dict(), 'step':i_step+1, 'args':args} ,\
                                    results_save_path+'/Models/model_{0}.pt'.format(i_step+1))
                    torch.save({'xformer': xformer.state_dict(), 'step':i_step+1, 'args':args} ,\
                                    final_save_path+'/Models/model_final.pt')
                    # torch.save({'xformer': xformer.state_dict(), 'step':i_step+1, 'args':args} ,\
                    #                 final_save_path)


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
                
            with open(os.path.join(results_save_path, 'values_validation.csv'), 'w') as f:

                # using csv.writer method from CSV package
                write = csv.writer(f)

                write.writerow(valid_steps)
                write.writerow(valid_bers)
                write.writerow(valid_tgt_bers)
                
                for i in range(target_K):
                    write.writerow([bitwise_bers[i] for bitwise_bers in valid_bitwise_bers])
                    
                write.writerow(valid_blers)
                write.writerow(valid_tgt_blers)
                
            print('Complete')

        except KeyboardInterrupt:
            torch.save({'xformer': xformer.state_dict(), 'step':i_step+1, 'args':args} ,\
                            results_save_path+'/Models/model_{0}.pt'.format(i_step+1))
            torch.save({'xformer': xformer.state_dict(), 'step':i_step+1, 'args':args} ,\
                                    final_save_path+'/Models/model_final.pt')
            # torch.save({'net': xformer.state_dict(), 'step':i_step+1, 'args':args} ,\
            #                 final_save_path)

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
                
            with open(os.path.join(results_save_path, 'values_validation.csv'), 'w') as f:

                # using csv.writer method from CSV package
                write = csv.writer(f)

                write.writerow(valid_steps)
                write.writerow(valid_bers)
                write.writerow(valid_tgt_bers)
                
                for i in range(target_K):
                    write.writerow([bitwise_bers[i] for bitwise_bers in valid_bitwise_bers])
    else:
        print("TESTING :")
        
        if args.plot_progressive:
            k = args.K
            plt.figure(figsize = (20,10))
            ber_tgt = []
            net_iters = [0]
            snr = args.validation_snr
            bers_SC_test = 0.
            bers_SCL_test = 0.
            bers_SC_test_bitwise = torch.zeros((1,args.target_K),device=device)
            num_batches = 10
            batch=1000
            tot = batch*num_batches
            for _ in tqdm(range(num_batches)):
                msg_bits = 1 - 2 * (torch.rand(batch, args.target_K, device=device) < 0.5).float()
                polar_code = polarTarget.encode_plotkin(msg_bits)
                noisy_code = polarTarget.channel(polar_code, snr)
                noise = noisy_code - polar_code
                SC_llrs, decoded_SC_msg_bits = polarTarget.sc_decode_new(noisy_code, snr)
                SCL_llrs, decoded_SCL_msg_bits = polarTarget.scl_decode(noisy_code.cpu(), snr, 4, use_CRC = False)
                ber_SC = errors_ber(msg_bits.cpu(), decoded_SC_msg_bits.sign().cpu()).item()
                ber_SC_bitwise = errors_bitwise_ber(msg_bits.cpu(), decoded_SC_msg_bits.sign().cpu()).squeeze()
                ber_SCL = errors_ber(msg_bits.cpu(), decoded_SCL_msg_bits.sign().cpu()).item()
                bers_SC_test += ber_SC/num_batches
                bers_SC_test_bitwise += ber_SC_bitwise/num_batches
                bers_SCL_test += ber_SCL/num_batches
            ber_SC_bitwise = ber_SC_bitwise.squeeze()
            while k <= args.target_K:
                if args.code == 'polar':
                    results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                                .format(k, args.N, args.rate_profile,  args.model, args.n_head,args.n_layers)
                elif args.code== 'pac':
                    results_save_path = './Supervised_Xformer_decoder_PAC_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                                .format(k, args.N, args.rate_profile,  args.model, args.n_head,args.n_layers)
                if ID != '':
                    results_scratch = results_save_path + '/' + 'scratch'
                    results_save_path = results_save_path + '/' + ID
                if args.run is not None:
                    results_save_path = results_save_path + '/' + '{0}'.format(args.run)
                rows = []
                with open(os.path.join(results_save_path, 'values_validation.csv')) as f:
                    csvRead = csv.reader(f)
                    for row in csvRead:
                        rows.append(list(map(float,row)))
                
                iterations = [it+net_iters[-1]+1 for it in rows[0]]
                net_iters = net_iters + iterations
                plt.axvline(x = net_iters[-1], color = 'grey', linestyle='dashed')
                ber = rows[1]
                ber_tgt = ber_tgt + rows[2]
                label = '{0},{1}'.format(k,args.N)
                plt.semilogy(iterations, ber, label=label)
                if k == target_K:
                    sc = np.ones(len(iterations)) * bers_SC_test
                    scl = np.ones(len(iterations)) * bers_SCL_test
                    
                    plt.semilogy(iterations,sc, label='SC'.format(k,args.N),linestyle='dashed')
                    plt.semilogy(iterations,scl, label='SCL'.format(k,args.N),linestyle='dashed')
                k += 1
            k-=1
            net_iters = net_iters[1:]
            print(len(net_iters))
            plt.semilogy(net_iters,ber_tgt, label='{0},{1} prog'.format(k,args.N))
            
            try:
                rows = []
                with open(os.path.join(results_scratch, 'values_validation.csv')) as f:
                        csvRead = csv.reader(f)
                        for row in csvRead:
                            rows.append(list(map(float,row)))
                #print(len(rows[2]))
                ber_scratch = rows[2]
                if len(ber_scratch) < len(net_iters):
                    plt.semilogy(net_iters[:len(ber_scratch)],ber_scratch, label='{0},{1} scr'.format(k,args.N))
                else:
                    plt.semilogy(net_iters,ber_scratch[:len(net_iters)], label='{0},{1} scr'.format(k,args.N))
            except:
                print("Did not find model trained from scratch")
            plt.legend(prop={'size': 7},loc='upper right', bbox_to_anchor=(1.1, 1))
            plt.ylim(bottom=1e-3)
            plt.ylim(top=0.6)
            plt.savefig(results_save_path +'/valid_progressive_log.pdf')
            plt.close()
            
            k = args.K
            plt.figure(figsize = (20,10))
            bitwise_ber = [[] for _ in range(args.target_K)]
            net_iters = [0]
            while k <= args.target_K:
                if args.code == 'polar':
                    results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                                .format(k, args.N, args.rate_profile,  args.model, args.n_head,args.n_layers)
                    info_inds1 = polarTarget.unsorted_info_positions.copy()
                elif args.code== 'pac':
                    results_save_path = './Supervised_Xformer_decoder_PAC_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                                .format(k, args.N, args.rate_profile,  args.model, args.n_head,args.n_layers)
                    info_inds1 = polarTarget.unsorted_info_positions.copy()
                if ID != '':
                    results_scratch = results_save_path + '/' + 'scratch'
                    results_save_path = results_save_path + '/' + ID
                if args.run is not None:
                    results_save_path = results_save_path + '/' + '{0}'.format(args.run)
                rows = []
                with open(os.path.join(results_save_path, 'values_validation.csv')) as f:
                    csvRead = csv.reader(f)
                    for row in csvRead:
                        rows.append(list(map(float,row)))
                for i in range(len(rows)-3):
                    bitwise_ber[i] = bitwise_ber[i] + rows[i+3]
                iterations = [it+net_iters[-1]+1 for it in rows[0]]
                net_iters = net_iters + iterations
                plt.axvline(x = net_iters[-1], color = 'grey', linestyle='dashed')
                
                k += 1
            k-=1
            net_iters = net_iters[1:]#net_iters[101:]
            for i in range(len(bitwise_ber)):
                plt.semilogy(net_iters,bitwise_ber[i], label='Bit {0}'.format(i))
                plt.annotate('{0}'.format(i), (net_iters[-1], bitwise_ber[i][-1]*0.99))
                sc = np.ones(len(net_iters)) * bers_SC_test_bitwise[0][i].item()
                plt.semilogy(net_iters,sc, label='SC bit {0}'.format(i),linestyle='dashed')
            plt.title("Hardest to easiest order : {0}".format(np.argsort(np.argsort(info_inds1))))
            plt.legend(prop={'size': 7},loc='upper right', bbox_to_anchor=(1.1, 1))
            plt.savefig(results_save_path +'/valid_progressive_bitwise_log.pdf')
            plt.close()
            
            if args.id == 'n2c' or args.id == 'c2n' or args.id == None:
                pass
            else:
                sys.exit()
            
            for i in range(len(bitwise_ber)):
                plt.figure(figsize = (20,10))
                bitwise_ber_n2c = [] 
                bitwise_ber_c2n = [] 
                bitwise_ber_scr = [] 
                net_iters = [0]
                k = args.K#+1
                
                while k <= args.target_K:
                    if args.code == 'polar':
                        results_save_path = './Supervised_Xformer_decoder_Polar_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                                    .format(k, args.N, args.rate_profile,  args.model, args.n_head,args.n_layers)
                        info_inds1 = polarTarget.unsorted_info_positions.copy()
                    elif args.code== 'pac':
                        results_save_path = './Supervised_Xformer_decoder_PAC_Results/Polar_{0}_{1}/Scheme_{2}/{3}/{4}_depth_{5}'\
                                                    .format(k, args.N, args.rate_profile,  args.model, args.n_head,args.n_layers)
                        info_inds1 = polarTarget.unsorted_info_positions.copy()
                    try:
                        if k == args.target_K:
                            rows = []
                            with open(os.path.join(results_save_path + '/scratch', 'values_validation.csv')) as f:
                                csvRead = csv.reader(f)
                                for row in csvRead:
                                    rows.append(list(map(float,row)))
                            bitwise_ber_scr = bitwise_ber_scr + rows[i+3]
                    except:
                        print("Did not find model trained from scratch")
                    try:
                        rows = []
                        with open(os.path.join(results_save_path + '/n2c', 'values_validation.csv')) as f:
                            csvRead = csv.reader(f)
                            for row in csvRead:
                                rows.append(list(map(float,row)))
                        bitwise_ber_n2c = bitwise_ber_n2c + rows[i+3]
                        rows = []
                        with open(os.path.join(results_save_path + '/c2n', 'values_validation.csv')) as f:
                            csvRead = csv.reader(f)
                            for row in csvRead:
                                rows.append(list(map(float,row)))
                    except:
                        print("Did not find n2c and c2n")
                        rows = []
                        with open(os.path.join(results_save_path, 'values_validation.csv')) as f:
                            csvRead = csv.reader(f)
                            for row in csvRead:
                                rows.append(list(map(float,row)))
                            
                    
                    bitwise_ber_c2n = bitwise_ber_c2n + rows[i+3]
                    iterations = [it+net_iters[-1]+1 for it in rows[0]]
                    net_iters = net_iters + iterations
                    plt.axvline(x = net_iters[-1], color = 'grey', linestyle='dashed')
                    k += 1
                net_iters = net_iters[1:]#net_iters[101:]
                plt.semilogy(net_iters,bitwise_ber_n2c, label='Bit {0} n2c'.format(i))
                plt.semilogy(net_iters,bitwise_ber_c2n, label='Bit {0} c2n'.format(i))
                if len(ber_scratch) < len(net_iters):
                    plt.semilogy(net_iters[:len(bitwise_ber_scr)],bitwise_ber_scr, label='Bit {0} scr'.format(i))
                else:
                    plt.semilogy(net_iters,bitwise_ber_scr[:len(net_iters)], label='Bit {0} scr'.format(i))
                sc = np.ones(len(net_iters)) * bers_SC_test_bitwise[0][i].item()
                plt.semilogy(net_iters,sc, label='SC bit {0}'.format(i),linestyle='dashed')
                plt.legend(prop={'size': 7},loc='upper right', bbox_to_anchor=(1.1, 1))
                plt.ylim(bottom=1e-3)
                plt.ylim(top=0.6)
                plt.title("Hardest to easiest order : {0}".format(np.argsort(np.argsort(info_inds1))))
                plt.savefig(results_save_path +'/z_progressive_bitwise_{0}.pdf'.format(i))
                plt.close()
            sys.exit()
        
        times = []
        results_load_path = final_save_path
        #print(results_load_path)
        if args.model_iters is not None:
            checkpoint1 = torch.load(results_save_path +'/Models/model_{0}.pt'.format(args.model_iters), map_location=lambda storage, loc: storage)
        elif args.test_load_path is not None:
            checkpoint1 = torch.load(args.test_load_path , map_location=lambda storage, loc: storage)
        else:
            checkpoint1 = torch.load(results_load_path +'/Models/model_final.pt', map_location=lambda storage, loc: storage)
            try:
                args.model_iters = i_step + 1
            except:
                pass
    
        #print(checkpoint1)
        loaded_step = checkpoint1['step']
        xformer.load_state_dict(checkpoint1['xformer'])
        xformer.to(device)
        print("Model loaded at step {}".format(loaded_step))
    
        xformer.eval()
    
        if args.snr_points == 1 and args.test_snr_start == args.test_snr_end:
            snr_range = [args.test_snr_start]
        else:
            snrs_interval = (args.test_snr_end - args.test_snr_start)* 1.0 /  (args.snr_points-1)
            snr_range = [snrs_interval* item + args.test_snr_start for item in range(args.snr_points)]
    
        Test_msg_bits = 2 * (torch.rand(args.test_size, args.K) < 0.5).float() - 1
        Test_Data_Mask = torch.ones(Test_msg_bits.size(),device=device).long()
    
        Test_Data_Generator = torch.utils.data.DataLoader(Test_msg_bits, batch_size=args.test_batch_size , shuffle=False, **kwargs)
        Test_Data_Mask =  torch.utils.data.DataLoader(Test_Data_Mask, batch_size=args.test_batch_size , shuffle=False, **kwargs)
        num_test_batches = len(Test_Data_Generator)
    
    
    
        ######
        ### MAP decoding stuff
        ######
    
        if args.are_we_doing_ML and args.code=='polar':
            all_msg_bits = []
            for i in range(2**args.K):
                d = dec2bitarray(i, args.K)
                all_msg_bits.append(d)
            all_message_bits = torch.from_numpy(np.array(all_msg_bits))
            all_message_bits = 1 - 2*all_message_bits.float()
            codebook = polar.encode_plotkin(all_message_bits)
            b_codebook = codebook.repeat(args.test_batch_size, 1, 1).to(device)
        if args.are_we_doing_ML and args.code=='pac':
            all_msg_bits = []
            for i in range(2**args.K):
                d = dec2bitarray(i, args.K)
                all_msg_bits.append(d)
            all_message_bits = torch.from_numpy(np.array(all_msg_bits)).to(device)
            all_message_bits = 1 - 2*all_message_bits.float()
            codebook = polar.pac_encode(all_message_bits, scheme = args.rate_profile)
            b_codebook = codebook.repeat(args.test_batch_size, 1, 1)
    
        start_time = time.time()
    
        if args.code == 'polar':
            bers_Xformer_test, blers_Xformer_test, bers_SC_test, blers_SC_test,bers_SCL_test, blers_SCL_test, bers_ML_test, blers_ML_test, bers_bitwise_Xformer_test,bers_bitwise_MAP_test, blers_bitwise_MAP_test = testXformer(xformer, polar, snr_range, Test_Data_Generator, device, run_ML=args.are_we_doing_ML)
            print("Test SNRs : ", snr_range)
            print("BERs of Xformer: {0}".format(bers_Xformer_test))
            print("BERs of SC decoding: {0}".format(bers_SC_test))
            print("BERs of ML: {0}".format(bers_ML_test))
            print("BLERs of ML: {0}".format(blers_ML_test))
            print("BERs of bitML: {0}".format(bers_bitwise_MAP_test))
            print("BLERs of bitML: {0}".format(blers_bitwise_MAP_test))
            print("BLERs of Xformer: {0}".format(blers_Xformer_test))
            print("Time taken = {} seconds".format(time.time() - start_time))
            ## BER
            plt.figure(figsize = (12,8))
            print(bers_bitwise_Xformer_test)
            ok = 0
            plt.semilogy(snr_range, bers_Xformer_test, label="Xformer decoder", marker='*', linewidth=1.5)
            plt.semilogy(snr_range, bers_SC_test, label="SC decoder", marker='^', linewidth=1.5)
            plt.semilogy(snr_range, bers_SCL_test, label="SCL decoder", marker='^', linewidth=1.5)
    
            if args.are_we_doing_ML:
                plt.semilogy(snr_range, bers_ML_test, label="ML decoder", marker='o', linewidth=1.5)
                plt.semilogy(snr_range, bers_bitwise_MAP_test, label="Bitwise ML decoder", marker='o', linewidth=1.5)
            # if args.run_fano:
            ## BLER
            plt.semilogy(snr_range, blers_Xformer_test, label="Xformer decoder (BLER)", marker='*', linewidth=1.5, linestyle='dashed')
            plt.semilogy(snr_range, blers_SC_test, label="SC decoder (BLER)", marker='^', linewidth=1.5, linestyle='dashed')
            plt.semilogy(snr_range, blers_SCL_test, label="SCL decoder (BLER)", marker='^', linewidth=1.5, linestyle='dashed')
    
            if args.are_we_doing_ML:
                plt.semilogy(snr_range, blers_ML_test, label="ML decoder", marker='o', linewidth=1.5, linestyle='dashed')
                plt.semilogy(snr_range, blers_bitwise_MAP_test, label="Bitwise ML decoder", marker='o', linewidth=1.5, linestyle='dashed')
            # if args.run_fano:
    
            plt.grid()
            plt.xlabel("SNR (dB)", fontsize=16)
            plt.ylabel("Error Rate", fontsize=16)
            if args.rate_profile == 'polar':
                plt.title("Polar({1}, {2}): Xformer trained at Dec_SNR = {0} dB".format(args.dec_train_snr, args.K,args.N))
            elif args.rate_profile == 'RM':
                plt.title("RM({1}, {2}): Xformer trained at Dec_SNR = {0} dB".format(args.dec_train_snr, args.K,args.N))
    
            plt.legend(prop={'size': 15})
            if args.test_load_path is not None:
                os.makedirs('Xformer_Polar_Results/figures', exist_ok=True)
                fig_save_path = 'Xformer_Polar_Results/figures/new_plot.pdf'
            else:
                fig_save_path = results_load_path + "/step_{}.pdf".format(args.model_iters if args.model_iters is not None else '_final')
            plt.savefig(fig_save_path)
    
            plt.close()
        elif args.code == 'pac':
            plot_fano = False
            fano_path = './data/pac/fano/Scheme_{3}/N{0}_K{1}_g{2}.p'.format(args.N, args.K, args.g, args.rate_profile)
            test_data_path = './data/pac/test/Scheme_{3}/test_N{0}_K{1}_g{2}.p'.format(args.N, args.K, args.g, args.rate_profile)
            if os.path.exists(fano_path):
                fanos = pickle.load(open(fano_path, 'rb'))
                snr_range_fano = fanos[0]
                bers_fano_test = fanos[1]
                blers_fano_test = fanos[2]
                run_fano = False
                plot_fano = True
            else:
                snr_range_fano = snr_range
                bers_fano_test = []
                blers_fano_test = []
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
                bers_RNN_test, blers_RNN_test, bers_Dumer_test, blers_Dumer_test, bers_ML_test, blers_ML_test, bers_fano_temp, blers_fano_temp = test_full_data(xformer,polar, snr_range, Test_Data_Generator, run_fano = args.run_fano, run_dumer = args.run_dumer)
            else:
                print("Testing on the standard data")
                msg_bits = test_dict['msg']
                received = test_dict['rec']
                snr_range = list(received.keys())
                print(snr_range)
                bers_RNN_test, blers_RNN_test, bers_Dumer_test, blers_Dumer_test, bers_ML_test, blers_ML_test, bers_fano_temp, blers_fano_temp,bers_bitwise_Xformer_test = test_standard(xformer,polar, msg_bits, received, run_fano = args.run_fano, run_dumer = args.run_dumer)
    
            if not os.path.exists(fano_path):
                bers_fano_test = bers_fano_temp
                blers_fano_test = blers_fano_temp
                snr_range_fano = snr_range
    
            
            try:
                print(bers_bitwise_Xformer_test)
            except:
                pass
            print("Test SNRs : ", snr_range)
            print("BERs of Xformer: {0}".format(bers_RNN_test))
            
            print("BERs of SC decoding: {0}".format(bers_Dumer_test))
            print("BERs of ML: {0}".format(bers_ML_test))
            print("BERs of Fano: {0}".format(bers_fano_test))
            print("BLERs of Xformer: {0}".format(blers_RNN_test))
            print("Time taken = {} seconds".format(time.time() - start_time))
            ## BER
            plt.figure(figsize = (12,8))
    
            ok = 0
            plt.semilogy(snr_range, bers_RNN_test, label="Xformer decoder", marker='*', linewidth=1.5)
    
            if args.run_dumer:
                plt.semilogy(snr_range, bers_Dumer_test, label="SC decoder", marker='^', linewidth=1.5)
    
            if args.are_we_doing_ML:
                plt.semilogy(snr_range, bers_ML_test, label="ML decoder", marker='o', linewidth=1.5)
            if plot_fano:
                plt.semilogy(snr_range_fano, bers_fano_test, label="Fano decoder", marker='P', linewidth=1.5)
    
            ## BLER
            plt.semilogy(snr_range, blers_RNN_test, label="Xformer decoder (BLER)", marker='*', linewidth=1.5, linestyle='dashed')
            if args.run_dumer:
                plt.semilogy(snr_range, blers_Dumer_test, label="SC decoder (BLER)", marker='^', linewidth=1.5, linestyle='dashed')
    
            if args.are_we_doing_ML:
                plt.semilogy(snr_range, blers_ML_test, label="ML decoder", marker='o', linewidth=1.5, linestyle='dashed')
            if plot_fano:
                plt.semilogy(snr_range_fano, blers_fano_test, label="Fano decoder", marker='P', linewidth=1.5, linestyle='dashed')
    
            plt.grid()
            plt.xlabel("SNR (dB)", fontsize=16)
            plt.ylabel("Error Rate", fontsize=16)
            plt.title("PAC({1}, {2}): Xformer trained at Dec_SNR = {0} dB".format(args.dec_train_snr, args.K,args.N))
            plt.legend(prop={'size': 15})
            if args.test_load_path is not None:
                os.makedirs('Xformer_PAC_Results/figures', exist_ok=True)
                fig_save_path = 'Xformer_PAC_Results/figures/new_plot.pdf'
            else:
                fig_save_path = results_load_path + "/step_{}.pdf".format(args.model_iters if args.model_iters is not None else '_final')
            plt.savefig(fig_save_path)
    
            plt.close()
