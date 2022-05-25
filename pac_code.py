import numpy as np
import torch
# import commpy.channelcoding as cc

import matplotlib.pyplot as plt
import pickle
import os
import argparse
import time

from polar import PolarCode
from utils import errors_ber, errors_bler, moving_average, get_msg_bits_batch, snr_db2sigma, log_sum_exp, log_sum_avoid_zero_NaN, Clamp, STESign, STEQuantize, new_log_sum, new_log_sum_avoid_zero_NaN, min_sum_log_sum_exp

def get_args():
    parser = argparse.ArgumentParser(description='(N,K) Polar code')

    parser.add_argument('-N', type=int, default=128, help='Polar code parameter N')
    parser.add_argument('-K', type=int, default=64, help='Polar code parameter K')
    parser.add_argument('-batch_size', type=int, default=1000, help='number of blocks')
    parser.add_argument('-test_ratio', type = float, default = 1, help = 'Number of test samples x batch_size')
    parser.add_argument('-test_snr_start', type=float, default=0., help='test snr start')
    parser.add_argument('-test_snr_end', type=float, default=10., help='test snr end')
    parser.add_argument('-id', type=int, default=100000)
    parser.add_argument('--hard_decision', dest = 'hard_decision', default=False, action='store_true', help='polar code sc decoding hard decision?')
    parser.add_argument('-soft_sign', type=str, default='tanh', choices=['tanh', 'STE'], help='type of differentiable operator for torch.sign')

    parser.add_argument('-g', type=int, default=91, help='Convolutional code generator polynomial')
    parser.add_argument('-rate_profile', type=str, default='RM', choices=['RM', 'polar', 'sorted', 'last', 'custom'], help='rate profiling scheme')
    parser.add_argument('-delta', type=int, default=2, help='Fano decoding threshold update delta')
    parser.add_argument('-bias', type=float, default=1.35, help='Fano metric bias term')
    parser.add_argument('-bias_frozen', type=float, default=0, help='Fano metric bias term - frozen positions')
    parser.add_argument('-bias_type', type=str, default='p_e', choices=['constant', 'p_e'], help='type of Fano bias')
    parser.add_argument('-maxd', type=float, default=5, help='Don\'t explore paths which are more bits different from SC path')
    parser.add_argument('-verbose', type=float, default=0, help='Verbose level 0/1/2/3')
    parser.add_argument('-printf', type=int, default=100, help='Verbose level 0/1/2/3')


    args = parser.parse_args()
    return args

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

def bitarray2dec(in_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.
    Parameters
    ----------
    in_bitarray : 1D ndarray of ints
        Input NumPy array of bits.
    Returns
    -------
    number : int
        Integer representation of input bit array.
    """

    number = 0

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i]*pow(2, len(in_bitarray)-1-i)

    return number

def countSetBits(n):

    count = 0
    while (n):
        n &= (n-1)
        count+= 1

    return count

class PAC():
    # For this implementation, we assume all inputs are in BPSK. Convention : 0 -> +1, 1 -> -1
    # With this convention, xor operation is the Hadamard product
    def __init__(self, args, N, K, g, infty = 1000., rate_profile='RM'):
        self.N = N
        self.n = int(np.log2(N))
        self.K = K
        self.args = args
        M = int(np.floor(np.log2(g))) + 1
        self.g_array = 1 - 2*dec2bitarray(g, M)
        self.rate_profile = rate_profile

        clamp_class = Clamp()
        self.clamp = clamp_class.apply

        # ste_class = STESign()
        ste_class = STEQuantize()
        self.ste_sign = ste_class.apply
        self.infty = infty
        # self.infty = float('inf')

        rmweight = np.array([countSetBits(i) for i in range(self.N)])
        B = np.argsort(rmweight)[-self.K:]
        self.unsorted_info_positions = np.argsort(rmweight)[-self.K:]
        B.sort()
        self.B = B

    def rate_profiler(self, msg_bits, scheme='RM',custom_info_positions = None):
        if custom_info_positions is None:
            if scheme is not None:
                assert scheme in ['RM','rev_RM', 'polar', 'sorted', 'sorted_last', 'last', 'custom', 'freeze_even', 'freeze_odd'], "Invalid rate profiler choice"
            else:
                scheme = self.rate_profile
            try:
                target_K = self.args.target_K
            except:
                target_K = self.N//2
            # Compute a set B such that u[B] = message , u[B^c] = 0
            if scheme == 'RM':
                rmweight = np.array([countSetBits(i) for i in range(self.N)])
                B = np.argsort(rmweight)[-self.K:]
                B.sort()
            if scheme == 'rev_RM':
                rmweight = np.array([countSetBits(i) for i in range(self.N)])
                first_half = np.argsort(rmweight)[-target_K:]
                B = first_half[:self.K].copy()
                B.sort()
            elif scheme == 'polar':
                rs = pickle.load(open('data/polar/rs{}.p'.format(self.N),'rb'))
                B = rs[:self.K].copy()
                B.sort()
            elif scheme == 'sorted':
                rmweight = np.array([countSetBits(i) for i in range(self.N)])
                B = np.argsort(rmweight)[-int(target_K):]
                B.sort()
                B = B[:self.K].copy()
            elif scheme == 'sorted_last':
                rmweight = np.array([countSetBits(i) for i in range(self.N)])
                B = np.argsort(rmweight)[-int(target_K):]
                B.sort()
                B = B[-self.K:].copy()
            elif scheme == 'last':
                B = np.arange(self.N-1, self.N - self.K - 1, -1)
                B.sort()
            elif scheme == 'custom': # save np array with ascending order of priority to be info bit, as a pickle file
                order = pickle.load(open('data/pac/rate_profile/pac{}.p'.format(self.N),'rb'))
                B = order[-self.K:].copy()
                B.sort()
            elif scheme == 'freeze_even':
                B = np.arange(self.N-1, -1, -2)
                B.sort()
            elif scheme == 'freeze_odd':
                B = np.arange(self.N-2, -1, -2)
                B.sort()
        else:
            B = custom_info_positions.copy()
            B.sort()

        u = torch.ones((msg_bits.shape[0], self.N), dtype=torch.float, device=msg_bits.device)
        u[:,B] = msg_bits

        self.B = B
        return u

    def convTrans(self, v, g):
        cState = torch.ones(len(g)-1).to(v.device)
        u = torch.ones_like(v, dtype=torch.float)
        for i in range(0, len(v)):
            u[i], cState = self.conv1bTrans(v[i], cState, g)
        return u

    def conv1bTrans(self, v, currState, g):
        u = v*(0.5*(1 - g[0]))
        for j in range(1, len(g)):
            if g[j] == -1:
                u = u * currState[j-1]
        nextState = torch.cat((torch.Tensor([v]).to(currState.device), currState[:-1]))
        return u, nextState

    def conv1bTrans_batch(self, v, currState, g):
        u = v*(0.5*(1 - g[0]))
        for j in range(1, len(g)):
            if g[j] == -1:
                u = u * currState[:, j-1]
        nextState = torch.cat((v.unsqueeze(1).to(currState.device), currState[:, :-1]), dim=1)

        return u, nextState

    def convolutional_encode(self, v):
        g = self.g_array
        cState = torch.ones(v.shape[0], len(g)-1).to(v.device)
        u = torch.ones_like(v, dtype=torch.float)
        for i in range(0, v.shape[1]):
            u[:, i], cState = self.conv1bTrans_batch(v[:, i], cState, g)
        return u

    def polar_encode(self, msg_bits):
        # Plotkin polar encoding - assuming rate = 1
        u = msg_bits
        for d in range(0, self.n):
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [u v] encoded to [(u xor v),v)]
                u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
        return u

    def pac_encode(self, msg_bits, scheme=None,custom_info_positions=None):
        v = self.rate_profiler(msg_bits, scheme, custom_info_positions = custom_info_positions)
        u = self.convolutional_encode(v)
        x = self.polar_encode(u)
        return x

    def channel(self, code, snr):
        sigma = snr_db2sigma(snr)

        noise = (sigma* torch.randn(code.shape, dtype = torch.float)).to(code.device)
        noisy_code = code + noise
        return noisy_code

    def define_partial_arrays(self, llrs):
        # Initialize arrays to store llrs and partial_sums useful to compute the partial successive cancellation process.
        llr_array = torch.zeros(llrs.shape[0], self.n+1, self.N, device=llrs.device)
        llr_array[:, self.n] = llrs
        partial_sums = torch.zeros(llrs.shape[0], self.n+1, self.N, device=llrs.device)
        return llr_array, partial_sums


    def updateLLR(self, leaf_position, llrs, partial_sums = None):

        #START
        depth = self.n
        decoded_bits = partial_sums[:,0].clone()
        llrs, partial_sums, decoded_bits = self.partial_decode(llrs, partial_sums, depth, 0, leaf_position, decoded_bits)
        return llrs, decoded_bits

    def updatePartialSums(self, leaf_position, decoded_bits, partial_sums):

        u = decoded_bits.clone()
        u[:, leaf_position+1:] = 0


        for d in range(0, self.n):
            partial_sums[:, d] = u
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [u v] encoded to [u xor(u,v)]
                u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
        partial_sums[:, self.n] = u
        return partial_sums


    def partial_decode(self, llrs, partial_sums, depth, bit_position, leaf_position, decoded_bits=None):
        # Function to call recursively, for partial SC decoder.
        # We are assuming that u_0, u_1, .... , u_{leaf_position -1} bits are known.
        # Partial sums computes the sums got through Plotkin encoding operations of known bits, to avoid recomputation.
        # this function is implemented for rate 1 (not accounting for frozen bits in polar SC decoding)

        # print("DEPTH = {}, bit_position = {}".format(depth, bit_position))
        half_index = 2 ** (depth - 1)
        leaf_position_at_depth = leaf_position // 2**(depth-1) # will tell us whether left_child or right_child

        # n = 2 tree case
        if depth == 1:
            # Left child
            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                u_hat = partial_sums[:, depth-1, left_bit_position:left_bit_position+1]
            elif leaf_position_at_depth == left_bit_position:
                if False: #left_bit_position in self.frozen_positions: #NEED TO CHANGE
                    # If frozen decoded bit is 0
                    u_hat = torch.ones_like(llrs[:, :half_index], dtype=torch.float)
                else:
                    Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                    # Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)

                    llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu
                    if self.args.hard_decision:
                        u_hat = torch.sign(Lu)
                    else:
                        u_hat = torch.tanh(Lu/2)

                    decoded_bits[:, left_bit_position] = u_hat.squeeze(1)

                    return llrs, partial_sums, decoded_bits

            # Right child
            right_bit_position = 2*bit_position + 1
            if leaf_position_at_depth > right_bit_position:
                pass
            elif leaf_position_at_depth == right_bit_position:
                if False:#right_bit_position in self.frozen_positions: #NEED TO CHANGE
                    # If frozen decoded bit is 0
                    v_hat = torch.ones_like(llrs[:, :half_index], dtype = torch.float)
                else:
                    Lv = u_hat * llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index] + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
                    llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv
                    if self.args.hard_decision:
                        v_hat = torch.sign(Lv)
                    else:
                        v_hat = torch.tanh(Lv/2)
                    decoded_bits[:, right_bit_position] = v_hat.squeeze(1)
                    return llrs, partial_sums, decoded_bits


        # General case
        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u
            # Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))

            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                Lu = llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                u_hat = partial_sums[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
            else:

                # Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])
                Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])

                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu
                llrs, partial_sums, decoded_bits = self.partial_decode(llrs, partial_sums, depth-1, left_bit_position, leaf_position, decoded_bits)

                return llrs, partial_sums, decoded_bits

            # RIGHT CHILD
            right_bit_position = 2*bit_position + 1

            Lv = u_hat * llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index] + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
            llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv
            llrs, partial_sums, decoded_bits = self.partial_decode(llrs, partial_sums, depth-1, right_bit_position, leaf_position, decoded_bits)

            return llrs, partial_sums, decoded_bits

    def get_metric(self, position, u_hat, llrs, u=None):
        L = llrs[0,0][position]
        metric = torch.log2(torch.sigmoid(u_hat*L))
        return metric

    def fano_decode(self, llrs, delta = 2, bias = 1.35, bias_frozen = 0, verbose = 0, maxDiversions = 5, bias_type = 'constant'):

        if bias_type == 'p_e':
            p_es = torch.load('data/pac/pe_{}.p'.format(self.N))
            biases = torch.log2(1 - p_es)
            bias_sum = torch.sum(biases)
        cState = torch.ones(len(self.g_array)-1)
        currState = torch.ones(self.K, len(self.g_array)-1)
        ii = 0
        j = 0
        threshold = 0
        visited = [[] for i in range(self.N)]

        metrics = torch.zeros(self.N)
        llr_array, partial_sums = self.define_partial_arrays(llrs.unsqueeze(0))
        # deltas = torch.zeros(self.N)
        t = torch.zeros(self.N, dtype=torch.int)

        u_hat = torch.zeros(1, self.N)
        v_hat = torch.zeros(1, self.N)
        onMainPath = True
        isBackTracking = False
        toDiverge = False
        # maxDiversions = 5

        biasUpdated = False

        path_metrics = torch.zeros(self.N)
        state_along_path = {} # (path_metric, P, v_hat, u_hat, cState, llr_array, partial_sums)
        state_along_path[-1] = (-float('inf'), [0, 0], 0, 0, None)

        num_visits = 0
        while ii < self.N:
            num_visits += 1
            if verbose > 1:
                print(ii)
                # print('u: {}, v: {}'.format(u_hat[0], v_hat[0]))
                print('Path metrics : {}, Threshold = {}'.format(path_metrics, threshold))
                print('State_along_path = {}'.format(state_along_path))
                print('t', t, '\n')
            if not isBackTracking:
                llr_array , decoded_bits = self.updateLLR(ii, llr_array.clone(), partial_sums)
            if verbose == 3:
                print('llr_array : {}'.format(llr_array[0]))
                print('partial_sums : {}'.format(partial_sums[0]))

            if ii not in self.B: #frozen
                v_hat[:, ii] = 1 #zero
                u_hat[:, ii], cState = self.conv1bTrans(1, cState, self.g_array)
                if ii>0:
                    # path_metrics[ii] = path_metrics[ii-1] + self.get_metric(ii, u_hat[:, ii], llr_array, u = u_hat[0]) - bias
                    if bias_type == 'p_e':
                        path_metrics[ii] = path_metrics[ii-1] + self.get_metric(ii, u_hat[:, ii], llr_array) - biases[ii]
                    else:
                        path_metrics[ii] = path_metrics[ii-1] + self.get_metric(ii, u_hat[:, ii], llr_array) - bias_frozen
                    # path_metrics[ii] = path_metrics[ii-1] + self.get_metric(ii, u_hat[:, :ii+1], llr_array)
                else:
                    # path_metrics[ii] = self.get_metric(ii, u_hat[:, ii], llr_array, u = u_hat[0]) - bias
                    if bias_type == 'p_e':
                        path_metrics[ii] = self.get_metric(ii, u_hat[:, ii], llr_array) + bias_sum
                    else:
                        path_metrics[ii] = self.get_metric(ii, u_hat[:, ii], llr_array) - bias_frozen
                    # path_metrics[ii] = self.get_metric(ii, u_hat[:, :ii+1], llr_array)
                partial_sums = self.updatePartialSums(ii, u_hat, partial_sums)
                ii += 1
                if ii< self.N:
                    t[ii] = 0
            else:
                ind = (self.B == ii).nonzero()[0][0]

                if isBackTracking is False and (ind not in state_along_path.keys()):
                    # recalculate if values not stored in state_along_path
                    u0, cState0 = self.conv1bTrans(1, cState, self.g_array)
                    u1, cState1 = self.conv1bTrans(-1, cState, self.g_array)

                    if bias_type == 'p_e':
                        pm0 = path_metrics[ii-1] + self.get_metric(ii, u0, llr_array.clone()) - biases[ii]
                        pm1 = path_metrics[ii-1] + self.get_metric(ii, u1, llr_array.clone()) - biases[ii]
                    else:
                        pm0 = path_metrics[ii-1] + self.get_metric(ii, u0, llr_array.clone()) - bias
                        pm1 = path_metrics[ii-1] + self.get_metric(ii, u1, llr_array.clone()) - bias
                    # pm0 = path_metrics[ii-1] + self.get_metric(ii, uhat0, llr_array.clone()) - bias
                    # pm1 = path_metrics[ii-1] + self.get_metric(ii, uhat1, llr_array.clone()) - bias

                    P = [(pm0, 1, u0, cState0), (pm1, -1, u1, cState1)]
                else:
                    P = state_along_path[ind][1]
                    cState = state_along_path[ind][4]
                    v_hat = state_along_path[ind][2]
                    u_hat = state_along_path[ind][3]
                    pm0, _, u0, cState0 = P[0]
                    pm1, _, u1, cState1 = P[1]
                metric_order = np.argsort([r.item() for r in [pm0,pm1]])[::-1]

                current_bit = metric_order[t[ii]]
                pm_max = P[current_bit][0]
                state_along_path[ind] = (pm_max, P, v_hat, u_hat, cState)

                if pm_max >= threshold:

                    path_metrics[ii], v_hat[:, ii], u_hat[:, ii], cState = P[current_bit]

                    str1 = ""
                    for i in v_hat[0, :ii+1]:
                        str1 += str(int(0.5 - 0.5*i.item()))
                    if str1 not in visited[ii]: # if first visit
                        # tighten threshold

                        threshold = threshold + delta * (pm_max // threshold)
                        # print('Tightening threshold at {}: threshold = {}'.format(str1, threshold))
                        visited[ii].append(str1) # add path to visited list


                    # path_metrics[i] = P[max_bit][0]
                    partial_sums = self.updatePartialSums(ii, u_hat, partial_sums)
                    ii += 1
                    if ii< self.N:
                        t[ii] = 0
                    if verbose >= 1:
                        print("Moved forward to {}. Threshold = {}".format(str1, threshold))
                    # state_along_path[ind] = (pm_max, P, v_hat, u_hat, cState)
                    isBackTracking = False

                else:
                    # check if previous node metric is less than threshold
                    while True:
                        if state_along_path[ind-1][0] < threshold:
                            #reduce threshold
                            threshold = threshold - delta
                            t[ii] = 0
                            str1 = ""
                            for i in v_hat[0, :ii]:
                                str1 += str(int(0.5 - 0.5*i.item()))
                            if verbose >= 1:
                                print("Reduced threshold at {}, to {}".format(str1, threshold))
                                isBackTracking = False
                            break
                        else:

                            _ = state_along_path.pop(ind)
                            ind = ind - 1
                            ii = self.B[ind]
                            # v_hat, u_hat, path_metrics = v_hat[:ii+1], u_hat[:ii+1], path_metrics[:ii+1]
                            # v_hat[:, ii+1:] = 0
                            # u_hat[:, ii+1:] = 0
                            # path_metrics[ii+1] = 0
                            v_hat[:, ii:] = 0
                            u_hat[:, ii:] = 0
                            path_metrics[ii] = 0

                            t[ii] = t[ii] + 1
                            if t[ii] == 2:
                                if verbose >= 1:
                                    print("Both branches explored")
                                t[ii] = 0
                                continue
                            elif t[:ii+1].sum() > maxDiversions:
                                if verbose >= 1:
                                    print("MaxDiversions, hence skipping path")
                                t[ii] = 0
                                continue
                            else:
                                str1 = ""
                                for i in v_hat[0, :ii]:
                                    str1 += str(int(0.5 - 0.5*i.item()))
                                if verbose >= 1:
                                    print("Moved back to {}".format(str1))
                                partial_sums = self.updatePartialSums(ii, u_hat, partial_sums)
                                isBackTracking = True
                                break
        str1 = ""
        for i in v_hat[0]:
            str1 += str(int(0.5 - 0.5*i.item()))
        if verbose >= 1:
            print("Ended at {}. Number of node visits = {}\n".format(str1, num_visits))
        return v_hat, path_metrics

    def extract(self, v_hat, B = None):
        if B is None:
            B = self.B
        return v_hat[:, B]

    def pac_sc_decode(self, corrupted_codewords, snr, use_gt_codeword = None):

        sigma = snr_db2sigma(snr)
        llrs = (2/sigma**2)*corrupted_codewords

        u_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        v_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)

        cState = torch.ones(corrupted_codewords.shape[0], len(self.g_array)-1, device=corrupted_codewords.device) #convolutional encoder state
        llr_array, partial_sums = self.define_partial_arrays(llrs)
        for ii in range(self.N):
            llr_array , decoded_bits = self.updateLLR(ii, llr_array.clone(), partial_sums)
            if ii not in self.B: #frozen
                v_hat[:, ii] = 1 #zero
                if use_gt_codeword is not None:
                    u_hat[:, ii] = use_gt_codeword[:, ii]
                else:
                    u_hat[:, ii], cState = self.conv1bTrans_batch(torch.ones_like(u_hat[:, ii]), cState, self.g_array)

            else: # non-frozen
                if use_gt_codeword is not None:
                    u_hat[:, ii] = use_gt_codeword[:, ii]
                else:
                    u_hat[:, ii] = torch.sign(llr_array[:, 0, ii])
                u0, cState0 = self.conv1bTrans_batch(torch.ones_like(u_hat[:, ii]), cState, self.g_array)
                u1, cState1 = self.conv1bTrans_batch(-1*torch.ones_like(u_hat[:, ii]), cState, self.g_array)

                z_inds =  u0 == u_hat[:, ii]
                o_inds = u1 == u_hat[:, ii]

                v_hat[z_inds, ii] = 1.
                cState[z_inds] = cState0[z_inds]

                v_hat[o_inds, ii] = -1.
                cState[o_inds] = cState1[o_inds]

            partial_sums = self.updatePartialSums(ii, u_hat, partial_sums)

        decoded_bits = v_hat[:, self.B]
        return llr_array[:, 0], decoded_bits, u_hat

    def pac_sc_decode_diff(self, corrupted_codewords, snr):
        # for g=5
        sigma = snr_db2sigma(snr)
        llrs = (2/sigma**2)*corrupted_codewords

        u_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        v_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)

        cState = torch.ones(corrupted_codewords.shape[0], len(self.g_array)-1, device=corrupted_codewords.device) #convolutional encoder state
        llr_array, partial_sums = self.define_partial_arrays(llrs)
        for ii in range(self.N):
            llr_array , decoded_bits = self.updateLLR(ii, llr_array.clone(), partial_sums)
            if ii not in self.B: #frozen
                v_hat[:, ii] = 1 #zero
                u_hat[:, ii], cState = self.conv1bTrans_batch(torch.ones_like(u_hat[:, ii]), cState, self.g_array)

            else: # non-frozen
                if self.args.soft_sign == 'tanh':
                    u_hat[:, ii] = torch.tanh(llr_array[:, 0, ii]/2)
                elif self.args.soft_sign == 'STE':
                    u_hat[:, ii] = self.ste_sign(llr_array[:, 0, ii])
                v_hat[:, ii] = u_hat[:, ii].clone()
                jj = ii - 2
                while(jj>=0):
                    v_hat[:, ii] = v_hat[:, ii] * u_hat[:, jj]
                    jj = jj-2

                u_hat[:, ii], cState = self.conv1bTrans_batch(v_hat[:, ii], cState, self.g_array)

            partial_sums = self.updatePartialSums(ii, u_hat, partial_sums)
        decoded_llrs = v_hat[:, self.B]
        decoded_bits = torch.sign(decoded_llrs)
        return llr_array[:, 0], decoded_bits, u_hat

    def pac_sc_decode_new(self, corrupted_codewords, snr):
        # for g=5
        sigma = snr_db2sigma(snr)
        llrs = (2/sigma**2)*corrupted_codewords

        u_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        v_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        v_llrs = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        u_llrs = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)

        cState = torch.ones(corrupted_codewords.shape[0], len(self.g_array)-1, device=corrupted_codewords.device) #convolutional encoder state
        llr_array, partial_sums = self.define_partial_arrays(llrs)
        for ii in range(self.N):
            llr_array , decoded_bits = self.updateLLR(ii, llr_array.clone(), partial_sums)
            if ii not in self.B: #frozen
                v_hat[:, ii] = 1 #zero
                v_llrs[:, ii] = self.infty
                # u_hat[:, ii], cState = self.conv1bTrans_batch(torch.ones_like(u_hat[:, ii]), cState, self.g_array)
                if ii < 2:
                    u_hat[:, ii] = v_hat[:, ii]
                else:
                    u_hat[:, ii] = v_hat[:, ii] * v_hat[:, ii-2]
                u_llrs[:, ii] = llr_array[:, 0, ii]
            else: # non-frozen
                u_llrs[:, ii] = llr_array[:, 0, ii]
                u_hat[:, ii] = torch.sign(u_llrs[:, ii])

                # if self.args.soft_sign == 'tanh':
                #     u_hat[:, ii] = torch.tanh(llr_array[:, 0, ii]/2)
                # elif self.args.soft_sign == 'STE':
                #     u_hat[:, ii] = self.ste_sign(llr_array[:, 0, ii])

                jj = ii - 2
                v_llrs[:, ii] = min_sum_log_sum_exp(u_llrs[:, ii].clone(), v_llrs[:, jj].clone())
                v_hat[:, ii] = torch.sign(v_llrs[:, ii])



            partial_sums = self.updatePartialSums(ii, u_hat, partial_sums)
        decoded_llrs = v_llrs[:, self.B]
        decoded_bits = torch.sign(decoded_llrs)
        return decoded_llrs, decoded_bits, u_hat

    def updateLLR_soft(self, leaf_position, llrs, partial_llrs, prior = None):

        #START
        depth = self.n
        decoded_bits = partial_llrs[:,0].clone()
        if prior is None:
            prior = torch.zeros(llrs.shape[0], self.args.N) #priors
        llrs, partial_llrs, decoded_bits = self.partial_decode_soft(llrs, partial_llrs, depth, 0, leaf_position, prior, decoded_bits)
        return llrs, decoded_bits


    def partial_decode_soft(self, llrs, partial_llrs, depth, bit_position, leaf_position, prior, decoded_bits=None):
        # Function to call recursively, for partial SC decoder.
        # We are assuming that u_0, u_1, .... , u_{leaf_position -1} bits are known.
        # Partial sums computes the sums got through Plotkin encoding operations of known bits, to avoid recomputation.
        # this function is implemented for rate 1 (not accounting for frozen bits in polar SC decoding)

        # print("DEPTH = {}, bit_position = {}".format(depth, bit_position))
        half_index = 2 ** (depth - 1)
        leaf_position_at_depth = leaf_position // 2**(depth-1) # will tell us whether left_child or right_child

        # n = 2 tree case
        if depth == 1:
            # Left child
            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                Lu = llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                L_u = partial_llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                #L_uv = log_sum_avoid_zero_NaN(L_u, llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index])
            elif leaf_position_at_depth == left_bit_position:
                Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index].clone(), llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index].clone()).sum(dim=1, keepdim=True)
                #Lu = self.clamp(Lu + prior[:, left_bit_position].unsqueeze(-1), -self.infty, self.infty)
                Lu = Lu + prior[:, left_bit_position].unsqueeze(-1)
                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu
                if self.args.hard_decision:
                    u_hat = torch.sign(Lu)
                else:
                    if self.args.soft_sign == 'tanh':
                        u_hat = torch.tanh(Lu/2)
                    elif self.args.soft_sign == 'STE':
                        u_hat = self.ste_sign(Lu)
                decoded_bits[:, left_bit_position] = u_hat.squeeze(1)

                return llrs, partial_llrs, decoded_bits

            # Right child
            right_bit_position = 2*bit_position + 1
            if leaf_position_at_depth > right_bit_position:
                pass
            elif leaf_position_at_depth == right_bit_position:
                L_uv = min_sum_log_sum_exp(L_u.clone(), llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index].clone())
                Lv = L_uv + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
                #Lv = self.clamp(Lv + prior[:, right_bit_position].unsqueeze(-1), -self.infty, self.infty)
                Lv = Lv + prior[:, right_bit_position].unsqueeze(-1)

                llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv
                if self.args.hard_decision:
                    v_hat = torch.sign(Lv)
                else:
                    if self.args.soft_sign == 'tanh':
                        v_hat = torch.tanh(Lv/2)
                    elif self.args.soft_sign == 'STE':
                        v_hat = self.ste_sign(Lv)
                decoded_bits[:, right_bit_position] = v_hat.squeeze(1)

                return llrs, partial_llrs, decoded_bits

        # General case
        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u
            # Lu = log_sum_avoid_zero_NaN(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))

            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                Lu = llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                L_u = partial_llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                # L_uv = log_sum_avoid_zero_NaN(L_u, llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index])
            else:
                Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index].clone(), llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index].clone())

                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu
                llrs, partial_llrs, decoded_bits = self.partial_decode_soft(llrs, partial_llrs, depth-1, left_bit_position, leaf_position, prior, decoded_bits)

                return llrs, partial_llrs, decoded_bits

            # RIGHT CHILD
            right_bit_position = 2*bit_position + 1
            L_uv = min_sum_log_sum_exp(L_u.clone(), llrs[:,depth, (left_bit_position)*half_index:(left_bit_position+1)*half_index].clone())
            Lv = L_uv + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
            llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv
            llrs, partial_llrs, decoded_bits = self.partial_decode_soft(llrs, partial_llrs, depth-1, right_bit_position, leaf_position, prior, decoded_bits)

            return llrs, partial_llrs, decoded_bits

    def updatePartialSums_soft(self, leaf_position, leaf_llrs, partial_llrs):
        # need to fix. this is wrong
        u = leaf_llrs.clone()
        u[:, leaf_position+1:] = 0

        for d in range(0, self.n):
            partial_llrs[:, d] = u
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [Lu Lv] encoded to [lse(Lu, Lv) Lv]

                # u = torch.cat((u[:, :i], log_sum_avoid_zero_NaN(u[:, i:i+num_bits].clone(), u[:, i+num_bits: i+2*num_bits]).float(), u[:, i+num_bits:]), dim=1)
                u = torch.cat((u[:, :i], min_sum_log_sum_exp(u[:, i:i+num_bits].clone(), u[:, i+num_bits: i+2*num_bits].clone()).float(), u[:, i+num_bits:]), dim=1)

        partial_llrs[:, self.n] = u

        return partial_llrs

    def pac_sc_decode_soft(self, corrupted_codewords, snr, priors=None):
        # for g=5
        sigma = snr_db2sigma(snr)
        llrs = (2/sigma**2)*corrupted_codewords
        if priors is None:
            priors = torch.zeros(llrs.shape[0], self.args.N).to(llrs.device)
        u_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        v_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)

        v_llrs = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        u_llrs = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)

        cState = torch.ones(corrupted_codewords.shape[0], len(self.g_array)-1, device=corrupted_codewords.device) #convolutional encoder state
        llr_array, partial_llrs = self.define_partial_arrays(llrs)
        for ii in range(self.N):
            if ii not in self.B: #frozen
                v_hat[:, ii] = 1 #zero
                v_llrs[:, ii] = self.infty
                if ii<2:
                    priors[:, ii] = v_llrs[:, ii]
                else:
                    priors[:, ii] = v_llrs[:, ii-2]#log_sum_avoid_zero_NaN(v_llrs[:, ii], v_llrs[:, ii-2])
                llr_array , decoded_bits = self.updateLLR_soft(ii, llr_array.clone(), partial_llrs, priors)
                u_llrs[:, ii] = llr_array[:, 0, ii]
            else: # non-frozen
                llr_array , decoded_bits = self.updateLLR_soft(ii, llr_array.clone(), partial_llrs, priors)
                u_llrs[:, ii] = llr_array[:, 0, ii]

                # v_llrs[:, ii] = u_llrs[:, ii].clone()
                jj = ii - 2
                # while(jj>=0):
                #     v_llrs[:, ii] = log_sum_avoid_zero_NaN(v_llrs[:, ii], u_llrs[:, jj])
                #     jj = jj-2

                v_llrs[:, ii] = min_sum_log_sum_exp(u_llrs[:, ii].clone(), v_llrs[:, jj].clone())
            partial_llrs = self.updatePartialSums_soft(ii, llr_array[:, 0, :], partial_llrs)
        decoded_llrs = v_llrs[:, self.B]
        decoded_bits = torch.sign(decoded_llrs)
        return decoded_llrs, decoded_bits, u_llrs

def pac_map_decode(args):
    all_msg_bits = []
    pac = PAC(args, args.N, args.K, args.g)
    for i in range(2**args.K):
        d = dec2bitarray(i, args.K)
        all_msg_bits.append(d)
    all_message_bits = torch.from_numpy(np.array(all_msg_bits))
    all_message_bits = 1 - 2*all_message_bits.float()
    codebook = pac.pac_encode(all_message_bits)
    b_codebook = codebook.repeat(args.batch_size, 1, 1)
    bers = {}
    blers = {}
    for r in range(int(args.test_ratio)):
        msg_bits = (torch.rand(args.batch_size, args.K) > 0.5).float()
        msg_bits_bpsk = 1 - 2*msg_bits
        pac_code = pac.pac_encode(msg_bits_bpsk, scheme = args.rate_profile)
        for snr in np.arange(args.test_snr_start, args.test_snr_end+1):

            sigma = snr_db2sigma(snr)
            # codes_G = polar.encode_G(msg_bits_bpsk)
            noisy_codes = pac.channel(pac_code, snr)
            b_noisy = noisy_codes.unsqueeze(1).repeat(1, 2**args.K, 1)
            diff = (b_noisy - b_codebook).pow(2).sum(dim=2)
            idx = diff.argmin(dim=1)
            decoded = all_message_bits[idx, :]

            errors = (decoded != msg_bits_bpsk).float()
            bit_error_rate = (torch.sum(errors)/(errors.shape[0]*errors.shape[1]))
            bler = torch.sum((torch.sum(errors, dim=1)>0).float())/errors.shape[0]
            # print("SNR = {}, BER = {}, BLER = {}".format(snr, bit_error_rate, bler))
            if r==0:
                bers[snr] = bit_error_rate
                blers[snr] = bler
            else:
                bers[snr] += bit_error_rate
                blers[snr] += bler
    bers = {key:(value/args.test_ratio).item() for (key,value) in bers.items()}
    blers = {key:(value/args.test_ratio).item() for (key,value) in blers.items()}
    return bers, blers


if __name__ == '__main__':
    global args
    args = get_args()
    g = args.g#91
    M = int(np.floor(np.log2(g))) + 1
    pac = PAC(args, args.N, args.K, g)

    # msg_bits = np.random.randint(0,2,(args.batch_size, args.K))
    blers = {}
    msg_bits = torch.randint(0,2,(args.batch_size, args.K), dtype=torch.float)
    msg_bits_bpsk = 1 - 2*msg_bits

    pac_code = pac.pac_encode(msg_bits_bpsk, scheme=args.rate_profile)

    for snr in np.arange(args.test_snr_start, args.test_snr_end+1):
        # snr = 5
        start_time = time.time()
        sigma = snr_db2sigma(snr)
        noisy_code = pac.channel(pac_code, snr)

        llrs = (2/sigma**2)*noisy_code
        # llr_array, partial_sums = pac.define_partial_arrays(llrs)

        u = torch.empty_like(msg_bits_bpsk)
        metrics = torch.empty_like(pac_code)
        for ii, vv in enumerate(llrs):
            v_hat, pm = pac.fano_decode(vv.unsqueeze(0), delta = args.delta, bias = args.bias, bias_frozen = args.bias_frozen, verbose = args.verbose, maxDiversions = args.maxd, bias_type = args.bias_type)
            u[ii] = pac.extract(v_hat)
            if args.verbose == 0.5:
                if ii%args.printf == args.printf - 1:
                    bler_t = ((u.sign()[:ii+1, :] != msg_bits_bpsk.sign()[:ii+1, :]).float().sum(dim=1)>0).float().sum() / (ii+1)
                    print("{} examples, SNR = {}dB, BLER = {}, Time = {}s".format(ii+1, snr, bler_t, time.time() - start_time))

            # metrics[ii] = pm
        bler = ((u.sign() != msg_bits_bpsk.sign()).float().sum(dim=1)>0).float().sum() / args.batch_size
        blers[snr] = bler.item()


        print('SNR = {}, BLER = {}, time_taken = {} seconds'.format(snr, blers[snr], time.time() -start_time))

    # ber_map, bler_map = pac_map_decode(args)
    # ber_polar, bler_polar = run_polar(args)
    # ber_polar_map, bler_polar_map = run_polar_map(args)
