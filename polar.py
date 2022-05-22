import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
import pickle
import os
import argparse
import sys
from collections import namedtuple

from utils import log_sum_exp, log_sum_avoid_zero_NaN, snr_db2sigma, STEQuantize, Clamp, min_sum_log_sum_exp, errors_ber, errors_bler
#from xformer_all import dec2bitarray

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
def get_args():
    parser = argparse.ArgumentParser(description='(N,K) Polar code')

    parser.add_argument('--N', type=int, default=4, help='Polar code parameter N')
    parser.add_argument('--K', type=int, default=3, help='Polar code parameter K')
    parser.add_argument('--rate_profile', type=str, default='polar', choices=['RM', 'polar', 'sorted', 'sorted_last', 'rev_polar'], help='Polar rate profiling')
    parser.add_argument('--hard_decision', dest = 'hard_decision', default=False, action='store_true')
    parser.add_argument('--only_args', dest = 'only_args', default=False, action='store_true')
    parser.add_argument('--list_size', type=int, default=1, help='SC List size')
    parser.add_argument('--crc_len', type=int, default='0', choices=[0, 3, 8, 16], help='CRC length')

    parser.add_argument('--batch_size', type=int, default=10000, help='size of the batches')
    parser.add_argument('--test_ratio', type = float, default = 1, help = 'Number of test samples x batch_size')
    parser.add_argument('--test_snr_start', type=float, default=-2., help='testing snr start')
    parser.add_argument('--test_snr_end', type=float, default=4., help='testing snr end')
    parser.add_argument('--snr_points', type=int, default=7, help='testing snr num points')
    args = parser.parse_args()

    return args

class PolarCode:

    def __init__(self, n, K, args, F = None, rs = None, use_cuda = True, infty = 1000.):

        assert n>=1
        self.args = args
        self.n = n
        self.N = 2**n
        self.K = K
        self.G2 = np.array([[1,0],[1,1]])
        self.G = np.array([1])
        for i in range(n):
            self.G = np.kron(self.G, self.G2)
        self.G = torch.from_numpy(self.G).float()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        clamp_class = Clamp()
        self.clamp = clamp_class.apply
        self.infty = infty

        if F is not None:
            assert len(F) == self.N - self.K
            self.frozen_positions = F
            self.unsorted_frozen_positions = self.frozen_positions
            self.frozen_positions.sort()

            self.info_positions = np.array(list(set(self.frozen_positions) ^ set(np.arange(self.N))))
            self.unsorted_info_positions = self.info_positions
            self.info_positions.sort()
        else:
            if rs is None:
                # in increasing order of reliability
                self.reliability_seq = np.arange(1023, -1, -1)
                self.rs = self.reliability_seq[self.reliability_seq<self.N]
            else:
                self.reliability_seq = rs
                self.rs = self.reliability_seq[self.reliability_seq<self.N]

                assert len(self.rs) == self.N
            # best K bits
            self.info_positions = self.rs[:self.K]
            self.unsorted_info_positions = self.reliability_seq[self.reliability_seq<self.N][:self.K]
            self.info_positions.sort()
            self.unsorted_info_positions=np.flip(self.unsorted_info_positions)
            # worst N-K bits
            self.frozen_positions = self.rs[self.K:]
            self.unsorted_frozen_positions = self.rs[self.K:]
            self.frozen_positions.sort()


            self.CRC_polynomials = {
            3: torch.Tensor([1, 0, 1, 1]).int(),
            8: torch.Tensor([1, 1, 1, 0, 1, 0, 1, 0, 1]).int(),
            16: torch.Tensor([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).int(),
                                    }

    def encode_G(self, message):

        u = torch.ones(message.shape[0], self.N, dtype=torch.float)
        u[:, self.info_positions] = message

        code = 1 - 2*((0.5 - 0.5*u).mm(self.G)%2)

        return code

    def encode_plotkin(self, message, scaling = None, custom_info_positions = None):

        # message shape is (batch, k)
        # BPSK convention : 0 -> +1, 1 -> -1
        # Therefore, xor(a, b) = a*b
        if custom_info_positions is not None:
            info_positions = custom_info_positions
        else:
            info_positions = self.info_positions
        u = torch.ones(message.shape[0], self.N, dtype=torch.float).to(message.device)
        u[:, info_positions] = message

        for d in range(0, self.n):
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [u v] encoded to [u xor(u,v)]
                u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
                # u[:, i:i+num_bits] = u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits].clone
        if scaling is not None:
            u = (scaling * np.sqrt(self.N)*u)/torch.norm(scaling)
        return u

    def neural_encode_plotkin(self, message, power_constraint_type = 'hard_power_block'):

        # message shape is (batch, k)
        # BPSK convention : 0 -> +1, 1 -> -1
        # Therefore, xor(a, b) = a*b

        u = torch.ones(message.shape[0], self.N, dtype=torch.float).to(self.device)
        u[:, self.info_positions] = message.to(self.device)

        for d in range(0, self.n):
            depth = self.n - d
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [u v] encoded to [u xor(u,v)]

                u = torch.cat((u[:, :i], self.gnet_dict[depth-1](u[:, i:i+2*num_bits]), u[:, i+num_bits:]), dim=1)
                # u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
                # u[:, i:i+num_bits] = u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits].clone
        return self.power_constraint(u, None, power_constraint_type, 'train')

    def power_constraint(self, codewords, gnet_top, power_constraint_type, training_mode):


        if power_constraint_type in ['soft_power_block','soft_power_bit']:

            this_mean = codewords.mean(dim=0) if power_constraint_type == 'soft_power_bit' else codewords.mean()
            this_std  = codewords.std(dim=0) if power_constraint_type == 'soft_power_bit' else codewords.std()

            if training_mode == 'train':          # Training
                power_constrained_codewords = (codewords - this_mean)*1.0 / this_std

                gnet_top.update_normstats_for_test(this_mean, this_std)

            elif training_mode == 'test':         # For inference
                power_constrained_codewords = (codewords - gnet_top.mean_scalar)*1.0/gnet_top.std_scalar

    #         else:                                 # When updating the stat parameters of g2net. Just don't do anything
    #             power_constrained_codewords = _

            return power_constrained_codewords


        elif power_constraint_type == 'hard_power_block':

            return F.normalize(codewords, p=2, dim=1)*np.sqrt(self.N)


        else: # 'hard_power_bit'

            return codewords/codewords.abs()

    def channel(self, code, snr):
        sigma = snr_db2sigma(snr)

        noise = (sigma* torch.randn(code.shape, dtype = torch.float)).to(code.device)
        r = code + noise

        return r

    def sc_decode(self, noisy_code, snr):
        # Successive cancellation decoder for polar codes

        noise_sigma = snr_db2sigma(snr)
        llrs = (2/noise_sigma**2)*noisy_code
        assert noisy_code.shape[1] == self.N
        decoded_bits = torch.zeros(noisy_code.shape[0], self.N)

        depth = 0

        # function is recursively called (DFS)
        # arguments: Beliefs at the input of node (LLRs at top node), depth of children, bit_position (zero at top node)
        decoded_codeword, decoded_bits = self.decode(llrs, depth, 0, decoded_bits)
        decoded_message = torch.sign(decoded_bits)[:, self.info_positions]

        return decoded_message

    def decode(self, llrs, depth, bit_position, decoded_bits=None):
        # Function to call recursively, for SC decoder

        # print("DEPTH = {}, bit_position = {}".format(depth, bit_position))
        half_index = 2 ** (self.n - depth - 1)

        # n = 2 tree case
        if depth == self.n - 1:
            # Left child
            left_bit_position = 2*bit_position
            if left_bit_position in self.frozen_positions:
                # If frozen decoded bit is 0
                u_hat = torch.ones_like(llrs[:, :half_index], dtype=torch.float)
            else:
                # Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)
                Lu = log_sum_avoid_zero_NaN(llrs[:, :half_index], llrs[:, half_index:]).sum(dim=1, keepdim=True)
                if self.args.hard_decision:
                    u_hat = torch.sign(Lu)
                else:
                    u_hat = torch.tanh(Lu/2)

            # Right child
            right_bit_position = 2*bit_position + 1
            if right_bit_position in self.frozen_positions:
                # If frozen decoded bit is 0
                v_hat = torch.ones_like(llrs[:, :half_index], dtype = torch.float)
            else:
                Lv = u_hat * llrs[:, :half_index] + llrs[:, half_index:]
                if self.args.hard_decision:
                    v_hat = torch.sign(Lv)
                else:
                    v_hat = torch.tanh(Lv/2)

            #print("DECODED: Bit positions {} : {} and {} : {}".format(left_bit_position, u_hat, right_bit postion, v_hat))

            decoded_bits[:, left_bit_position] = u_hat.squeeze(1)
            decoded_bits[:, right_bit_position] = v_hat.squeeze(1)

            return torch.cat((u_hat * v_hat, v_hat), dim = 1).float(), decoded_bits

        # General case
        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u
            # Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))
            Lu = log_sum_avoid_zero_NaN(llrs[:, :half_index], llrs[:, half_index:])

            u_hat, decoded_bits = self.decode(Lu, depth+1, bit_position*2, decoded_bits)

            # RIGHT CHILD
            Lv = u_hat * llrs[:, :half_index] + llrs[:, half_index:]
            v_hat, decoded_bits = self.decode(Lv, depth+1, bit_position*2 + 1, decoded_bits)

            return torch.cat((u_hat * v_hat, v_hat), dim=1), decoded_bits

    def sc_decode_soft(self, noisy_code, snr, priors=None):
        # Soft successive cancellation decoder for polar codes
        # Left subtree : L_u^ = LSE(L_1, L_2) + prior (like normal)
        # Right subtree : L_v^ = LSE(L_u^, L_1) + L_2
        # Return up: L_1^, L_2^ = LSE(L_u^, L_v^), L_v^


        noise_sigma = snr_db2sigma(snr)
        llrs = (2/noise_sigma**2)*noisy_code
        assert noisy_code.shape[1] == self.N
        decoded_bits = torch.zeros(noisy_code.shape[0], self.N)

        if priors is None:
            priors = torch.zeros(self.N)

        depth = 0

        # function is recursively called (DFS)
        # arguments: Beliefs at the input of node (LLRs at top node), depth of children, bit_position (zero at top node)
        decoded_codeword, decoded_bits = self.decode_soft(llrs, depth, 0, priors, decoded_bits)
        decoded_message = torch.sign(decoded_bits)[:, self.info_positions]

        return decoded_message

    def decode_soft(self, llrs, depth, bit_position, prior, decoded_bits=None):
        # Function to call recursively, for soft SC decoder

        # print("DEPTH = {}, bit_position = {}".format(depth, bit_position))
        half_index = 2 ** (self.n - depth - 1)

        # n = 2 tree case
        if depth == self.n - 1:
            # Left child
            left_bit_position = 2*bit_position

            # Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)
            Lu = log_sum_avoid_zero_NaN(llrs[:, :half_index], llrs[:, half_index:]).sum(dim=1, keepdim=True)
            Lu = self.clamp(Lu + prior[left_bit_position]*torch.ones_like(Lu), -1000, 1000)
            if self.args.hard_decision:
                u_hat = torch.sign(Lu)
            else:
                u_hat = torch.tanh(Lu/2)
            L_uv = log_sum_avoid_zero_NaN(Lu, llrs[:, :half_index]).sum(dim=1, keepdim=True)

            # Right child
            right_bit_position = 2*bit_position + 1

            Lv = L_uv + llrs[:, half_index:]
            Lv = self.clamp(Lv + prior[right_bit_position]*torch.ones_like(Lv), -1000, 1000)
            if self.args.hard_decision:
                v_hat = torch.sign(Lv)
            else:
                v_hat = torch.tanh(Lv/2)

            #print("DECODED: Bit positions {} : {} and {} : {}".format(left_bit_position, u_hat, right_bit postion, v_hat))

            decoded_bits[:, left_bit_position] = u_hat.squeeze(1)
            decoded_bits[:, right_bit_position] = v_hat.squeeze(1)

            # print(depth, Lu.shape, Lv.shape, log_sum_avoid_zero_NaN(Lu, Lv).shape, torch.cat((log_sum_avoid_zero_NaN(Lu, Lv).sum(dim=1, keepdim=True), Lv), dim = 1).shape)
            return torch.cat((log_sum_avoid_zero_NaN(Lu, Lv).sum(dim=1, keepdim=True), Lv), dim = 1).float(), decoded_bits

        # General case
        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u
            # Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))
            Lu = log_sum_avoid_zero_NaN(llrs[:, :half_index], llrs[:, half_index:])

            L_u, decoded_bits = self.decode_soft(Lu, depth+1, bit_position*2, prior, decoded_bits)
            L_uv = log_sum_avoid_zero_NaN(L_u, llrs[:, :half_index])

            # RIGHT CHILD
            Lv = L_uv + llrs[:, half_index:]
            L_v, decoded_bits = self.decode_soft(Lv, depth+1, bit_position*2 + 1, prior, decoded_bits)
            # print(depth, L_u.shape, L_v.shape, log_sum_avoid_zero_NaN(L_u, L_v).shape, torch.cat((log_sum_avoid_zero_NaN(L_u, L_v).sum(dim=1, keepdim=True), L_v), dim = 1).shape)

            return torch.cat((log_sum_avoid_zero_NaN(L_u, L_v), L_v), dim = 1).float(), decoded_bits


    def define_partial_arrays(self, llrs):
        # Initialize arrays to store llrs and partial_sums useful to compute the partial successive cancellation process.
        llr_array = torch.zeros(llrs.shape[0], self.n+1, self.N, device=llrs.device)
        llr_array[:, self.n] = llrs
        partial_sums = torch.zeros(llrs.shape[0], self.n+1, self.N, device=llrs.device)
        return llr_array, partial_sums


    def updateLLR(self, leaf_position, llrs, partial_llrs = None, prior = None):

        #START
        depth = self.n
        decoded_bits = partial_llrs[:,0].clone()
        if prior is None:
            prior = torch.zeros(self.N) #priors
        llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth, 0, leaf_position, prior, decoded_bits)
        return llrs, decoded_bits


    def partial_decode(self, llrs, partial_llrs, depth, bit_position, leaf_position, prior, decoded_bits=None):
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
                u_hat = partial_llrs[:, depth-1, left_bit_position:left_bit_position+1]
            elif leaf_position_at_depth == left_bit_position:
                Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                # Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu + prior[left_bit_position]*torch.ones_like(Lu)
                if self.args.hard_decision:
                    u_hat = torch.sign(Lu)
                else:
                    u_hat = torch.tanh(Lu/2)

                decoded_bits[:, left_bit_position] = u_hat.squeeze(1)

                return llrs, partial_llrs, decoded_bits

            # Right child
            right_bit_position = 2*bit_position + 1
            if leaf_position_at_depth > right_bit_position:
                pass
            elif leaf_position_at_depth == right_bit_position:
                Lv = u_hat * llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index] + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
                llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv + prior[right_bit_position] * torch.ones_like(Lv)
                if self.args.hard_decision:
                    v_hat = torch.sign(Lv)
                else:
                    v_hat = torch.tanh(Lv/2)
                decoded_bits[:, right_bit_position] = v_hat.squeeze(1)
                return llrs, partial_llrs, decoded_bits

        # General case
        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u
            # Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))

            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                Lu = llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                u_hat = partial_llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
            else:

                Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])
                # Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])
                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu
                llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth-1, left_bit_position, leaf_position, prior, decoded_bits)

                return llrs, partial_llrs, decoded_bits

            # RIGHT CHILD
            right_bit_position = 2*bit_position + 1

            Lv = u_hat * llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index] + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
            llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv
            llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth-1, right_bit_position, leaf_position, prior, decoded_bits)

            return llrs, partial_llrs, decoded_bits

    def updatePartialSums(self, leaf_position, decoded_bits, partial_llrs):

        u = decoded_bits.clone()
        u[:, leaf_position+1:] = 0

        for d in range(0, self.n):
            partial_llrs[:, d] = u
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [u v] encoded to [u xor(u,v)]
                u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
        partial_llrs[:, self.n] = u
        return partial_llrs

    def sc_decode_new(self, corrupted_codewords, snr, use_gt = None):

        # step-wise implementation using updateLLR and updatePartialSums
        sigma = snr_db2sigma(snr)
        llrs = (2/sigma**2)*corrupted_codewords

        priors = torch.zeros(self.N)
        priors[self.frozen_positions] = self.infty

        u_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        llr_array, partial_llrs = self.define_partial_arrays(llrs)
        for ii in range(self.N):
            llr_array , decoded_bits = self.updateLLR(ii, llr_array.clone(), partial_llrs, priors)
            if use_gt is None:
                u_hat[:, ii] = torch.sign(llr_array[:, 0, ii])
            else:
                u_hat[:, ii] = use_gt[:, ii]
            partial_llrs = self.updatePartialSums(ii, u_hat, partial_llrs)
        decoded_bits = u_hat[:, self.info_positions]
        return llr_array[:, 0, :].clone(), decoded_bits

    def updateLLR_soft(self, leaf_position, llrs, partial_llrs, prior = None):

        #START
        depth = self.n
        decoded_bits = partial_llrs[:,0].clone()
        if prior is None:
            prior = torch.zeros(self.N) #priors
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
                Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                Lu = self.clamp(Lu + prior[left_bit_position]*torch.ones_like(Lu), -1000, 1000)

                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu + prior[left_bit_position]*torch.ones_like(Lu)
                if self.args.hard_decision:
                    u_hat = torch.sign(Lu)
                else:
                    u_hat = torch.tanh(Lu/2)

                decoded_bits[:, left_bit_position] = u_hat.squeeze(1)

                return llrs, partial_llrs, decoded_bits

            # Right child
            right_bit_position = 2*bit_position + 1
            if leaf_position_at_depth > right_bit_position:
                pass
            elif leaf_position_at_depth == right_bit_position:

                L_uv = log_sum_avoid_zero_NaN(L_u, llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index])
                Lv = L_uv + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
                Lv = self.clamp(Lv + prior[right_bit_position]*torch.ones_like(Lv), -1000, 1000)

                llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv + prior[right_bit_position] * torch.ones_like(Lv)
                if self.args.hard_decision:
                    v_hat = torch.sign(Lv)
                else:
                    v_hat = torch.tanh(Lv/2)
                decoded_bits[:, right_bit_position] = v_hat.squeeze(1)

                return llrs, partial_llrs, decoded_bits

        # General case
        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u
            # Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))

            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                Lu = llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                L_u = partial_llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                # L_uv = log_sum_avoid_zero_NaN(L_u, llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index])
            else:

                Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])
                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu
                llrs, partial_llrs, decoded_bits = self.partial_decode_soft(llrs, partial_llrs, depth-1, left_bit_position, leaf_position, prior, decoded_bits)

                return llrs, partial_llrs, decoded_bits

            # RIGHT CHILD
            right_bit_position = 2*bit_position + 1
            L_uv = log_sum_avoid_zero_NaN(L_u, llrs[:,depth, (left_bit_position)*half_index:(left_bit_position+1)*half_index])
            Lv = L_uv + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
            llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv
            llrs, partial_llrs, decoded_bits = self.partial_decode_soft(llrs, partial_llrs, depth-1, right_bit_position, leaf_position, prior, decoded_bits)

            return llrs, partial_llrs, decoded_bits

    def updatePartialSums_soft(self, leaf_position, leaf_llrs, partial_llrs):
        # In the partial sum array, we store the L^ of the decoded positions.
        # LLR for (u^ xor v^, v^) will be (LSE(L_u^, L_v^), L_v^)
        u = leaf_llrs.clone()
        u[:, leaf_position+1:] = 0

        for d in range(0, self.n):
            partial_llrs[:, d] = u
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [Lu Lv] encoded to [lse(Lu, Lv) Lv]
                u = torch.cat((u[:, :i], log_sum_avoid_zero_NaN(u[:, i:i+num_bits].clone(), u[:, i+num_bits: i+2*num_bits]).float(), u[:, i+num_bits:]), dim=1)
        partial_llrs[:, self.n] = u

        return partial_llrs

    def sc_decode_soft_new(self, corrupted_codewords, snr, priors=None):
        # uses updateLLR_soft and updatePartialSums_soft

        sigma = snr_db2sigma(snr)
        llrs = (2/sigma**2)*corrupted_codewords
        if priors is None:
            priors = torch.zeros(self.N)

        u_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        llr_array, partial_llrs = self.define_partial_arrays(llrs)
        for ii in range(self.N):
            llr_array , decoded_bits = self.updateLLR_soft(ii, llr_array.clone(), partial_llrs, priors)
            u_hat[:, ii] = torch.sign(llr_array[:, 0, ii])
            partial_llrs = self.updatePartialSums_soft(ii, llr_array[:, 0, :], partial_llrs)
        decoded_bits = u_hat[:, self.info_positions]
        return decoded_bits

    def neural_sc_decode(self, noisy_code, snr, p = None):

        noise_sigma = snr_db2sigma(snr)
        llrs = ((2/noise_sigma**2)*noisy_code).to(self.device)

        assert noisy_code.shape[1] == self.N
        # if frozen bit, llr = very large (high likelihood of 0) (P.S.: after BPSK, 0 -> +1 , 1 -> -1)
        decoded_llrs = 1000*torch.ones(noisy_code.shape[0], self.N).to(self.device)

        depth = 0
        if p is None:
            p = 0.5*torch.ones(self.N)
        # function is recursively called (DFS)
        # arguments: Beliefs at the input of node (LLRs at top node), depth of children, bit_position (zero at top node)
        # depth of root node = 0, => depth of leaves will be n

        decoded_codeword, decoded_llrs = self.neural_decode(llrs, depth, 0, decoded_llrs, p)
        # decoded_message = torch.sign(decoded_bits)[:, self.info_positions]

        return decoded_llrs[:, self.info_positions]

    def neural_decode(self, llrs, depth, bit_position, decoded_llrs=None, p=None):

        # print("DEPTH = {}, bit_position = {}".format(depth, bit_position))
        half_index = 2 ** (self.n - depth - 1) # helper variable: half of length of belief (LLR) vector


        if depth == self.n - 1: # n = 2 tree case - penultimate layer of tree
            # Left child
            left_bit_position = 2*bit_position
            if left_bit_position in self.frozen_positions:
                # If frozen decoded bit is 0
                u_hat = torch.ones_like(llrs[:, :half_index], dtype=torch.float)
            else:
                if self.args.no_sharing_weights:
                    Lu = self.fnet_dict[depth+1][2*bit_position](llrs)
                else:
                    Lu = self.fnet_dict[depth+1]['left'](llrs)
                if self.args.augment:
                    Lu = Lu + log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)
                #Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)

                prior = torch.log((p[left_bit_position])/(1 - p[left_bit_position]))
                Lu = self.clamp(Lu - torch.ones_like(Lu)*prior.item(), -1000, 1000)
                decoded_llrs[:, left_bit_position] = Lu.squeeze(1)

                if self.args.hard_decision:
                    u_hat = torch.sign(Lu)
                else:
                    u_hat = torch.tanh(Lu/2)

            # Right child
            right_bit_position = 2*bit_position + 1
            if right_bit_position in self.frozen_positions:
                # If frozen decoded bit is 0
                v_hat = torch.ones_like(llrs[:, :half_index], dtype = torch.float)
            else:
                if self.args.no_sharing_weights:
                    Lv = self.fnet_dict[depth+1][2*bit_position+1](torch.cat((llrs, u_hat), dim=1))
                else:
                    Lv = self.fnet_dict[depth+1]['right'](torch.cat((llrs, u_hat), dim=1))
                if self.args.augment:
                    Lv = Lv + u_hat * llrs[:, :half_index] + llrs[:, half_index:]
                prior = torch.log((p[right_bit_position])/(1 - p[right_bit_position]))
                Lv = self.clamp(Lv - torch.ones_like(Lv)*prior.item(), -1000, 1000)

                decoded_llrs[:, right_bit_position] = Lv.squeeze(1)
                # Lv = u_hat * llrs[:, :half_index] + llrs[:, half_index:]
                if self.args.hard_decision:
                    v_hat = torch.sign(Lv)
                else:
                    v_hat = torch.tanh(Lv/2)

            #print("DECODED: Bit positions {} : {} and {} : {}".format(left_bit_position, u_hat, right_bit postion, v_hat))




            if self.args.no_sharing_weights:
                num_positions_on_level = 2**depth
                if bit_position == num_positions_on_level - 1:
                    return torch.cat((u_hat * v_hat, v_hat), dim = 1).float(), decoded_llrs
                else:
                    p0 = self.gnet_dict[depth][bit_position](torch.cat((u_hat, v_hat), dim = 1))
                    return torch.cat((p0, v_hat), dim=1), decoded_llrs
            else:
                p0 = self.gnet_dict[depth](torch.cat((u_hat, v_hat), dim = 1))
                return torch.cat((p0, v_hat), dim=1), decoded_llrs
            # return torch.cat((u_hat * v_hat, v_hat), dim = 1).float(), decoded_bits

        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u

            #Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))
            if self.args.no_sharing_weights:
                Lu = self.fnet_dict[depth+1][2*bit_position](llrs)
            else:
                # print('LLRs device: ', llrs.device)
                Lu = self.fnet_dict[depth+1]['left'](llrs.to(self.device))
            if self.args.augment:
                Lu = Lu + log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)

            u_hat, decoded_llrs = self.neural_decode(Lu, depth+1, bit_position*2, decoded_llrs, p)

            # RIGHT CHILD
            #Lv = u_hat * llrs[:, :half_index] + llrs[:, half_index:]
            # need to verify dimensions
            if self.args.no_sharing_weights:
                Lv = self.fnet_dict[depth+1][2*bit_position+1](torch.cat((llrs, u_hat), dim=1))
            else:
                Lv = self.fnet_dict[depth+1]['right'](torch.cat((llrs, u_hat), dim=1))
            if self.args.augment:
                Lv = Lv + u_hat * llrs[:, :half_index] + llrs[:, half_index:]
            v_hat, decoded_llrs = self.neural_decode(Lv, depth+1, bit_position*2 + 1, decoded_llrs, p)

            if self.args.no_sharing_weights:
                num_positions_on_level = 2**depth
                if bit_position == num_positions_on_level - 1: # no need to learn reconstruction of codeword
                    return torch.cat((u_hat * v_hat, v_hat), dim=1), decoded_llrs
                else:
                    #reconstruct parent llr, p0
                    p0 = self.gnet_dict[depth][bit_position](torch.cat((u_hat, v_hat), dim = 1))
                    return torch.cat((p0, v_hat), dim=1), decoded_llrs

            else:
                p0 = self.gnet_dict[depth](torch.cat((u_hat, v_hat), dim = 1))
                return torch.cat((p0, v_hat), dim=1), decoded_llrs

    def get_CRC(self, message):

        # need to optimize.
        # inout message should be int

        padded_bits = torch.cat([message, torch.zeros(polar.CRC_len).int()])
        while len(padded_bits[0:polar.K_minus_CRC].nonzero()):
            cur_shift = (padded_bits != 0).int().argmax(0)
            padded_bits[cur_shift: cur_shift + polar.CRC_len + 1] ^= polar.CRC_polynomials[polar.CRC_len]

        return padded_bits[self.K_minus_CRC:]

    def CRC_check(self, message):

        # need to optimize.
        # input message should be int

        padded_bits = message
        while len(padded_bits[0:polar.K_minus_CRC].nonzero()):
            cur_shift = (padded_bits != 0).int().argmax(0)
            padded_bits[cur_shift: cur_shift + polar.CRC_len + 1] ^= polar.CRC_polynomials[polar.CRC_len]

        if padded_bits[polar.K_minus_CRC:].sum()>0:
            return 0
        else:
            return 1

    def encode_with_crc(self, message, CRC_len):
        self.CRC_len = CRC_len
        self.K_minus_CRC = self.K - CRC_len

        if CRC_len == 0:
            return self.encode_plotkin(message)
        else:
            crcs = 1-2*torch.vstack([self.get_CRC((0.5+0.5*message[jj]).int()) for jj in range(message.shape[0])])
            encoded = self.encode_plotkin(torch.cat([message, crcs], 1))

            return encoded

    def pruneLists(self, llr_array_list, partial_llrs_list, u_hat_list, metric_list, L):
        _, inds = torch.topk(-1*metric_list, L, 0) # select L gratest indices in every row
        sorted_inds, _ = torch.sort(inds, 0)
        batch_size = partial_llrs_list.shape[1]

        # llr_array_list = torch.gather(llr_array_list, 0, sorted_inds.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, llr_array_list.shape[2], llr_array_list.shape[3]))
        # partial_llrs_list = torch.gather(partial_llrs_list, 0, sorted_inds.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, partial_llrs_list.shape[2], partial_llrs_list.shape[3]))
        # metric_list = torch.gather(metric_list, 0, sorted_inds)
        # u_hat_list = torch.gather(u_hat_list, 0, sorted_inds.unsqueeze(-1).repeat(1, 1, u_hat_list.shape[2]))
        llr_array_list = llr_array_list[sorted_inds, torch.arange(batch_size)]
        partial_llrs_list = partial_llrs_list[sorted_inds, torch.arange(batch_size)]
        metric_list = metric_list[sorted_inds, torch.arange(batch_size)]
        u_hat_list = u_hat_list[sorted_inds, torch.arange(batch_size)]

        return llr_array_list, partial_llrs_list, u_hat_list, metric_list

    def scl_decode(self, corrupted_codewords, snr, L=1, use_CRC = False):

        # step-wise implementation using updateLLR and updatePartialSums
        sigma = snr_db2sigma(snr)
        llrs = (2/sigma**2)*corrupted_codewords
        batch_size = corrupted_codewords.shape[0]

        priors = torch.zeros(self.N)
        # add frozen priors later only
        #priors[self.frozen_positions] = self.infty

        u_hat_list = torch.zeros(1, corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        llr_array, partial_llrs = self.define_partial_arrays(llrs)
        llr_array_list = llr_array.unsqueeze(0)
        partial_llrs_list = partial_llrs.unsqueeze(0)
        metric_list = torch.zeros(1, llrs.shape[0])
        for ii in range(self.N):
            list_size = llr_array_list.shape[0]
            if ii in self.frozen_positions:
                llr_array , decoded_bits = self.updateLLR(ii, llr_array_list.reshape(-1, self.n+1, self.N).clone(), partial_llrs_list.reshape(-1, self.n+1, self.N), priors)
                metric = torch.abs(llr_array[:, 0, ii])*(llr_array[:, 0, ii].sign() != 1*torch.ones(llr_array.shape[0])).float()
                # add the infty prior only later, since metric uses |LLR|
                llr_array[:, 0, ii] = llr_array[:, 0, ii] + self.infty * torch.ones_like(llr_array[:, 0, ii])

                u_hat_list[:, :, ii] = torch.ones(list_size, batch_size, device=corrupted_codewords.device)
                partial_llrs = self.updatePartialSums(ii, u_hat_list.reshape(-1, self.N), partial_llrs_list.reshape(-1, self.n+1, self.N).clone())


                llr_array_list = llr_array.reshape(list_size, batch_size, self.n+1, self.N)
                partial_llrs_list = partial_llrs.reshape(list_size, batch_size, self.n+1, self.N)
                metric_list = metric_list + metric.reshape(list_size, batch_size)

                assert llr_array_list.shape[0] == partial_llrs_list.shape[0] == metric_list.shape[0] == u_hat_list.shape[0]

            else:
                llr_array , decoded_bits = self.updateLLR(ii, llr_array_list.reshape(-1, self.n+1, self.N).clone(), partial_llrs_list.reshape(-1, self.n+1, self.N), priors)
                metric = torch.abs(llr_array[:, 0, ii])

                #Duplicate lists
                u_hat_list = torch.vstack([u_hat_list, u_hat_list])
                u_hat_list[:list_size, :, ii] = torch.sign(llr_array[:, 0, ii]).reshape(list_size, batch_size)
                u_hat_list[list_size:, :, ii] = -1* torch.sign(llr_array[:, 0, ii]).reshape(list_size, batch_size)

                # same LLRs for both decisions
                llr_array_list = torch.vstack([llr_array.reshape(list_size, batch_size, self.n+1, self.N), llr_array.reshape(list_size, batch_size, self.n+1, self.N)])
                llr_array_list = torch.vstack([llr_array.reshape(list_size, batch_size, self.n+1, self.N), llr_array.reshape(list_size, batch_size, self.n+1, self.N)])

                # update partial sums for both decisions
                partial_llrs_list = self.updatePartialSums(ii, u_hat_list.reshape(-1, self.N), torch.vstack([partial_llrs_list, partial_llrs_list]).reshape(-1, self.n+1, self.N).clone()).reshape(2*list_size, batch_size, self.n+1, self.N)
                # no additional penalty for SC path
                metric_list = torch.vstack([metric_list, metric_list + metric.reshape(list_size, batch_size)])

                if llr_array_list.shape[0] > L: # prune list
                    llr_array_list, partial_llrs_list, u_hat_list, metric_list = self.pruneLists(llr_array_list, partial_llrs_list, u_hat_list, metric_list, L)

        list_size = llr_array_list.shape[0]
        if use_CRC:
            u_hat = u_hat_list[:, :, self.info_positions]
            decoded_bits = torch.zeros(batch_size, self.K_minus_CRC)
            llr_array = torch.zeros(batch_size, self.N)

            # optimize this later
            crc_checked = torch.zeros(list_size).int()
            for jj in range(batch_size):
                for kk in range(list_size):
                    crc_checked[kk] = self.CRC_check((0.5+0.5*u_hat[kk, jj]).int())

                if crc_checked.sum() == 0: #no code in list passes. pick lowest metric
                    decoded_bits[jj] = u_hat[metric_list[:, jj].argmin(), jj, :self.K_minus_CRC]
                    llr_array[jj] = llr_array_list[metric_list[:, jj].argmin(), jj, 0, :]
                else: # pick code that has lowest metric among ones that passed crc
                    inds = crc_checked.nonzero()
                    decoded_bits[jj] = u_hat[inds[metric_list[inds, jj].argmin()], jj, :self.K_minus_CRC]
                    llr_array[jj] = llr_array_list[inds[metric_list[inds, jj].argmin()], jj, 0, :]

        else: # do ML decision among the L messages in the list
            u_hat = u_hat_list[:, :, self.info_positions]
            codeword_list = self.encode_plotkin(u_hat.reshape(-1, self.K)).reshape(list_size, batch_size, self.N)
            inds = ((codeword_list - corrupted_codewords.unsqueeze(0))**2).sum(2).argmin(0)
            # get ML decision for each sample.
            decoded_bits = u_hat[inds, torch.arange(batch_size)]
            llr_array = llr_array_list[inds, torch.arange(batch_size), 0, :]

        return llr_array, decoded_bits
    
    
    def bitwise_MAP(self,noisy_enc,device,snr): # take bitwise independent map decisions and return output, this does not use the approximation -> log sum exp = max
        sigma = snr_db2sigma(snr)
        noisy_enc=(2/sigma**2)*noisy_enc
        all_msg_bits = []
        for i in range(2**self.K):
            d = dec2bitarray(i, self.K)
            all_msg_bits.append(d)
        all_message_bits = torch.from_numpy(np.array(all_msg_bits)).to(device)
        all_message_bits = 1 - 2*all_message_bits.float()
        #codebooks = []
        outputs = torch.ones(noisy_enc.shape[0],self.K,device=device)
        for bit in range(self.K):
            codebook1 = self.encode_plotkin(all_message_bits[all_message_bits[:,bit]==1.])
            codebook2 = self.encode_plotkin(all_message_bits[all_message_bits[:,bit]==-1.])
            dec1 = torch.logsumexp(torch.matmul(codebook1,noisy_enc.T).T,-1).unsqueeze(0)
            dec2 = torch.logsumexp(torch.matmul(codebook2,noisy_enc.T).T,-1).unsqueeze(0)
            dec = torch.cat((dec1,dec2),0)
            bit_dec = 1.-2.*torch.max(dec,0).indices
            outputs[:,bit] = bit_dec.T
            
        return outputs
            
        
    def get_generator_matrix(self,custom_info_positions=None):
        if custom_info_positions is not None:
            info_inds = custom_info_positions
        else:
            info_inds = self.info_positions
        msg = 1-2*torch.eye(self.K)
        code = 1.*(self.encode_plotkin(msg)==-1.)
        mat = torch.zeros((self.N,self.N))
        mat[info_inds,:] = code
        mat = mat.T
        return mat
    
    def get_min_xor_matrix(self):
        gen_mat = self.get_generator_matrix()
        xor_mat = gen_mat[polar.info_positions,:]
        return xor_mat
        
    def get_difficulty_seq(self,unrolling_seq):
        difficulty_seq = torch.zeros((self.N,self.K))
        gen_mat = self.get_generator_matrix()
        count = 0
        for bit in unrolling_seq:
            u = unrolling_seq[0:count+1]
            u.sort()
            difficulty = torch.sum(gen_mat[:,u],1)
            difficulty_seq[u,count] = difficulty[u]
            count += 1
        fin = difficulty_seq[self.info_positions,:]
        shifted = fin.clone()
        transfer = fin.clone()
        transfer[:,0] = 0
        shifted[:,:-1] = shifted[:,1:]-shifted[:,:-1]
        transfer[:,1:] = shifted[:,:-1]
        return fin,transfer
    
    def calculate_transfer_metric(self,unrolling_seq):
        _,deltas = self.get_difficulty_seq(unrolling_seq)
        avg = torch.sum(deltas)/torch.sum(1.0*(deltas > 0))
        return torch.max(deltas).item(),avg.item()

        
    def plot_standard_schemes(self,path='data'):
        h2e = self.unsorted_info_positions.tolist()
        e2h = self.unsorted_info_positions.tolist()
        e2h.reverse()
        l2r = self.info_positions.tolist()
        r2l = self.info_positions.tolist()
        r2l.reverse()
        bottom = -1
        top = 10
        
        path = path + '/polar_transfer_{0}_{1}'.format(self.K,self.N)
        os.makedirs(path, exist_ok=True)
        
        diff_seq_h2e1,diff_seq_h2e_transfer1 = self.get_difficulty_seq(h2e)
        diff_seq_h2e = diff_seq_h2e1.tolist()
        diff_seq_h2e_transfer = diff_seq_h2e_transfer1.tolist()
        plt.figure(figsize = (20,10))
        for i in range((self.K)):
            plt.step([float(elem) for elem in list(range(len(h2e)))], diff_seq_h2e[i], label = 'Bit {0}'.format(i))
        plt.ylim(bottom=bottom)
        plt.ylim(top=top)
        plt.ylabel("Learning Difficulty")
        plt.xlabel("Progressive training")
        plt.title("Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.savefig(path +'/polar_h2e_all_{0}_{1}.pdf'.format(self.K,self.N))
        plt.title("H2E plot , Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.close()
        plt.figure(figsize = (20,10))
        for i in range((self.K)):
            plt.step([float(elem) for elem in list(range(len(h2e)))], diff_seq_h2e_transfer[i], label = 'Bit {0}'.format(i))
        plt.ylim(bottom=bottom)
        plt.ylim(top=top)
        plt.title("Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.ylabel("Transfer Difficulty")
        plt.xlabel("Progressive training")
        plt.savefig(path +'/polar_transfer_h2e_all_{0}_{1}.pdf'.format(self.K,self.N))
        plt.title("H2E plot , Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.close()
        
        diff_seq_e2h1,diff_seq_e2h_transfer1 = self.get_difficulty_seq(e2h)
        diff_seq_e2h = diff_seq_e2h1.tolist()
        diff_seq_e2h_transfer = diff_seq_e2h_transfer1.tolist()
        plt.figure(figsize = (20,10))
        for i in range((self.K)):
            plt.step([float(elem) for elem in list(range(len(e2h)))], diff_seq_e2h[i], label = 'Bit {0}'.format(i))
        plt.ylim(bottom=bottom)
        plt.ylim(top=top)
        plt.title("Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.ylabel("Learning Difficulty")
        plt.xlabel("Progressive training")
        plt.savefig(path +'/polar_e2h_all_{0}_{1}.pdf'.format(self.K,self.N))
        plt.title("e2h plot , Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.close()
        plt.figure(figsize = (20,10))
        for i in range((self.K)):
            plt.step([float(elem) for elem in list(range(len(h2e)))], diff_seq_e2h_transfer[i], label = 'Bit {0}'.format(i))
        plt.ylim(bottom=bottom)
        plt.ylim(top=top)
        plt.title("Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.savefig(path +'/polar_transfer_e2h_all_{0}_{1}.pdf'.format(self.K,self.N))
        plt.title("e2h plot , Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.close()
        
        diff_seq_l2r1,diff_seq_l2r_transfer1 = self.get_difficulty_seq(l2r)
        diff_seq_l2r = diff_seq_l2r1.tolist()
        diff_seq_l2r_transfer = diff_seq_l2r_transfer1.tolist()
        plt.figure(figsize = (20,10))
        for i in range((self.K)):
            plt.step([float(elem) for elem in list(range(len(l2r)))], diff_seq_l2r[i], label = 'Bit {0}'.format(i))
        plt.ylim(bottom=bottom)
        plt.ylim(top=top)
        plt.title("Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.ylabel("Learning Difficulty")
        plt.xlabel("Progressive training")
        plt.savefig(path +'/polar_l2r_all_{0}_{1}.pdf'.format(self.K,self.N))
        plt.title("l2r plot , Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.close()
        plt.figure(figsize = (20,10))
        for i in range((self.K)):
            plt.step([float(elem) for elem in list(range(len(h2e)))], diff_seq_l2r_transfer[i], label = 'Bit {0}'.format(i))
        plt.ylim(bottom=bottom)
        plt.ylim(top=top)
        plt.title("Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.savefig(path +'/polar_transfer_l2r_all_{0}_{1}.pdf'.format(self.K,self.N))
        plt.title("l2r plot , Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.close()
        
        diff_seq_r2l1,diff_seq_r2l_transfer1 = self.get_difficulty_seq(r2l)
        diff_seq_r2l = diff_seq_r2l1.tolist()
        diff_seq_r2l_transfer = diff_seq_r2l_transfer1.tolist()
        plt.figure(figsize = (20,10))
        for i in range((self.K)):
            plt.step([float(elem) for elem in list(range(len(r2l)))], diff_seq_r2l[i], label = 'Bit {0}'.format(i))
        plt.ylim(bottom=bottom)
        plt.ylim(top=top)
        plt.title("Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.ylabel("Learning Difficulty")
        plt.xlabel("Progressive training")
        plt.savefig(path +'/polar_r2l_all_{0}_{1}.pdf'.format(self.K,self.N))
        plt.title("r2l plot , Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.close()
        plt.figure(figsize = (20,10))
        for i in range((self.K)):
            plt.step([float(elem) for elem in list(range(len(h2e)))], diff_seq_r2l_transfer[i], label = 'Bit {0}'.format(i))
        plt.ylim(bottom=bottom)
        plt.ylim(top=top)
        plt.title("Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.savefig(path +'/polar_transfer_r2l_all_{0}_{1}.pdf'.format(self.K,self.N))
        plt.title("r2l plot , Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.close()
        
        max1,avg1 = self.calculate_transfer_metric(h2e)
        print("Max Transfer Difficulty for H2E : {0}".format(max1)) 
        print("Avg Transfer Difficulty for H2E : {0}".format(avg1))
        print("\n")
        max1,avg1 = self.calculate_transfer_metric(e2h)
        print("Max Transfer Difficulty for E2H : {0}".format(max1)) 
        print("Avg Transfer Difficulty for E2h : {0}".format(avg1))
        print("\n")
        max1,avg1 = self.calculate_transfer_metric(l2r)
        print("Max Transfer Difficulty for L2R : {0}".format(max1)) 
        print("Avg Transfer Difficulty for L2R : {0}".format(avg1))
        print("\n")
        max1,avg1 = self.calculate_transfer_metric(r2l)
        print("Max Transfer Difficulty for R2L : {0}".format(max1)) 
        print("Avg Transfer Difficulty for R2L : {0}".format(avg1))
        print("\n")
        
        
        for i in range(self.K):
            fig = plt.figure(figsize = (20,10))
            ax = fig.add_subplot(1, 1, 1)
            #plt.step([float(elem) for elem in list(range(len(h2e)))], diff_seq_h2e[i], label = 'H2E Bit {0}'.format(i), marker='*', linewidth=1.5)
            #plt.step([float(elem) for elem in list(range(len(e2h)))], diff_seq_e2h[i], label = 'E2H Bit {0}'.format(i), marker='^', linewidth=1.5)
            plt.step([float(elem) for elem in list(range(len(l2r)))], diff_seq_l2r[i], label = '                     ',  where='post', color='tab:orange', marker='^', linewidth=2.5)
            plt.step([float(elem) for elem in list(range(len(r2l)))], diff_seq_r2l[i], label = '                     ',  where='post', color='g',marker='v', linewidth=2.5)
            plt.ylim(bottom=bottom)
            plt.ylim(top=top)
            #plt.title("Plot for bit {0}, Hardest to easiest order : {1}".format(i,np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
            plt.legend(prop={'size': 15},loc='lower left', bbox_to_anchor=(0.85,0.9))
            major_ticks = np.arange(0, self.K+1, 1)
            majory_ticks = np.arange(0, 10, 1)
            ax.set_xticks(major_ticks)
            ax.set_yticks(majory_ticks)
            ax.grid(which='major')
            #plt.ylabel("Learning Difficulty",fontsize=20)
            #plt.xlabel("Progressive training steps",fontsize=20)
            plt.savefig(path +'/polar_all_{0}_{1}_bit_{2}.pdf'.format(self.K,self.N,i))
            plt.close()
            
            fig = plt.figure(figsize = (20,10))
            ax = fig.add_subplot(1, 1, 1)
            plt.step([float(elem) for elem in list(range(len(h2e)))], diff_seq_h2e_transfer[i], label = 'H2E Bit {0}'.format(i), marker='*', linewidth=1.5)
            plt.step([float(elem) for elem in list(range(len(e2h)))], diff_seq_e2h_transfer[i], label = 'E2H Bit {0}'.format(i), marker='^', linewidth=1.5)
            plt.step([float(elem) for elem in list(range(len(l2r)))], diff_seq_l2r_transfer[i], label = 'L2R Bit {0}'.format(i), marker='o', linewidth=1.5)
            plt.step([float(elem) for elem in list(range(len(r2l)))], diff_seq_r2l_transfer[i], label = 'R2L Bit {0}'.format(i), marker='x', linewidth=1.5)
            plt.ylim(bottom=bottom)
            plt.ylim(top=top)
            plt.title("Transfer Plot for bit {0}, Hardest to easiest order : {1}".format(i,np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
            plt.legend(prop={'size': 15},loc='upper right', bbox_to_anchor=(1.1, 1))
            major_ticks = np.arange(0, self.K+1, 1)
            ax.set_xticks(major_ticks)
            ax.set_yticks(majory_ticks)
            ax.grid(which='major')
            plt.ylabel("Transfer Difficulty",fontsize=20)
            plt.xlabel("Progressive training steps",fontsize=20)
            plt.savefig(path +'/polar_transfer_all_{0}_{1}_bit_{2}.pdf'.format(self.K,self.N,i))
            plt.close()
            
        fig = plt.figure(figsize = (20,10))
        ax = fig.add_subplot(1, 1, 1)
        plt.step([float(elem) for elem in list(range(len(h2e)))], torch.sum(diff_seq_h2e1,0), label = 'H2E sum', marker='*', linewidth=1.5)
        plt.step([float(elem) for elem in list(range(len(e2h)))], torch.sum(diff_seq_e2h1,0), label = 'E2H sum', marker='^', linewidth=1.5)
        plt.step([float(elem) for elem in list(range(len(l2r)))], torch.sum(diff_seq_l2r1,0), label = 'L2R sum', marker='o', linewidth=1.5)
        plt.step([float(elem) for elem in list(range(len(r2l)))], torch.sum(diff_seq_r2l1,0), label = 'R2L sum', marker='x', linewidth=1.5)
        plt.ylim(bottom=bottom)
        plt.ylim(top=top)
        plt.title("Sum plot, Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.legend(prop={'size': 15},loc='upper right', bbox_to_anchor=(1.1, 1))
        major_ticks = np.arange(0, self.K+1, 1)
        majory_ticks = np.arange(0, 180, 10)
        ax.set_xticks(major_ticks)
        ax.set_yticks(majory_ticks)
        ax.grid(which='major')
        plt.ylabel("Learning Difficulty")
        plt.xlabel("Progressive training steps")
        plt.savefig(path +'/all_polar_transfer_sum.pdf')
        plt.close()
        
        fig = plt.figure(figsize = (20,10))
        ax = fig.add_subplot(1, 1, 1)
        plt.step([float(elem) for elem in list(range(len(h2e)))], torch.max(diff_seq_h2e1,0).values, label = 'H2E max', marker='*', linewidth=1.5)
        plt.step([float(elem) for elem in list(range(len(e2h)))], torch.max(diff_seq_e2h1,0).values, label = 'E2H max', marker='^', linewidth=1.5)
        plt.step([float(elem) for elem in list(range(len(l2r)))], torch.max(diff_seq_l2r1,0).values, label = 'L2R max', marker='o', linewidth=1.5)
        plt.step([float(elem) for elem in list(range(len(r2l)))], torch.max(diff_seq_r2l1,0).values, label = 'R2L max', marker='x', linewidth=1.5)
        plt.ylim(bottom=bottom)
        plt.ylim(top=top)
        plt.title("Max plot, Hardest to easiest order : {0}".format(np.argsort(np.argsort(self.unsorted_info_positions.copy()))))
        plt.legend(prop={'size': 15},loc='upper right', bbox_to_anchor=(1.1, 1))
        major_ticks = np.arange(0, self.K+1, 1)
        majory_ticks = np.arange(0, 10, 1)
        ax.set_xticks(major_ticks)
        ax.set_yticks(majory_ticks)
        ax.grid(which='major')
        plt.ylabel("Learning Difficulty")
        plt.xlabel("Progressive training steps")
        plt.savefig(path +'/all_polar_transfer_max.pdf')
        plt.close()
        
        


if __name__ == '__main__':
    args = get_args()

    n = int(np.log2(args.N))


    if args.ratel2rofile == 'polar':
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
        rs = rs[rs<args.N]

        ###############
        ### Polar code
        ##############

        ### Encoder

    elif args.rate_profile == 'RM':
        rmweightr2lnp.array([countSetBits(i) for i in range(args.N)])
        Fr = np.argsort(rmweight)[:-K]
        Fr.sort()


    elif args.rate_profile == 'sorted':
        if n == 5:
            rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

        elif n == 4:
            rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
        elif n == 3:
            rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
        elif n == 2:
            rs = np.array([3, 2, 1, 0])

        rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

        rs = rs[rs<args.N]
        first_inds = rs[:args.N//2].copy()
        first_inds.sort()
        rs[:args.N//2] = first_inds

    elif args.rate_profile == 'sorted_last':
        if n == 5:
            rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

        elif n == 4:
            rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
        elif n == 3:
            rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
        elif n == 2:
            rs = np.array([3, 2, 1, 0])

        rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

        rs = rs[rs<args.N]
        first_inds = rs[:args.N//2].copy()
        first_inds.sort()
        rs[:args.N//2] = first_inds[::-1]

    elif args.rate_profile == 'rev_polar':
        if n == 5:
            rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

        elif n == 4:
            rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
        elif n == 3:
            rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
        elif n == 2:
            rs = np.array([3, 2, 1, 0])

        rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

        rs = rs[rs<args.N]
        first_inds = rs[:args.N//2].copy()
        rs[:args.N//2] = first_inds[::-1]
    polar = PolarCode(n, args.K, args, rs = rs)

    # Multiple SNRs:
    if args.snr_points == 1 and args.test_snr_start == args.test_snr_end:
        snr_range = [args.test_snr_start]
    else:
        snrs_interval = (args.test_snr_end - args.test_snr_start)* 1.0 /  (args.snr_points-1)
        snr_range = [snrs_interval* item + args.test_snr_start for item in range(args.snr_points)]

    if args.only_args:
        print("Loaded args. Exiting")
        sys.exit()

    bers_SCL = [0. for ii in snr_range]
    blers_SCL = [0. for ii in snr_range]
    bers_SC = [0. for ii in snr_range]
    blers_SC = [0. for ii in snr_range]

    for r in range(int(args.test_ratio)):
        msg_bits = 1 - 2*(torch.rand(args.batch_size, args.K) > 0.5).float()
        # msg_bits = 1 - 2*torch.zeros(args.batch_size, args.K).float()
        # scl_msg_bits = 1 - 2*(torch.rand(args.batch_size, args.K - args.crc_len) > 0.5).float()
        codes = polar.encode_plotkin(msg_bits)
        # scl_codes = polar.encode_with_crc(scl_msg_bits, args.crc_len)


        for snr_ind, snr in enumerate(snr_range):

            # codes_G = polar.encode_G(msg_bits_bpsk)
            noisy_code = polar.channel(codes, snr)
            noise = noisy_code - codes
            # scl_noisy_code = scl_codes + noise

            SC_llrs, decoded_SC_msg_bits = polar.sc_decode_new(noisy_code, snr)
            ber_SC = errors_ber(msg_bits, decoded_SC_msg_bits.sign()).item()
            bler_SC = errors_bler(msg_bits, decoded_SC_msg_bits.sign()).item()

            # print("SNR = {}, BER = {}, BLER = {}".format(snr, bit_error_rate, bler))
            bers_SC[snr_ind] += ber_SC/args.test_ratio
            blers_SC[snr_ind] += bler_SC/args.test_ratio

            # SCL_llrs, decoded_SCL_msg_bits = polar.scl_decode(scl_noisy_code, snr, args.list_size)
            # ber_SCL = errors_ber(scl_msg_bits, decoded_SCL_msg_bits.sign()).item()
            # bler_SCL = errors_bler(scl_msg_bits, decoded_SCL_msg_bits.sign()).item()
            # print("SNR = {}, BER = {}, BLER = {}".format(snr, bit_error_rate, bler))

            SCL_llrs, decoded_SCL_msg_bits = polar.scl_decode(noisy_code, snr, args.list_size, use_CRC = False)
            ber_SCL = errors_ber(msg_bits, decoded_SCL_msg_bits.sign()).item()
            bler_SCL = errors_bler(msg_bits, decoded_SCL_msg_bits.sign()).item()

            bers_SCL[snr_ind] += ber_SCL/args.test_ratio
            blers_SCL[snr_ind] += bler_SCL/args.test_ratio

    print("Test SNRs : ", snr_range)
    print("BERs of SC: {0}".format(bers_SC))
    print("BERs of SCL: {0}".format(bers_SCL))
    print("BLERs of SC: {0}".format(blers_SC))
    print("BLERs of SCL: {0}".format(blers_SCL))
