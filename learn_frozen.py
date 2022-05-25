# __author__ = 'hebbarashwin'
# experimental code

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import RelaxedBernoulli

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import time

from polar import PolarCode
from utils import moving_average, snr_db2sigma, log_sum_exp, Clamp, STEQuantize

def get_args():
    parser = argparse.ArgumentParser(description='(N,K) Polar code')

    """ Polar code parameters """
    parser.add_argument('-N', type=int, default=4, help='Polar code parameter N')
    parser.add_argument('-K', type=int, default=3, help='Polar code parameter K')
    parser.add_argument('--hard_decision', dest = 'hard_decision', default=False, action='store_true')

    """ Training parameters """
    parser.add_argument('-batch_size', type=int, default=10000, help='minibatch size')
    parser.add_argument('-dec_train_snr', type=float, default=0., help='training channel snr')
    parser.add_argument('-power_constraint_type', type=str, default='hard_power_block', help='typer of power constraint')
    parser.add_argument('-method', type=str, default='pgd', choices=['pgd', 'alm'], help='method to optimize p, with constraint')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-num_steps', type=int, default=2000)
    parser.add_argument('-num_runs', type=int, default=1)
    parser.add_argument('-optimizer_method', type=str, default='adam', choices=['gradient_descent', 'adam', 'rmsprop'], help='loss function')
    parser.add_argument('-loss', type=str, default='mse', choices=['bce', 'mse'], help='loss function')

    """ Testing parameters """
    parser.add_argument('-test_snr_start', type=float, default=-10., help='test snr start')
    parser.add_argument('-test_snr_end', type=float, default=10., help='test snr end')
    parser.add_argument('-test_ratio', type = float, default = 10, help = 'Number of test samples x batch_size')

    """ Frozen bit learning parameters """
    parser.add_argument('-p_init', type=str, default='p49', choices=['p49', 'equal','random', 'local_opt', 'ffn', 'tv', 'custom'], help='how to initialize p')
    parser.add_argument('-p_init_path', type=str, default='../learn_polar_files/p_input.p', help='path to pickle file containing p tensor to initialize training (if args.p_init == custom)')
    parser.add_argument('-bernoulli_beta', type=float, default=0.01)
    parser.add_argument('-pgd_eps', type=float, default=-1)
    parser.add_argument('-pgd_lambda', type=float, default=1)
    parser.add_argument('-pgd_step_size', type=float, default=0.0001)
    parser.add_argument('-pgd_num_steps', type=int, default=1000)
    parser.add_argument('-pgd_update_steps', type=int, default=1)
    parser.add_argument('-pgd_constraint', type=str, default='L1', choices = ['entropy', 'L1'])

    parser.add_argument('-alm_lambda', type=float, default=0)
    parser.add_argument('-alm_c', type=float, default=0)
    parser.add_argument('-alm_c_max', type=float, default=0.5)
    parser.add_argument('-alm_lamb_max', type=float, default=1)
    parser.add_argument('--alm_add', dest = 'alm_add', default=False, action='store_true')
    parser.add_argument('-alm_factor', type=float, default=1)
    parser.add_argument('-alm_update_iters', type=int, default=1000)
    parser.add_argument('-alm_uncon', type=int, default=0)

    parser.add_argument('--compute_bers', dest = 'compute_bers', default=False, action='store_true', help='compute and show BERs every 1000 steps')
    parser.add_argument('-gpu', type=int, default=0, help='gpus used for training - e.g 0,1,3')
    parser.add_argument('--verbose', dest = 'verbose', default=False, action='store_true')

    args = parser.parse_args()

    return args

class LearnFrozen(PolarCode):
    def __init__(self, n, K, args, F=None, rs= None, use_cuda=True):
        super().__init__(n, K, args, F, rs, use_cuda)

    def sc_decode_with_prior(self, noisy_codes, snr, p, scaling = None):
        # Successive cancellation decoder for polar codes

        # priors = torch.zeros(noisy_codes.shape[0], self.N, dtype=torch.float)
        # assuming k = N in this case
        # priors = p[self.info_positions]
        priors = p
        noise_sigma = snr_db2sigma(snr)
        llrs = (2/noise_sigma**2)*noisy_codes
        if scaling is not None:
            llrs = llrs * (scaling*np.sqrt(self.args.N)/torch.norm(scaling))
        assert noisy_codes.shape[1] == self.N
        decoded_bits = torch.zeros(noisy_codes.shape[0], self.N)
        decoded_llrs = torch.zeros(noisy_codes.shape[0], self.N)
        depth = 0

        # function is recursively called (DFS)
        # arguments: Beliefs at the input of node (LLRs at top node), depth of children, bit_position (zero at top node)
        decoded_codeword, decoded_bits, decoded_llrs = self.decode_with_prior(llrs, depth, 0, priors, decoded_bits, decoded_llrs)
        # decoded_message = torch.sign(decoded_bits)[:, self.info_positions]
        soft_decoded = decoded_llrs[:, self.info_positions]
        decoded_message = decoded_bits[:, self.info_positions]


        return decoded_message, soft_decoded

    def decode_with_prior(self, llrs, depth, bit_position, p, decoded_bits=None, decoded_llrs = None):
        # Function to call recursively, for SC decoder
        quantize = STEQuantize()
        # print("DEPTH = {}, bit_position = {}".format(depth, bit_position))
        half_index = 2 ** (self.n - depth - 1)

        # n = 2 tree case
        if depth == self.n - 1:
            # Left child
            left_bit_position = 2*bit_position
            if left_bit_position in self.frozen_positions:
                # If frozen decoded bit is 0
                u_hat = torch.ones_like(llrs[:, :half_index], dtype=torch.float)
                Lu = 1000*torch.ones_like(llrs[:, :half_index], dtype=torch.float)
            else:
                Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1)).sum(dim=1, keepdim=True)
                # Adding prior from the sampling probabilities of message bits (-ve because of our BPSK convention)
                p_pos_u = np.where(self.info_positions == left_bit_position)
                prior = torch.log((p[left_bit_position])/(1 - p[left_bit_position]))
                Lu = Lu - torch.ones_like(Lu)*prior.item()

                if self.args.hard_decision:
                    # u_hat = torch.sign(Lu)
                    u_hat = quantize.apply(Lu)
                else:
                    u_hat = torch.tanh(Lu/2)

            # Right child
            right_bit_position = 2*bit_position + 1
            if right_bit_position in self.frozen_positions:
                # If frozen decoded bit is 0
                v_hat = torch.ones_like(llrs[:, :half_index], dtype = torch.float)
                Lv = 1000*torch.ones_like(llrs[:, :half_index], dtype=torch.float)
            else:
                Lv = u_hat * llrs[:, :half_index] + llrs[:, half_index:]
                p_pos_v = np.where(self.info_positions == right_bit_position)
                prior = torch.log((p[right_bit_position])/(1 - p[right_bit_position]))
                Lv = Lv - torch.ones_like(Lv)*prior.item()

                if self.args.hard_decision:
                    # v_hat = torch.sign(Lv)
                    v_hat = quantize.apply(Lv)
                else:
                    v_hat = torch.tanh(Lv/2)


            # print("DECODED: Bit positions {} : {} and {} : {}. Lu = {}, Lv = {}".format(left_bit_position, u_hat[0], right_bit_position, v_hat[0], Lu[0], Lv[0]))

            decoded_bits[:, left_bit_position] = u_hat.squeeze(1)
            decoded_bits[:, right_bit_position] = v_hat.squeeze(1)
            decoded_llrs[:, left_bit_position] = Lu.squeeze(1)
            decoded_llrs[:, right_bit_position] = Lv.squeeze(1)

            return torch.cat((u_hat * v_hat, v_hat), dim = 1).float(), decoded_bits, decoded_llrs

        # General case
        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u
            Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))
            u_hat, decoded_bits, decoded_llrs = self.decode_with_prior(Lu, depth+1, bit_position*2, p, decoded_bits, decoded_llrs)

            # RIGHT CHILD
            Lv = u_hat * llrs[:, :half_index] + llrs[:, half_index:]
            v_hat, decoded_bits, decoded_llrs = self.decode_with_prior(Lv, depth+1, bit_position*2 + 1, p, decoded_bits, decoded_llrs)

            # print('decoded ', decoded_bits[0])
            # print('Lu : {}, Lv : {}'.format( Lu[0], Lv[0]))
            return torch.cat((u_hat * v_hat, v_hat), dim=1), decoded_bits, decoded_llrs


def elementwise_entropy(p):
    clamp_class = Clamp()
    clamp = clamp_class.apply

    q = 1 - p
    p = clamp(p)
    q = clamp(q)
    entropy = -(p*torch.log2(p)+ q*torch.log2(q))
#    entropy[p<=0] = 0
#    entropy[p>=1] = 0

    return entropy

def projected_gradient_descent(p_hat, K, lamb=1, step_size = 0.00001, num_steps = 5000, optimizer = 'gd', verbose=False):
    p_hat = p_hat.clone().detach()
    # p = torch.rand(args.N).requires_grad_(True)
    p = p_hat.clone().detach().requires_grad_(True)

    #lamb = 1
    clamp_class = Clamp()
    clamp = clamp_class.apply
    entropy = elementwise_entropy(p)
    sum_entropy = torch.sum(entropy)

    saved = {}
    if (sum_entropy >= K):
        return p_hat, saved

    saved['loss'] = []
    saved['entropy'] = []
    saved['p'] = None
    saved['mse'] = []
    saved['cons_loss'] = []
    for step in range(num_steps):
        entropy = elementwise_entropy(p)
        sum_entropy = torch.sum(entropy)
        mse = torch.norm(p_hat - p)
        constraint_loss = lamb* max(0, (K - sum_entropy))

        if constraint_loss == 0:
            loss = mse
        else:
            loss = mse + constraint_loss

        saved['mse'].append(mse.item())
        saved['cons_loss'].append(constraint_loss)
        saved['entropy'].append(sum_entropy.item())
        saved['loss'].append(loss)
        if verbose:
            print('step: {}, p = {}, entropy = {}, loss = {}, mse = {}, con = {}'.format(step, p.data, sum_entropy, loss, mse, constraint_loss))
        if saved['p'] is None:
            saved['p'] = p.unsqueeze(0)
        else:
            saved['p'] = torch.cat((saved['p'], p.unsqueeze(0)), dim=0)
        loss.backward()
        p.data = clamp(p - step_size*p.grad)
        p.grad.zero_()
    ind = np.argmin(saved['loss'])
    p = saved['p'][ind]
            # if step%10 == 0:
                # print(step, 'loss ',loss.item(), ' norm :',torch.norm(p_hat - p).item(), 'entropy: ', sum_entropy.item())
    return p, saved


def projected_gradient_descent_L1(p_hat, K, lamb=1, step_size = 0.00001, num_steps = 1000, optimizer = 'gd'):
    p_hat = p_hat.clone().detach()
    # p = torch.rand(args.N).requires_grad_(True)
    p = p_hat.clone().detach().requires_grad_(True)
    if optimizer == 'adam':
        pgd_optimizer = optim.Adam([p], lr=step_size)

    #lamb = 1
    clamp_class = Clamp()
    clamp = clamp_class.apply

    sum_p = torch.sum(p)

    saved = {}
    if (sum_p >= K/2 and not (p>0.5).any()):
        return p_hat, saved

    saved['loss'] = []
    saved['sum'] = []
    saved['p'] = None
    saved['mse'] = []
    saved['cons_loss'] = []
    for step in range(num_steps):
        sum_p = torch.sum(p)
        mse = torch.norm(p_hat - p)
        constraint_loss = lamb* max(0, (K/2 - sum_p))
        p5_loss = lamb* torch.sum(  torch.max(torch.zeros_like(p),  p - 0.5))
        loss = mse + constraint_loss + p5_loss

        saved['mse'].append(mse.item())
        saved['cons_loss'].append(constraint_loss)
        saved['sum'].append(sum_p.item())
        saved['loss'].append(loss)
        if saved['p'] is None:
            saved['p'] = p.unsqueeze(0)
        else:
            saved['p'] = torch.cat((saved['p'], p.unsqueeze(0)), dim=0)
        loss.backward()
        if optimizer == 'adam':
            pgd_optimizer.step()
            p.data = clamp(p)
            pgd_optimizer.zero_grad()
        elif optimizer == 'gd':
            p.data = clamp(p - step_size*p.grad)
            p.grad.zero_()
    ind = np.argmin(saved['loss'])
    p = saved['p'][ind]
            # if step%10 == 0:
                # print(step, 'loss ',loss.item(), ' norm :',torch.norm(p_hat - p).item(), 'entropy: ', sum_entropy.item())
    return p, saved


def test_ber(args, p, beta=0.001, snr_start = 0, snr_end = 10, only_info = False):
    beta = torch.Tensor([beta])
    bern = RelaxedBernoulli(beta, p)
    bers = {}
    blers = {}

    F1 = p.detach().numpy().argsort()[:args.N-args.K]
    I = p.detach().numpy().argsort()[args.N-args.K:]

    for r in range(int(args.test_ratio)):
        msg_bits = bern.rsample(torch.Size([args.batch_size]))
        msg_bits_bpsk = 1 - 2*msg_bits
        msg_bits_bpsk = quantize.apply(msg_bits_bpsk)

        polar = LearnFrozen(n, args.N, args)
        codes = polar.encode_plotkin(msg_bits_bpsk)

        for snr in np.arange(snr_start, snr_end+1):
            noisy_codes = polar.channel(codes, snr)
            decoded, llrs = polar.sc_decode_with_prior(noisy_codes, snr, p)
            if only_info:
                errors = (decoded[:,I].sign() != msg_bits_bpsk[:,I].sign()).float()
            else:
                errors = (decoded.sign() != msg_bits_bpsk.sign()).float()
            ber = (torch.sum(errors)/(errors.shape[0]*errors.shape[1]))
            bler = torch.sum((torch.sum(errors, dim=1)>0).float())/errors.shape[0]
            # print("SNR = {}, BER = {}, BLER = {}".format(snr, ber, bler))
            if r== 0:
                bers[snr] = ber.item()
                blers[snr] = bler.item()
            else:
                bers[snr] += ber.item()
                blers[snr] += bler.item()
    bers = {key:(value/args.test_ratio) for (key,value) in bers.items()}
    blers = {key:(value/args.test_ratio) for (key,value) in blers.items()}
    return bers, blers

class BernoulliSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, batch_size):

        ctx.save_for_backward(p)
        U = torch.rand(batch_size, p.shape[0])
        b = (U<p).float()

        return b

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        return grad_input, None

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, p):

        # ctx.save_for_backward(U)
        # U = torch.rand(batch_size, p.shape[0])
        b = (U<p).float()

        return b

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        return grad_input, None

if __name__ == '__main__':
    args = get_args()
    if args.pgd_constraint == 'entropy':
        proj = 'ent'
    elif args.pgd_constraint == 'L1':
        if args.pgd_eps < 0 :
            proj = 'L1'
        else:
            proj = 'L1_eps'

    n = int(np.log2(args.N))
    clamp_class = Clamp()
    clamp = clamp_class.apply
    quantize = STEQuantize()

    """Computing Tal-Vardy frozen bits"""
    # computed for SNR = 0
    if n == 6:
        rs = np.array([63, 62, 61, 59, 55, 47, 31, 60, 58, 57, 54, 53, 51, 46, 45, 43, 30,
           29, 39, 27, 56, 23, 52, 50, 44, 49, 15, 42, 41, 28, 38, 26, 37, 25,
           22, 35, 21, 48, 14, 19, 40, 13, 11, 36, 24, 34,  7, 20, 33, 18, 12,
           17, 10,  9,  6,  5, 32,  3, 16,  8,  4,  2,  1,  0])
    elif n == 5:
        rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

    elif n == 4:
        rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
    elif n == 3:
        rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
    elif n == 2:
        rs = np.array([3, 2, 1, 0])

    else:
        rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1

    rs = rs[rs<args.N]
    F2 = rs.copy()[args.K:]
    F2.sort()
    p_new = 0.5*torch.ones(args.N)
    p_new[F2] = 0
    ber_polar, bler_polar = test_ber(args, clamp(p_new), beta = args.bernoulli_beta, only_info=True)


    # polar = PolarCode(n, args.N, args)
    polar = LearnFrozen(n, args.N, args)

    num_runs = args.num_runs
    saved = None
    init_end = {}
    datas = {}
    en = None

    for run1 in range(num_runs):
        try:
            # INITIALIZE p
            if args.p_init == 'random':
                p = (0.5*torch.rand(args.N, dtype=torch.float)).requires_grad_(True) # parameter to be learnt
            # p = torch.Tensor([0.5, 0, 0.5, 0.5]).requires_grad_(True)
            elif args.p_init == 'p49':
                p = 0.49*torch.ones(args.N)
                p.requires_grad_(True)
            elif args.p_init == 'equal':
                e = args.K/(2*args.N)
                p = e*torch.ones(args.N)
                p.requires_grad_(True)
            elif args.p_init == 'local_opt':
                print("Enter {} frozen bits from 0 to {}".format(args.N - args.K, args.N-1))
                frozen_list = []
                for i in range(0, args.N-args.K):
                    ele = int(input())
                    assert ele <args.N
                    frozen_list.append(ele)
                p = 0.5*torch.ones(args.N)
                p[np.array(frozen_list)] = 0
                p.requires_grad_(True)
            elif args.p_init == 'ffn':
                # fix first N
                if en is None:
                    print("Enter N (first N bits to freeze initially):")
                    en = int(input())
                # else use the input from previous instance
                p = 0.5*torch.ones(args.N)
                p[:en] = 0
                p.requires_grad_(True)

            elif args.p_init == 'tv':
                p = p_new.clone()
                p.requires_grad_(True)

            elif args.p_init == 'custom':

                p = torch.load(open(args.p_init_path,'rb'))
                if p.shape[0] != args.N:
                    p = 0.49*torch.ones(args.N)
                p.requires_grad_(True)
            print("Started : p_init = {}".format(p))
            beta = torch.Tensor([args.bernoulli_beta])
            num_steps = args.num_steps
            if args.optimizer_method == 'gradient_descent':
                step_size = 0.0001
                min_lr = 0.000001
                reduce_lr_patience = 10
                bad_steps = 0
                min_loss = 1000000

            elif args.optimizer_method == 'adam': #default
                optimizer =optim.Adam([p], lr = args.lr)
            elif args.optimizer_method == 'rmsprop':
                optimizer =optim.RMSprop([p], lr = args.lr)


            init_end[run1] = {}
            init_end[run1]['init'] = p



            data = {}
            data['sum_entropy'] = []
            data['sum_p'] = []
            data['p'] = None
            data['p_grad'] = None
            data['entropy'] = None
            data['loss'] = []
            data['f'] = []
            data['lambda'] = []
            data['c'] = []
            start_time = time.time()
            for step in range(num_steps):

                p.data = clamp(p, 0, 0.5)
                # p.data = clamp(p)
                # Sample from Bernoulli distribution with Gumbel-Max reparametrization trick
                bern = RelaxedBernoulli(beta, p)
                msg_bits = bern.rsample(torch.Size([args.batch_size])).requires_grad_(True)
                msg_bits_bpsk = 1 - 2*msg_bits
                # Using a STE to quantize message bits to +1 or -1.
                msg_bits_bpsk = quantize.apply(msg_bits_bpsk)
                codes = polar.encode_plotkin(msg_bits_bpsk)

                snr = args.dec_train_snr
                noisy_codes = polar.channel(codes, snr)
                # decoded = polar.sc_decode(noisy_codes, snr)
                decoded_bits, decoded_llrs = polar.sc_decode_with_prior(noisy_codes, snr, p)
                if args.loss == 'mse':
                    loss = F.mse_loss(decoded_bits, msg_bits_bpsk)
                elif args.loss == 'bce':
                    loss = F.binary_cross_entropy_with_logits(decoded_llrs, 0.5+ 0.5*msg_bits_bpsk)
                entropy = elementwise_entropy(p)
                sum_entropy = torch.sum(entropy)
                sum_p = torch.sum(p)
                if torch.isnan(sum_entropy):
                    print("Nan at step {}".format(step))
                    break

                if args.method == 'alm':
                    if step == 0:
                        lamb = args.alm_lambda
                        c = args.alm_c
                        print('Using ALM with {} constraint. c = {}, lambda= {}'.format(args.pgd_constraint, args.alm_c, args.alm_lambda))
                    elif step < args.alm_uncon:
                        c = 0
                        lamb = 0
                    elif step == args.alm_uncon:
                        c = args.alm_c
                        lamb = args.alm_lambda
                    elif step%args.alm_update_iters == 0:
                        if c < args.alm_c_max and args.alm_c >0:
                            if args.alm_add:
                                c = c+ args.alm_factor
                            else:
                                c = c*args.alm_factor
                            if args.verbose:
                                print('c changed to {}'.format(c))
                        else:
                            if args.verbose and args.alm_c > 0:
                                print('c unchanged at {}'.format(c))

                        if lamb < args.alm_lamb_max and args.alm_lambda > 0:
                            if args.alm_add:
                                lamb = lamb + args.alm_factor
                            else:
                                lamb = lamb*args.alm_factor
                            if args.verbose:
                                print('lamb changed to {}'.format(lamb))
                        else:
                            if args.verbose:
                                pass
                                # print('c unchanged at {}'.format(c))
                    # Lagrangen
                    # if args.pgd_constraint == 'entropy':
                    #     f = loss - lamb*(sum_entropy - args.K)
                    # elif args.pgd_constraint == 'L1':
                    #     f = loss + lamb*max(0, (args.K/2 - sum_p))

                    # Augmented Lagrangen
                    if args.pgd_constraint == 'entropy':
                        f = loss - lamb*(sum_entropy - args.K) + c*(sum_entropy - args.K)**2
                    elif args.pgd_constraint == 'L1':
                        f = loss + lamb*max(0, (args.K/2 - sum_p)) + c*(max(0, (args.K/2 - sum_p))**2)

                    # Equality constraint
                    # if args.pgd_constraint == 'entropy':
                    #     f = loss + c*(sum_entropy - args.K)**2
                    # elif args.pgd_constraint == 'L1':
                    #     f = loss + c*(max(0, (args.K/2 - sum_p))**2)

                else:
                    f = loss
                f.backward()

                data['sum_entropy'].append(sum_entropy.item())
                data['sum_p'].append(sum_p.item())
                data['loss'].append(loss.item())
                data['f'].append(f.item())
                if args.method=='alm':
                    data['lambda'].append(lamb)
                    data['c'].append(c)
                if data['p'] is None:
                    data['p'] = p.unsqueeze(0)
                    data['p_grad'] = p.grad.unsqueeze(0)
                #     data['entropy'] = entropy.unsqueeze(0)
                else:
                    data['p'] = torch.cat((data['p'], p.unsqueeze(0)), dim=0)
                    data['p_grad'] = torch.cat((data['p_grad'], p.grad.unsqueeze(0)), dim=0)
                #     data['entropy'] = torch.cat((data['entropy'], entropy.unsqueeze(0)), dim=0)

                if args.verbose:
                    if step%100 == 0:
                        print('Step: {}, Loss: {}, {} : {}, p: {}'.format(step, loss.item(), args.pgd_constraint, sum_entropy if args.pgd_constraint == 'entropy' else sum_p, p.data))
                        F1 = p.clone().detach().numpy().argsort()[:args.N-args.K]
                        F1.sort()
                        print("Frozen bits are: {}".format(list(F1)))




                        if step%1000 == 0 and step>0 and args.compute_bers:
                            p_test = 0.5*torch.ones(args.N)
                            p_test[F1] = 0
                            bers, blers = test_ber(args, clamp(p_test), beta = args.bernoulli_beta)
                            print("BERs : {}".format(bers))

                    if step == 0:
                        print("First gradient: {}".format(p.grad))
                if step%1000 == 0 and step>0:
                    datas[run1] = data
                    torch.save(datas, open('../learn_polar_files/data_{}_{}_{}_{}_{}.p'.format(args.loss, args.N, args.K, args.p_init, proj), 'wb'))
                #GD
                if args.optimizer_method == 'gradient_descent':
                    p.data = clamp(p - step_size*p.grad, 0, 0.5)
                    p.grad.zero_()
                    if loss.item() <= min_loss:
                        min_loss = loss.item()
                        bad_steps = 0
                    else:
                        bad_steps += 1
                        if bad_steps == reduce_lr_patience:
                            if step_size > min_lr:
                                step_size = step_size/10
                                bad_steps = 0
                                print('LR reduced to {} at step {}. Min loss was {}. Current loss = {}'.format(step_size, step, min_loss, loss.item()))
                                min_loss = loss.item()



                #Adam
                elif args.optimizer_method == 'adam' or args.optimizer_method == 'rmsprop':
                    optimizer.step()
                    p.data = clamp(p)
                    optimizer.zero_grad()

                if args.method == 'pgd':
                    # Project it to constraint:
                    if step%args.pgd_update_steps == args.pgd_update_steps-1:
                        if args.pgd_constraint == 'entropy':

                            pgd_out, saved1 = projected_gradient_descent(p, args.K, step_size = args.pgd_step_size, lamb=args.pgd_lambda, num_steps=args.pgd_num_steps, optimizer='gd')
                            if bool(saved1):
                                saved = saved1
                            # if (pgd_out != p.data).any():
                            #     print("Projected {} to {}. Entropy = {} to {}".format(p.data, pgd_out, sum_entropy, torch.sum(elementwise_entropy(pgd_out))))
                        elif args.pgd_constraint == 'L1':
                            sum_p = torch.sum(p)

                            if args.pgd_eps < 0:
                                #closed form projection : need to check
                                p_hat = p + max(0, (args.K/2 - sum_p)/args.N)
                                # x_inds = torch.where(p_hat<=1e-10)

                                #if pgd going above 0.5, constrain it to 0.5
                                y_inds = torch.where(p_hat>=0.5)
                                # inds = torch.where((p_hat<0.5) & (p_hat>1e-10))


                                inds = torch.where((p_hat<0.5))
                                p_hat[y_inds] = 0.5
                                p_hat[inds] = p[inds] + max(0, (args.K/2 - 0.5*len(y_inds[0]) - torch.sum(p[inds]))/len(inds[0]))
                                pgd_out = p_hat.clone()
                                # pgd_out, saved1 = projected_gradient_descent_L1(p, args.K, step_size = args.pgd_step_size, lamb=args.pgd_lambda, num_steps=args.pgd_num_steps, optimizer='gd')
                                # if bool(saved1):
                                #     saved = saved1
                                # pgd_out = clamp(p - min(0, (sum_p - args.K/2)/args.N), 0, 0.5)
                                # pgd_out = clamp(p - (sum_p - args.K/2)/args.N, 0, 0.5)
                                # if (pgd_out != p.data).any():
                                #     print("Projected {} to {}. Sum_p = {} to {}".format(p.data, pgd_out.data, sum_p.item(), torch.sum(pgd_out).item()))
                            else:
                                # if p_i is in (0, epsilon) or (0.5 - epsilon, 0.5), don't consider it for projection.
                                assert args.pgd_eps < 0.5

                                eps_inds_z = torch.where((p >= 0) & (p < args.pgd_eps))
                                eps_inds_p5 = torch.where(p >= 0.5 - args.pgd_eps)
                                non_eps_inds = torch.where((p < 0.5 - args.pgd_eps) & (p > args.pgd_eps))

                                p_hat = p.clone()
                                p_hat[non_eps_inds] = p[non_eps_inds] + max(0, (args.K/2 - 0.5*len(eps_inds_p5[0]) - torch.sum(p[non_eps_inds]))/len(non_eps_inds[0]))
                                p_hat[eps_inds_p5] = torch.min(0.5*torch.ones_like(p_hat[eps_inds_p5]), p_hat[eps_inds_p5])
                                p_hat[eps_inds_z] = torch.max(torch.zeros_like(p_hat[eps_inds_z]), p_hat[eps_inds_z])

                                y_inds = torch.where(p_hat>0.5)
                                inds = torch.where((p_hat<=0.5) & (p < 0.5 - args.pgd_eps) & (p > args.pgd_eps))
                                p_hat[y_inds] = 0.5
                                p_hat[inds] = p[inds] + max(0, (args.K/2 - 0.5*len(y_inds[0]) - 0.5*len(eps_inds_p5[0]) - torch.sum(p[inds]))/len(inds[0]))
                                pgd_out = p_hat.clone()

                        p.data = pgd_out
            if args.method == 'pgd':
                p_opt = data['p'][np.argmin(data['loss'][1:])+1]
            else:
                p_opt = data['p'][-1]

            print("Run {}, p = {}, {} = {}".format(run1, p_opt.data, args.pgd_constraint, sum_entropy if args.pgd_constraint == 'entropy' else sum_p))
            print('Time taken = {} minutes'.format((time.time()-start_time)/60))

            F1 = p_opt.detach().numpy().argsort()[:args.N-args.K]
            F1.sort()
            print("\nFrozen bits are: {}".format(list(F1)))
            print("Tal-Vardy frozen bits are: {}".format(list(F2)))

            p_test = 0.5*torch.ones(args.N)
            p_test[F1] = 0

            bers, blers = test_ber(args, clamp(p_test), beta = args.bernoulli_beta, only_info=True)
            print("BERs : {}, BLERs: {}".format(bers, blers))
            print("Tal-Vardy BERs : {}, BLERs: {}\n".format(ber_polar, bler_polar))

            init_end[run1]['end'] = data['p'][np.argmin(data['loss'][1:])+1]
            torch.save(init_end, open('../learn_polar_files/init_end_{}_{}_{}_{}_{}.p'.format(args.loss, args.N, args.K, args.p_init, proj), 'wb'))
            datas[run1] = data
            torch.save(datas, open('../learn_polar_files/data_{}_{}_{}_{}_{}.p'.format(args.loss, args.N, args.K, args.p_init, proj), 'wb'))
        except KeyboardInterrupt:
            print('Graceful Exit')
            if args.method == 'pgd':
                p_opt = data['p'][np.argmin(data['loss'][1:])+1]
            else:
                p_opt = data['p'][-1]
            print("Run {}, p = {}, {} = {}".format(run1, p_opt.data, args.pgd_constraint, sum_entropy if args.pgd_constraint == 'entropy' else sum_p))
            print('Time taken = {} minutes'.format((time.time()-start_time)/60))
            F1 = p_opt.detach().numpy().argsort()[:args.N-args.K]
            F1.sort()
            print("Frozen bits are: {}".format(list(F1)))
            print("Tal-Vardy frozen bits are: {}".format(list(F2)))

            p_test = 0.5*torch.ones(args.N)
            p_test[F1] = 0

            bers, blers = test_ber(args, clamp(p_test), beta = args.bernoulli_beta, only_info=True)
            print("BERs : {}, BLERs: {}".format(bers, blers))
            print("Tal-Vardy BERs : {}, BLERs: {}\n".format(ber_polar, bler_polar))

            init_end[run1]['end'] = data['p'][np.argmin(data['loss'][1:])+1]
            torch.save(init_end, open('../learn_polar_files/init_end_{}_{}_{}_{}_{}.p'.format(args.loss, args.N, args.K, args.p_init, proj), 'wb'))
            datas[run1] = data
            torch.save(datas, open('../learn_polar_files/data_{}_{}_{}_{}_{}.p'.format(args.loss, args.N, args.K, args.p_init, proj), 'wb'))





    # elif args.method == 'alm':
    #     for run1 in range(num_runs):
    #         # p = (0.5*torch.rand(args.N, dtype=torch.float)).requires_grad_(True) # parameter to be learnt
    #         p = torch.Tensor([0, 0.5, 0.5, 0.5], dtype=torch.float).requires_grad_(True)
    #         # p = 0.49*torch.ones(args.N)
    #         p.requires_grad_(True)
    #         print("Started")
    #         beta = torch.Tensor([0.001])
    #         num_steps = args.num_steps
    #         if args.optimizer_method == 'gradient_descent':
    #             step_size = 0.0001
    #             min_lr = 0.000001
    #             reduce_lr_patience = 10
    #             bad_steps = 0
    #             min_loss = 1000000
    #
    #         elif args.optimizer_method == 'adam': #default
    #             optimizer =optim.Adam([p], lr = 0.0003)
    #         lamb = (0.2*torch.ones(1)).requires_grad_(True)
    #
    #         clamp_class = Clamp()
    #         clamp = clamp_class.apply
    #
    #
    #         c = 1
    #         c_factor = 2
    #         c_update_steps = 100
    #         c_max = 1000
    #
    #         data = {}
    #         data['sum_entropy'] = []
    #         data['lambda'] = []
    #         data['lambda_grad'] = []
    #         data['p'] = None
    #         data['p_grad'] = None
    #         data['entropy'] = None
    #         data['c'] = []
    #         # data['loss'] = []
    #         start_time = time.time()
    #         for step in range(num_steps):
    #
    #             p.data = clamp(p)
    #             # Sample from Bernoulli distribution with reparametrization trick
    #             bern = RelaxedBernoulli(beta, p)
    #             msg_bits = bern.rsample(torch.Size([args.batch_size])).requires_grad_(True)
    #             msg_bits_bpsk = 1 - 2*msg_bits
    #             codes = polar.encode_plotkin(msg_bits_bpsk)
    #
    #             snr = args.dec_train_snr
    #             noisy_codes = polar.channel(codes, snr)
    #             # decoded = polar.sc_decode(noisy_codes, snr)
    #             decoded = polar.sc_decode_with_prior(noisy_codes, snr, p)
    #             if args.loss == 'mse':
    #                 loss = F.mse_loss(decoded, msg_bits_bpsk)
    #             elif args.loss_functio == 'bce':
    #                 loss = F.binary_cross_entropy_with_logits(decoded, 0.5+ 0.5*msg_bits_bpsk)
    #             entropy = elementwise_entropy(p)
    #             sum_entropy = torch.sum(entropy)
    #             if torch.isnan(sum_entropy):
    #                 print("Nan at step {}".format(step))
    #                 break
    #
    #
    #             # # Constraints : sum_entropy >= K, p belongs to [0,1]
    #             # # Augmented Lagrangen method
    #
    #             # Lagrangen
    #             # f = loss - lamb*(sum_entropy - args.K)
    #
    #             # Augmented Lagrangen
    #             f = loss - lamb*(sum_entropy - args.K) + c*(sum_entropy - args.K)**2
    #
    #             # Equality constraint
    #             # f = loss + c*(sum_entropy - args.K)**2
    #
    #             if step%c_update_steps == c_update_steps-1:
    #                 if c*c_factor <= c_max:
    #                     c = c*c_factor
    #
    #
    #             # loss.backward()
    #             f.backward()
    #
    #             # step_size_lambda = 0.001
    #             # if step%100 == 0:
    #             #     print('Step: {}'.format(step))
    #             #     print('p ', p, 'lambda', lamb)
    #             #     print('p.grad ', p.grad, 'lambda.grad', lamb.grad)
    #             #     print('entropy sum = ', sum_entropy, 'f =' , f, 'c =', c, '\n')
    #
    #             # data['c'].append(c)
    #             # data['lambda'].append(lamb.item())
    #             # # data['lambda_grad'].append(lamb.grad.item())
    #             data['sum_entropy'].append(sum_entropy.item())
    #             data['loss'].append(loss.item())
    #             if data['p'] is None:
    #                 data['p'] = p.unsqueeze(0)
    #             #     data['p_grad'] = p.grad.unsqueeze(0)
    #             #     data['entropy'] = entropy.unsqueeze(0)
    #             else:
    #                 data['p'] = torch.cat((data['p'], p.unsqueeze(0)), dim=0)
    #             #     data['p_grad'] = torch.cat((data['p_grad'], p.grad.unsqueeze(0)), dim=0)
    #             #     data['entropy'] = torch.cat((data['entropy'], entropy.unsqueeze(0)), dim=0)
    #
    #
    #             if step%1 == 0:
    #                 print('Step: {}, Loss: {}, Entropy: {}, p: {}'.format(step, loss.item(), sum_entropy, p.data))
    #             if step == 0:
    #                 print("First gradient: {}".format(p.grad))
    #             #GD
    #             if args.optimizer_method == 'gradient_descent':
    #                 p.data = clamp(p - step_size*p.grad, 0, 0.5)
    #                 if loss.item() <= min_loss:
    #                     min_loss = loss.item()
    #                     bad_steps = 0
    #                 else:
    #                     bad_steps += 1
    #                     if bad_steps == reduce_lr_patience:
    #                         if step_size > min_lr:
    #                             step_size = step_size/10
    #                             bad_steps = 0
    #                             print('LR reduced to {} at step {}. Min loss was {}. Current loss = {}'.format(step_size, step, min_loss, loss.item()))
    #                             min_loss = loss.item()
    #
    #
    #
    #             #Adam
    #             elif args.optimizer_method == 'adam':
    #                 optimizer.step()
    #                 p.data = clamp(p)
    #             # lamb.data = clamp(lamb - step_size_lambda*lamb.grad, lamb.item()/3, 3*lamb.item())
    #             # lamb.data = lamb - 2*c*(sum_entropy - args.K)
    # #             lamb.data = clamp(lamb - 2*c*(sum_entropy - args.K), lamb.item()/3, 3*lamb.item())
    #
    #         print("Run {}, p = {}, Entropy = {}".format(run1, p.data, sum_entropy))
    #         print('Time taken = {} minutes'.format((time.time()-start_time)/60))
    #         datas[run1] = data
