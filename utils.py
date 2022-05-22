import torch
import numpy as np
from collections import Counter, OrderedDict

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def get_msg_bits_batch(data_generator):
    msg_bits_batch = next(data_generator)
    return msg_bits_batch

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def errors_ber(y_true, y_pred, mask=None):
    if mask == None:
        mask=torch.ones(y_true.size(),device=y_true.device)
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)
    mask = mask.view(mask.shape[0], -1, 1)
    myOtherTensor = (mask*torch.ne(torch.round(y_true), torch.round(y_pred))).float()
    res = sum(sum(myOtherTensor))/(torch.sum(mask))
    return res

def errors_bitwise_ber(y_true, y_pred, mask=None):
    if mask == None:
        mask=torch.ones(y_true.size(),device=y_true.device)
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)
    mask = mask.view(mask.shape[0], -1, 1)
    myOtherTensor = (mask*torch.ne(torch.round(y_true), torch.round(y_pred))).float()
    res = torch.sum(myOtherTensor,0)/torch.sum(mask,0)
    return res

def errors_bler(y_true, y_pred, get_pos = False):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test       = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])

    if not get_pos:
        return bler_err_rate
    else:
        err_pos = list(np.nonzero((np.sum(tp0,axis=1)>0).astype(int))[0])
        return bler_err_rate, err_pos
    
def extract_block_errors(y_true, y_pred, thresh=0):
    y_true_out = y_true.clone()
    y_pred_out = y_pred.clone()
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test       = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_err_rate = (np.sum(tp0,axis=1)>thresh)*1.0
    return np.where(bler_err_rate > 0)
    
def extract_block_nonerrors(y_true, y_pred, thresh=1):
    y_true_out = y_true.clone()
    y_pred_out = y_pred.clone()
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test       = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_correct_rate = (np.sum(tp0,axis=1)<thresh)*1.0
    return np.where(bler_correct_rate > 0)

def get_epos(k1, k2):
    # return counter for bit ocations of first-errors
    bb = torch.ne(k1.cpu().sign(), k2.cpu().sign())
    # inds = torch.nonzero(bb)[:, 1].numpy()
    idx = []
    for ii in range(bb.shape[0]):
        try:
            iii = list(bb.cpu().float().numpy()[ii]).index(1)
            idx.append(iii)
        except:
            pass
    counter = Counter(idx)
    ordered_counter = OrderedDict(sorted(counter.items()))
    return ordered_counter

def countSetBits(n):
    count = 0
    while (n):
        n &= (n-1)
        count+= 1
    return count

def get_minD(code):
    all_msg_bits = []

    for i in range(2**code.K):
        d = dec2bitarray(i, code.K)
        all_msg_bits.append(d)
    all_message_bits = torch.from_numpy(np.array(all_msg_bits))
    all_message_bits = 1 - 2*all_message_bits.float()

    codebook = 0.5*code.encode(all_message_bits)+0.5
    b_codebook = codebook.unsqueeze(0)
    dist = 1000
    for ii in range(codebook.shape[0]):
        a = ((b_codebook[:, ii] - codebook)**2).sum(1)
        a[ii] = 1000
        m = torch.min(a)
        if m < dist:
            dist = m
    return dist

def get_pairwiseD(code, size = None):
    if size is None:
        all_msg_bits = []

        for i in range(2**code.K):
            d = dec2bitarray(i, code.K)
            all_msg_bits.append(d)
        all_message_bits = torch.from_numpy(np.array(all_msg_bits))
        all_message_bits = 1 - 2*all_message_bits.float()
    else:
        all_message_bits = 1 - 2 *(torch.rand(size, code.K) < 0.5).float()
    codebook = 0.5*code.encode(all_message_bits)+0.5
    b_codebook = codebook.unsqueeze(0)

    dist_counts = {}
    for ii in range(codebook.shape[0]):
        a = ((b_codebook[:, ii] - codebook)**2).sum(1)
        counts = Counter(a.int().numpy())
        for key in counts:
            if key not in dist_counts.keys():
                dist_counts[key] = counts[key]
            else:
                dist_counts[key] += counts[key]

    # minimum distance : np.sqrt(min(dist_counts.keys()))
    # average distance : np.array([np.sqrt(key)*value for (key, value) in dist_counts.items()]).sum()/np.array(list(dist_counts.values())).sum()
    return {key:value//2 for (key, value) in dist_counts.items() if key != 0}

def get_pairwiseD_weight(code, size = None):
    if size is None:
        all_msg_bits = []

        for i in range(2**code.K):
            d = dec2bitarray(i, code.K)
            all_msg_bits.append(d)
        all_message_bits = torch.from_numpy(np.array(all_msg_bits))
        all_message_bits = 1 - 2*all_message_bits.float()
    else:
        all_message_bits = 1 - 2 *(torch.rand(size, code.K) < 0.5).float()
    codebook = 1 - (0.5*code.encode(all_message_bits)+0.5)
    b_codebook = codebook.unsqueeze(0)

    a = codebook.sum(1)
    dist_counts = Counter(a.int().numpy())
    # minimum distance : np.sqrt(min(dist_counts.keys()))
    # average distance : np.array([np.sqrt(key)*value for (key, value) in dist_counts.items()]).sum()/np.array(list(dist_counts.values())).sum()
    return dist_counts


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

class STEQuantize(torch.autograd.Function):
    #self.args.fb_quantize_limit, self.args.fb_quantize_level
    @staticmethod
    def forward(ctx, inputs, quant_limit=1, quant_level=2):

        ctx.save_for_backward(inputs)

        x_lim_abs  = quant_limit
        x_lim_range = 2.0 * x_lim_abs
        x_input_norm =  torch.clamp(inputs, -x_lim_abs, x_lim_abs)

        if quant_level == 2:
            outputs_int = torch.sign(x_input_norm)
        else:
            outputs_int  = torch.round((x_input_norm +x_lim_abs) * ((quant_level - 1.0)/x_lim_range)) * x_lim_range/(quant_level - 1.0) - x_lim_abs

        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # let's see what happens....
        # grad_output[torch.abs(input)>1.5]=0
        # grad_output[torch.abs(input)<0.5]=0

        # grad_output[input>1.0]=0
        # grad_output[input<-1.0]=0

        grad_output = torch.clamp(grad_output, -0.25, +0.25)

        grad_input = grad_output.clone()

        return grad_input, None, None, None

class STESign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

class Clamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, min=0, max=1):
        return input.clamp(min=min+(1e-10), max=max-(1e-10))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def min_sum_log_sum_exp(x, y):

    log_sum_ms = torch.min(torch.abs(x), torch.abs(y))*torch.sign(x)*torch.sign(y)
    return log_sum_ms

def log_sum_exp_diff(x, y):

    c1 = torch.max(x+y,torch.zeros_like(x))
    c2 = torch.max(x, y)

    # log_sum_standard = torch.log(1 + (x+y).exp()) - x - torch.log(1 + (y-x).exp() )
    log_sum_standard = c1 + torch.log((-c1).exp() + (x+y-c1).exp()) - c2 - torch.log((x-c2).exp() + (y-c2).exp())

    # log_sum_standard = torch.min(torch.abs(x), torch.abs(y))*torch.sign(x)*torch.sign(y)
    return log_sum_standard

def log_sum_exp(LLR_vector):

    sum_vector = LLR_vector.sum(dim=1, keepdim=True)
    sum_concat = torch.cat([sum_vector, torch.zeros_like(sum_vector)], dim=1)

    return torch.logsumexp(sum_concat, dim=1)- torch.logsumexp(LLR_vector, dim=1)

def log_sum_avoid_NaN(x, y):

    a = torch.max(x, y)
    b = torch.min(x, y)

    log_sum_standard = torch.log(1 + (x+y).exp()) - x - torch.log(1 + (y-x).exp() )

    # print("Original one:", log_sum_standard)

    ## Check for NaN or infty or -infty once here.
    if (torch.isnan(log_sum_standard).sum() > 0) | ((log_sum_standard == float('-inf')).sum() > 0 )| ( (log_sum_standard == float('inf')).sum() > 0) :

        # print("Had to avoid NaNs!")
        # 80 for float32 and 707 for float64.
        #big_threshold = 80. if log_sum_standard.dtype == torch.float32 else 700.
        big_threshold = 200. if log_sum_standard.dtype == torch.float32 else 700.
        idx_1 = (x + y > big_threshold)
        subset_1 = idx_1 & ((x-y).abs() < big_threshold)

        idx_2 = (x + y < -big_threshold)
        subset_2 = idx_2 & ((x-y).abs() < big_threshold)

        idx_3 = ((x - y).abs() > big_threshold) & ( (x+y).abs() < big_threshold )

        # Can be fastened
        if idx_1.sum() > 0 :

            if subset_1.sum() > 0:
                log_sum_standard[subset_1] = y[subset_1]- torch.log(1 + (y[subset_1] - x[subset_1]).exp() )
                # print("After 11 modification", log_sum_standard)

            if (idx_1 ^ subset_1).sum() > 0:
                log_sum_standard[idx_1 ^ subset_1] = b[idx_1 ^ subset_1]
                # print("After 12 modification", log_sum_standard)

        if idx_2.sum() > 0:

            if subset_2.sum() > 0:
                log_sum_standard[subset_2] = -x[subset_2]- torch.log(1 + (y[subset_2] - x[subset_2]).exp() )
                # print("After 21 modification", log_sum_standard)

            if (idx_2 ^ subset_2).sum() > 0:
                log_sum_standard[idx_2 ^ subset_2] = -a[idx_2 ^ subset_2]
                # print("After 22 modification", log_sum_standard)

        if idx_3.sum() > 0:

            log_sum_standard[idx_3] = torch.log(1 + (x[idx_3]+ y[idx_3]).exp() ) - a[idx_3]
            # print("After 3 modification", log_sum_standard)

    return log_sum_standard


def log_sum_avoid_zero_NaN(x, y):

    avoided_NaN = log_sum_avoid_NaN(x,y)

    zero_idx = (avoided_NaN == 0.)

    data_type = x.dtype

    if zero_idx.sum() > 0:

        # print("Had to avoid zeros!")

        x_subzero = x[zero_idx]
        y_subzero = y[zero_idx]

        nume = torch.relu(x_subzero + y_subzero)
        denom = torch.max(x_subzero , y_subzero)
        delta = 1e-7 if data_type == torch.float32 else 1e-16

        term_1 = 0.5 *( (-nume).exp() + (x_subzero + y_subzero - nume).exp() )
        term_2 = 0.5 * ( (x_subzero - denom).exp() + (y_subzero - denom).exp() )

        # close_1 = torch.tensor( (term_1 - 1).abs() < delta, dtype= data_type)
        close_1 = ((term_1 - 1).abs() < delta).clone().float()
        T_1 =  (term_1 - 1.) * close_1 + torch.log(term_1) * (1-close_1)

        # close_2 = torch.tensor( (term_2 - 1).abs() < delta, dtype= data_type)
        close_2 = ((term_2 - 1).abs() < delta).clone().float()
        T_2 =  (term_2 - 1.) * close_2 + torch.log(term_2) * (1-close_2)

        corrected_ans = nume - denom + T_1 - T_2

        further_zero = (corrected_ans == 0.)

        if further_zero.sum() > 0:

                x_sub_subzero = x_subzero[further_zero]
                y_sub_subzero = y_subzero[further_zero]

                positive_idx = ( x_sub_subzero + y_sub_subzero > 0.)

                spoiled_brat = torch.min(- x_sub_subzero, - y_sub_subzero)

                spoiled_brat[positive_idx] = torch.min(x_sub_subzero[positive_idx], y_sub_subzero[positive_idx])

                corrected_ans[further_zero] = spoiled_brat

        avoided_NaN[zero_idx] = corrected_ans

    return avoided_NaN


def new_log_sum(x, y):
    # log_sum_standard = torch.nan_to_num(torch.nan_to_num(torch.log(torch.abs((1 - (x+y).exp()))) - torch.log(torch.abs(((y-x).exp() - 1) ))) - x)
    # log_sum_standard = torch.log(torch.abs((1 - (x+y).exp())/((y-x).exp() - 1) )) - x
    # log_sum_standard = torch.nan_to_num(torch.nan_to_num(torch.log(torch.abs(((x-y).exp() - (2*x).exp()))) - torch.log(torch.abs(((x-y).exp() - 1) ))) - x)
    # log_sum_standard = torch.log(torch.abs(((x-y).exp() - (2*x).exp()))) - torch.log(torch.abs(((x-y).exp() - 1) )) - x

    x = x.clone() + torch.isclose(x,y).float()* 1e-5 #otherwise we get inf
    c1 = torch.max(x+y,torch.zeros_like(x))
    c2 = torch.max(x, y)
    log_sum_standard = torch.log(torch.abs((-1*c1).exp() - (x+y-c1).exp())) + c1 - torch.log(torch.abs((x-c2).exp() - (y-c2).exp())) - c2

    if torch.isnan(log_sum_standard).any():
        print(c1, c2, (-1*c1).exp(), (x+y-c1).exp(), (x-c2).exp(), (y-c2).exp())
        id1 = np.random.randint(0,100)

        torch.save([x, y], 'errors_' + str(id1)+'.pt')
        print('Nan detected! Saved at {}'.format('errors_' + str(id1)+'.pt'))

    if torch.isinf(log_sum_standard).any():
        print(c1, c2, (-1*c1).exp(), (x+y-c1).exp(), (x-c2).exp(), (y-c2).exp())
        id1 = np.random.randint(0,100)

        torch.save([x, y], 'errors_inf_' + str(id1)+'.pt')
        print('Inf detected! Saved at {}'.format('errors_inf_' + str(id1)+'.pt'))

    return log_sum_standard

def new_log_sum_avoid_NaN(x, y):

    a = torch.max(x, y)
    b = torch.min(x, y)

    x = x.clone() + torch.isclose(x,y).float()* 1e-5 #otherwise we get inf

    # log_sum_standard = torch.log(torch.abs((1 - (x+y).exp())/(y.exp() - x.exp())) )
    log_sum_standard = torch.log(torch.abs((1 - (x+y).exp()))) - x - torch.log(torch.abs(((y-x).exp() - 1) ))

    # log_sum_standard = torch.log(torch.abs((-1*x).exp() - (-1*y).exp()) )

    # print("Original one:", log_sum_standard)

    ## Check for NaN or infty or -infty once here.
    if (torch.isnan(log_sum_standard).sum() > 0) | ((log_sum_standard == float('-inf')).sum() > 0 )| ( (log_sum_standard == float('inf')).sum() > 0) :

        # print("Had to avoid NaNs!")
        # 80 for float32 and 707 for float64.
        #big_threshold = 80. if log_sum_standard.dtype == torch.float32 else 700.
        big_threshold = 200. if log_sum_standard.dtype == torch.float32 else 700.
        idx_1 = (x + y > big_threshold)
        subset_1 = idx_1 & ((x-y).abs() < big_threshold)

        idx_2 = (x + y < -big_threshold)
        subset_2 = idx_2 & ((x-y).abs() < big_threshold)

        idx_3 = ((x - y).abs() > big_threshold) & ( (x+y).abs() < big_threshold )

        # Can be fastened
        if idx_1.sum() > 0 :

            if subset_1.sum() > 0:
                log_sum_standard[subset_1] = y[subset_1]- torch.log(1 + (y[subset_1] - x[subset_1]).exp() )
                # print("After 11 modification", log_sum_standard)

            if (idx_1 ^ subset_1).sum() > 0:
                log_sum_standard[idx_1 ^ subset_1] = b[idx_1 ^ subset_1]
                # print("After 12 modification", log_sum_standard)

        if idx_2.sum() > 0:

            if subset_2.sum() > 0:
                log_sum_standard[subset_2] = -x[subset_2]- torch.log(1 + (y[subset_2] - x[subset_2]).exp() )
                # print("After 21 modification", log_sum_standard)

            if (idx_2 ^ subset_2).sum() > 0:
                log_sum_standard[idx_2 ^ subset_2] = -a[idx_2 ^ subset_2]
                # print("After 22 modification", log_sum_standard)

        if idx_3.sum() > 0:

            log_sum_standard[idx_3] = torch.log(1 + (x[idx_3]+ y[idx_3]).exp() ) - a[idx_3]
            # print("After 3 modification", log_sum_standard)

    return log_sum_standard


def new_log_sum_avoid_zero_NaN(x, y):

    avoided_NaN = new_log_sum_avoid_NaN(x,y)

    zero_idx = (avoided_NaN == 0.)

    data_type = x.dtype

    if zero_idx.sum() > 0:

        # print("Had to avoid zeros!")

        x_subzero = x[zero_idx]
        y_subzero = y[zero_idx]

        nume = torch.relu(x_subzero + y_subzero)
        denom = torch.max(x_subzero , y_subzero)
        delta = 1e-7 if data_type == torch.float32 else 1e-16

        term_1 = 0.5 *( (-nume).exp() + (x_subzero + y_subzero - nume).exp() )
        term_2 = 0.5 * ( (x_subzero - denom).exp() + (y_subzero - denom).exp() )

        # close_1 = torch.tensor( (term_1 - 1).abs() < delta, dtype= data_type)
        close_1 = ((term_1 - 1).abs() < delta).clone().float()
        T_1 =  (term_1 - 1.) * close_1 + torch.log(term_1) * (1-close_1)

        # close_2 = torch.tensor( (term_2 - 1).abs() < delta, dtype= data_type)
        close_2 = ((term_2 - 1).abs() < delta).clone().float()
        T_2 =  (term_2 - 1.) * close_2 + torch.log(term_2) * (1-close_2)

        corrected_ans = nume - denom + T_1 - T_2

        further_zero = (corrected_ans == 0.)

        if further_zero.sum() > 0:

                x_sub_subzero = x_subzero[further_zero]
                y_sub_subzero = y_subzero[further_zero]

                positive_idx = ( x_sub_subzero + y_sub_subzero > 0.)

                spoiled_brat = torch.min(- x_sub_subzero, - y_sub_subzero)

                spoiled_brat[positive_idx] = torch.min(x_sub_subzero[positive_idx], y_sub_subzero[positive_idx])

                corrected_ans[further_zero] = spoiled_brat

        avoided_NaN[zero_idx] = corrected_ans

    return avoided_NaN
