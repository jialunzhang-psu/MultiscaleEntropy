import numpy as np
import math
from collections import Iterable

def _init_return_type(return_type):
    if return_type == 'dict':
        return {}
    else:
        return []

def _check_type(x, num_type, name):
    if isinstance(x, num_type):
        tmp = [x]
    elif not isinstance(x, Iterable):
        raise ValueError(name + ' should be a ' + num_type.__name__ + ' or an iterator of ' + num_type.__name__)
    else:
        tmp = []
        for i in x:
            tmp.append(i)
            if not isinstance(i, num_type):
                raise ValueError(name + ' should be a ' + num_type.__name__ + ' or an iterator of ' + num_type.__name__)
    return tmp

# sum of seperate intervals of x
def _coarse_grain(x, scale_factor):
    x = np.array(x)
    x_len = len(x)
    if x_len % scale_factor:
        padded_len = (1+int(x_len/scale_factor))*scale_factor
    else:
        padded_len = x_len
    tmp_x = np.zeros(padded_len)
    tmp_x[:x_len] = x
    tmp_x = np.reshape(tmp_x, (int(padded_len/scale_factor), scale_factor))
    ans = np.reshape(np.sum(tmp_x, axis=1), (-1))/scale_factor

    return ans

def sample_entropy(x, m=[2], r=[0.15], sd=None, return_type='dict', safe_mode=False):      
    '''[Sample Entropy, the threshold will be r*sd]
    
    Arguments:
        x {[input signal]} -- [an iterator of numbers]
    
    Keyword Arguments:
        m {list} -- [m in sample entropy] (default: {[2]})
        r {list} -- [r in sample entropy] (default: {[0.15]})
        sd {number} -- [standard derivation of x, if None, will be calculated] (default: {None})
        return_type {str} -- [can be dict or list] (default: {'dict'})
        safe_mode {bool} -- [if set True, type checking will be skipped] (default: {False})
    
    Raises:
        ValueError -- [some values too big]
    
    Returns:
        [dict or list as return_type indicates] -- [if dict, nest as [scale_factor][m][r] for each value of m, r; if list, nest as [i][j] for lengths of m, r]
    '''
    # type checking
    if not safe_mode:
        m = _check_type(m, int, 'm')
        r = _check_type(r, float, 'r')
        if not (sd == None) and not (isinstance(sd, float) or isinstance(sd, int)):
            raise ValueError('sd should be a number')
    try:
        x = np.array(x)
    except:
        raise ValueError('x should be a sequence of numbers')
    # value checking
    if len(x) < max(m):
        raise ValueError('the max m is bigger than x\'s length')
    
    # initialization
    if sd == None:
        sd = np.sqrt(np.var(x))
    ans = _init_return_type(return_type)

    # calculation
    for i, rr in enumerate(r):
        threshold = rr * sd
        if return_type == 'dict':
            ans[rr] = _init_return_type(return_type)
        else:
            ans.append(_init_return_type(return_type))
        count = {}
        tmp_m = []
        for mm in m:
            tmp_m.append(mm)
            tmp_m.append(mm+1)
        tmp_m = list(set(tmp_m))
        for mm in tmp_m:
            count[mm] = 0

        for j in range(1, len(x)-min(m)+1):
            cont = 0
            for inc in range(0, len(x)-j):
                if abs(x[inc]-x[j+inc]) < threshold:
                    cont += 1
                elif cont > 0:
                    for mm in tmp_m:
                        tmp = cont - mm + 1
                        count[mm] += tmp if tmp > 0 else 0
                    cont = 0
            if cont > 0:
                for mm in tmp_m:
                    tmp = cont - mm + 1
                    count[mm] += tmp if tmp > 0 else 0
        for mm in m:
            if count[mm+1] == 0 or count[mm] == 0:
                t = len(x)-mm+1
                tmp = -math.log(1/(t*(t-1)))
            else:
                tmp = -math.log(count[mm+1]/count[mm])
            if return_type == 'dict': 
                ans[rr][mm] = tmp
            else:
                ans[i].append(tmp)
    return ans

def mse(x, scale_factor=[i for i in range(1,21)], m=[2], r=[0.15], return_type='dict', safe_mode=False):
    '''[Multiscale Entropy]
    
    Arguments:
        x {[input signal]} -- [an iterator of numbers]
    
    Keyword Arguments:
        scale_factor {list} -- [scale factors of coarse graining] (default: {[i for i in range(1,21)]})
        m {list} -- [m in sample entropy] (default: {[2]})
        r {list} -- [r in sample entropy] (default: {[0.15]})
        return_type {str} -- [can be dict or list] (default: {'dict'})
        safe_mode {bool} -- [if set True, type checking will be skipped] (default: {False})
    
    Raises:
        ValueError -- [some values too big]
    
    Returns:
        [dict or list as return_type indicates] -- [if dict, nest as [scale_factor][m][r] for each value of scale_factor, m, r; if list nest as [i][j][k] for lengths of scale_factor, m, r]
    '''
    # type checking
    if not safe_mode:
        m = _check_type(m, int, 'm')
        r = _check_type(r, float, 'r')
        scale_factor = _check_type(scale_factor, int, 'scale_factor')
    try:
        x = np.array(x)
    except:
        print('x should be a sequence of numbers')
    # value checking
    if max(scale_factor) > len(x):
        raise ValueError('the max scale_factor is bigger than x\'s length')

    # calculation
    sd = np.sqrt(np.var(x))
    ms_en = _init_return_type(return_type)
    for s_f in scale_factor:
        y = _coarse_grain(x, s_f)
        if return_type == 'dict':
            ms_en[s_f] = sample_entropy(y, m, r, sd, 'dict', True)
        else:
            ms_en.append(sample_entropy(y, m, r, sd, 'list', True))

    return ms_en