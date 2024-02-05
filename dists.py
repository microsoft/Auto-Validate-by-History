
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import chebyshev
from scipy.spatial.distance import cosine
from scipy.stats import chi2_contingency
from numpy.lib.arraysetops import unique
from scipy.stats import fisher_exact
import global_efficiency as gloeff
from scipy.stats import ks_2samp
from scipy.stats import entropy
from scipy.stats import moment
from scipy.stats import skew
from divergence import *
import pandas as pd
import numpy as np
import string
import scipy
import time


def compute_probs(data, n=10):
    # data = np.array(data, dtype=float)
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p


def support_intersection(p, q): 
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int


def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q


def kl_divergence(p, q): 
    return np.sum(p*np.log(p/q))


def js_divergence(p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*kl_divergence(p, m) + (1./2.)*kl_divergence(q, m)


def compute_kl_divergence(train_sample, test_sample, n_bins=10): 
    """
    Computes the KL Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    
    return kl_divergence(p, q)


def compute_js_divergence(train_sample, test_sample, n_bins=10): 
    """
    Computes the JS Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)
    
    list_of_tuples = support_intersection(p,q)
    p, q = get_probs(list_of_tuples)
    
    return js_divergence(p, q)


def cohen_d(x, y):
    """
    Cohen's d distance
    """

    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    mu1 = np.mean(x)
    mu2 = np.mean(y)
    var1 = np.var(x, ddof=1)
    var2 = np.var(y, ddof=1)
    
    sp = np.sqrt(((nx-1) * var1 + (ny-1) * var2) / dof)

    return (mu1-mu2) / (sp +0.0000000001)


def parse_pattern(value):
    # character, digit, punc
    value = list(str(value))
    pat_value = []
    for i, val in enumerate(value):
        if i==0:
            if val.isdigit():
                pat_value.append('\d')
            elif val.isalpha():
                pat_value.append('\l')
            else:
                pat_value.append('-')
        else:
            if val.isdigit() and not value[i-1].isdigit():
                pat_value.append('\d')
            elif val.isalpha() and not value[i-1].isalpha():
                pat_value.append('\l')
            elif val in string.punctuation and value[i-1] not in string.punctuation:
                pat_value.append('-')

    pat_value = ''.join(pat_value)

    return pat_value


def comp_sample_pattern(sample):
    pat_sample = sample.apply(parse_pattern)
    return pat_sample

def comp_string_len(value):

    if not pd.isna(value):
        return len(value)
    else:
        return 0

def comp_char_len(value):
    char_len = 0
    if not pd.isna(value):
        for i in value:
            if i.isalpha():
                char_len += 1
        return char_len
    else:
        return 0

def comp_digit_len(value):
    digit_len = 0
    if not pd.isna(value):
        for i in value:
            if i.isdigit():
                digit_len += 1
        return digit_len
    else:
        return 0

def comp_punc_len(value):
    punc_len = 0
    if not pd.isna(value):
        for i in value:
            if not i.isdigit() and not i.isalpha():
                punc_len += 1
        return punc_len
    else:
        return 0


def comp_dist(sample_p, sample_q, dtype='numeric'):
    """
    compute distance between tow columns
    """

    # start = time.time()
    row_count_p = len(sample_p)   # row number

    sample_p = sample_p.dropna()
    sample_q = sample_q.dropna()

    if dtype=='numeric':

        # single dist
        min_val = sample_p.min()
        max_val = sample_p.max()
        mean_val = sample_p.mean()
        median_val = sample_p.median()
        sum_val = sample_p.sum()
        range_val = max_val - min_val
        skewness = skew(sample_p)             # check ok
        moment2 = moment(sample_p, moment=2)
        moment3 = moment(sample_p, moment=3)  # check ok

        unique_ratio = len(pd.unique(sample_p)) / len(sample_p)
        complete_ratio = len(sample_p.dropna()) / row_count_p     # update

        # single_end = time.time()
        # gloeff.set_value('single_time', gloeff.get_value('single_time') + single_end - start)

        # two dist
        emd = wasserstein_distance(sample_p, sample_q)        # EMD (Wasserstein Distance)
        js_div = compute_js_divergence(sample_p, sample_q)
        kl_div = compute_kl_divergence(sample_p, sample_q)
        _, ks_p_val = ks_2samp(sample_p, sample_q)            # KS distance
        cohen_dist = cohen_d(sample_p, sample_q)              # Cohen's d

        # two_end = time.time()
        # gloeff.set_value('two_time', gloeff.get_value('two_time') + two_end - single_end)


        # kl_div = relative_entropy_from_samples(c1, c2, discrete=False)           # KL Divergence from divergence lib, fail
        # js_div = jensen_shannon_divergence_from_samples(c1, c2, discrete=False)  # JS Divergence from divergence lib, fail   integration too slow

        dist_list = [emd, js_div, kl_div,  ks_p_val, cohen_dist,
                     min_val, max_val, mean_val, median_val, row_count_p, 
                     sum_val, range_val, skewness, moment2, moment3, unique_ratio, complete_ratio]
        
    else:
        sample_p = sample_p.astype(str)
        sample_q = sample_q.astype(str)

        # single dist   norm 
        str_len = int(np.mean(sample_p.apply(comp_string_len)))
        char_len = int(np.mean(sample_p.apply(comp_char_len)))
        digit_len = int(np.mean(sample_p.apply(comp_digit_len)))
        punc_len = int(np.mean(sample_p.apply(comp_punc_len)))
        unique_ratio = len(pd.unique(sample_p)) / len(sample_p)
        complete_ratio = len(sample_p.dropna()) / row_count_p     # update

        # Chebyshev 
        dist_val_count = len(pd.unique(sample_p))

        # single_end = time.time()
        # gloeff.set_value('single_time', gloeff.get_value('single_time') + single_end - start)


        # convert to PMF from two samples
        combined_frequencies_p, combined_frequencies_q, _, _, _ = \
            convert_freq_for_two_samples(sample_p, sample_q)

        l_1 = minkowski(combined_frequencies_p, combined_frequencies_q, 1)     # L-1
        l_inf = chebyshev(combined_frequencies_p, combined_frequencies_q)      # L-inf
        cos = cosine(combined_frequencies_p, combined_frequencies_q)           # cosine
        _, p_val = comp_cat_chisq(sample_p, sample_q)                          # Chisquare for two sample
        js_div = jensenshannon(combined_frequencies_p, combined_frequencies_q) # JS divergence
        kl_div = comp_cat_kl_div(sample_p, sample_q)                           # KL divergence for two samples

        # pattern
        pat_sample_p = comp_sample_pattern(sample_p)
        pat_sample_q = comp_sample_pattern(sample_q)

        # convert to PMF from two samples
        pat_combined_frequencies_p, pat_combined_frequencies_q, _, _, _ = \
            convert_freq_for_two_samples(pat_sample_p, pat_sample_q)

        pat_l_1 = minkowski(pat_combined_frequencies_p, pat_combined_frequencies_q, 1)     # pattern L-1
        pat_l_inf = chebyshev(pat_combined_frequencies_p, pat_combined_frequencies_q)      # pattern L-inf
        pat_cos = cosine(pat_combined_frequencies_p, pat_combined_frequencies_q)           # pattern cosine
        _, pat_p_val = comp_cat_chisq(pat_sample_p, pat_sample_q)                          # pattern Chisquare for two sample
        pat_js_div = jensenshannon(pat_combined_frequencies_p, pat_combined_frequencies_q) # pattern JS divergence
        pat_kl_div = comp_cat_kl_div(pat_sample_p, pat_sample_q)                           # pattern KL divergence for two samples

        # two_end = time.time()
        # gloeff.set_value('two_time', gloeff.get_value('two_time') + two_end - single_end)

        dist_list = [l_1, l_inf, cos, p_val, row_count_p, js_div, kl_div, 
                     str_len, char_len, digit_len, punc_len, unique_ratio, complete_ratio,
                     dist_val_count, 
                     pat_l_1, pat_l_inf, pat_cos, pat_p_val, pat_js_div, pat_kl_div]
    
    return dist_list


def comp_cat_js_div(sample_p, sample_q):

    combined_frequencies_p, combined_frequencies_q, _, _ = \
        convert_freq_for_two_samples(sample_p, sample_q)

    return jensenshannon(combined_frequencies_p, combined_frequencies_q)


def comp_cat_kl_div(sample_p, sample_q):
    """ cpmpute categorical JS Divergence """

    combined_sample = np.hstack((sample_p, sample_q))
    unique_combined = np.unique(combined_sample)

    unique_p, counts_p = np.unique(sample_p, return_counts=True)
    unique_q, counts_q = np.unique(sample_q, return_counts=True)

    combined_counts_p = np.ones((len(unique_combined), ))
    combined_counts_q = np.ones((len(unique_combined), ))

    for i, sample in enumerate(unique_combined):
        if sample in unique_p:
            idx = int(np.argwhere(unique_p==sample))
            combined_counts_p[i] = counts_p[idx] + 1

        if sample in unique_q:
            idx = int(np.argwhere(unique_q==sample))
            combined_counts_q[i] = counts_q[idx] + 1

    combined_frequencies_p = combined_counts_p / (len(sample_p) + len(unique_combined))
    combined_frequencies_q = combined_counts_q / (len(sample_q) + len(unique_combined))

    return entropy(combined_frequencies_p, combined_frequencies_q)


def comp_cat_chisq(sample_p, sample_q):
    """ cpmpute categorical JS Divergence """

    combined_sample = np.hstack((sample_p, sample_q))
    unique_combined = np.unique(combined_sample)

    unique_p, counts_p = np.unique(sample_p, return_counts=True)
    unique_q, counts_q = np.unique(sample_q, return_counts=True)

    combined_counts_p = np.ones((len(unique_combined), ))
    combined_counts_q = np.ones((len(unique_combined), ))

    for i, sample in enumerate(unique_combined):
        if sample in unique_p:
            idx = int(np.argwhere(unique_p==sample))
            combined_counts_p[i] = counts_p[idx] + 1

        if sample in unique_q:
            idx = int(np.argwhere(unique_q==sample))
            combined_counts_q[i] = counts_q[idx] + 1

    #  Null hypothesis: the two groups have no significant difference.
    obs = np.vstack((combined_counts_p, combined_counts_q))
    chi2, chi2_p_val, _, _ = chi2_contingency(obs)

    return chi2, chi2_p_val


def convert_freq_for_two_samples(sample_p, sample_q):

    combined_sample = np.hstack((sample_p, sample_q))
    unique_combined = np.unique(combined_sample)

    unique_p, counts_p = np.unique(sample_p, return_counts=True)
    unique_q, counts_q = np.unique(sample_q, return_counts=True)

    combined_counts_p = np.zeros((len(unique_combined), ))
    combined_counts_q = np.zeros((len(unique_combined), ))

    for i, sample in enumerate(unique_combined):
        if sample in unique_p:
            idx = int(np.argwhere(unique_p==sample))
            combined_counts_p[i] = counts_p[idx]

        if sample in unique_q:
            idx = int(np.argwhere(unique_q==sample))
            combined_counts_q[i] = counts_q[idx]

    combined_frequencies_p = combined_counts_p / len(sample_p)
    combined_frequencies_q = combined_counts_q / len(sample_q)

    return combined_frequencies_p, combined_frequencies_q, combined_counts_p, combined_counts_q, unique_combined


def convert_freq_for_one_samples(sample_p):

    unique_p, counts_p = np.unique(sample_p, return_counts=True)

    frequencies_p = counts_p / len(sample_p)

    return frequencies_p, counts_p, unique_p



def comp_cat_l_inf(sample_p, sample_q):
    """compute categorical L-infinity"""
    # sample_p = np.array([str(x) for x in sample_p])
    # sample_q = np.array([str(x) for x in sample_q])

    combined_frequencies_p, combined_frequencies_q, _, _, _= \
        convert_freq_for_two_samples(sample_p, sample_q)

    return chebyshev(combined_frequencies_p, combined_frequencies_q)


def comp_cat_cos(sample_p, sample_q):
    """compute categorical Cosine"""

    combined_frequencies_p, combined_frequencies_q, _, _, _ = \
        convert_freq_for_two_samples(sample_p, sample_q)

    return cosine(combined_frequencies_p, combined_frequencies_q)


if __name__ == '__main__':
    pass
    
# Unit Test
# Case 1  same ditribution
#     print('----------Same distribution 1----------')
#     sample_p = np.array(['1', '1', '1', '2', 'aaa', 'a234ddddddd', '0.5', 'uuuu'])
#     sample_q = np.array([ 'uuuu', '2', '1', '0.5', '1', 'aaa','a234ddddddd', '1'])

#     expected_frequencies = np.array([0.125, 0.375, 0.125, 0.125, 0.125, 0.125])
#     expected_counts = np.array([1, 3, 1, 1, 1, 1])

#     combined_frequencies_p, combined_frequencies_q, combined_counts_p, combined_counts_q, _ = \
#         convert_freq_for_two_samples(sample_p, sample_q)

#     assert (combined_frequencies_p==expected_frequencies).all()
#     assert (combined_frequencies_q==expected_frequencies).all()
#     assert (combined_counts_p==expected_counts).all()
#     assert (combined_counts_q==expected_counts).all()
    
#     assert np.isclose(comp_cat_kl_div(sample_p, sample_q), 0)  # KL Divergence  for two samples
#     assert np.isclose(comp_cat_cos(sample_p, sample_q), 0)     # Cosine Similarity for two samples
#     assert np.isclose(comp_cat_l_inf(sample_p, sample_q), 0)   # L-infinity  for two samples

#     assert np.isclose(jensenshannon(combined_frequencies_p, combined_frequencies_q), 0) # JS Divergence for two PMFs
#     assert np.isclose(minkowski(combined_frequencies_p, combined_frequencies_q, 1) , 0) # L1 for two PMFs
#     assert np.isclose(chebyshev(combined_frequencies_p, combined_frequencies_q), 0) # l_inf for two PMFs
#     assert np.isclose(cosine(combined_frequencies_p, combined_frequencies_q), 0) # Cosine for two PMFs
#     chisq, p_val = chisquare(combined_counts_p, combined_counts_q)  # Chisquare for two PMFs
#     assert np.isclose(chisq, 0)
#     assert np.isclose(p_val, 1)

#     print("Category L-1: ", minkowski(combined_frequencies_p, combined_frequencies_q, 1))
#     print("Category KL-div: ", comp_cat_kl_div(sample_p, sample_q))
#     print("Category L-inf: ", comp_cat_l_inf(sample_p, sample_q))    # L-infinity for two samples
#     print("Category Chisq-p-val: ", p_val) 
#     print("Category JS-div: ", jensenshannon(combined_frequencies_p, combined_frequencies_q))
#     print("Category Cosine: ", cosine(combined_frequencies_p, combined_frequencies_q))

#     # Case 2  same ditribution
#     print('----------Same distribution 2 Repeat----------')
    
#     sample_p = np.array(['uswest(BA,BY)', '8,9', 'uswest(BA,BY)', 'BD3.Boydton3GSGO;BN3.Boydton3GSGO', 
#                         'SN5.SN5S', 'DefaultCustomer', 'DefaultCustomer', 'aeed5fdc-b28a-4bb9-847c-0a7bc6867c50', 
#                         'aeed5fdc-b28a-4bb9-847c-0a7bc6867c50', 'DefaultCustomer'])
#     repeat_time = 2
#     sample_q = sample_p.repeat(repeat_time).copy()
#     np.random.shuffle(sample_q)
    
#     expected_frequencies_p = np.array([0.1, 0.1, 0.3, 0.1, 0.2, 0.2])
#     expected_frequencies_q = np.array([0.1, 0.1, 0.3, 0.1, 0.2, 0.2])

#     expected_counts_p = np.array([1, 1, 3, 1, 2, 2])
#     expected_counts_q = np.array([1, 1, 3, 1, 2, 2]) * repeat_time

#     combined_frequencies_p, combined_frequencies_q, combined_counts_p, combined_counts_q, _ = \
#         convert_freq_for_two_samples(sample_p, sample_q)

#     assert (combined_frequencies_p==expected_frequencies_p).all()
#     assert (combined_frequencies_q==expected_frequencies_q).all()
#     assert (combined_counts_p==expected_counts_p).all()
#     assert (combined_counts_q==expected_counts_q).all()
    
#     assert comp_cat_kl_div(sample_p, sample_q) < 0.014        # Smooth KL Divergence  for two samples
#     assert np.isclose(comp_cat_cos(sample_p, sample_q), 0)    # Cosine Similarity for two samples
#     assert np.isclose(comp_cat_l_inf(sample_p, sample_q), 0)   # L-infinity  for two samples

#     assert np.isclose(jensenshannon(combined_frequencies_p, combined_frequencies_q), 0) # JS Divergence for two PMFs
#     assert np.isclose(minkowski(combined_frequencies_p, combined_frequencies_q, 1) , 0) # L1 for two PMFs
#     assert np.isclose(chebyshev(combined_frequencies_p, combined_frequencies_q), 0) # l_inf for two PMFs
#     assert np.isclose(cosine(combined_frequencies_p, combined_frequencies_q), 0) # Cosine for two PMFs

#     # Null hypothesis: the two groups have no significant difference.
#     # although two cols have the same distribution, their counts affect p-value of chisquare
#     # chisq, p_val = chisquare(combined_frequencies_p, combined_frequencies_q)  # Chisquare for two PMFs
#     chisq, p_val = chisquare(combined_counts_p, combined_counts_q)  # p-value=5% usually indicate that a difference is significant

    
#     print("Category L-1: ", minkowski(combined_frequencies_p, combined_frequencies_q, 1))
#     print("Category KL-div: ", comp_cat_kl_div(sample_p, sample_q))
#     print("Category L-inf: ", comp_cat_l_inf(sample_p, sample_q))    # L-infinity for two samples
#     print("Category Chisq-p-val: ", p_val) 
#     print("Category JS-div: ", jensenshannon(combined_frequencies_p, combined_frequencies_q))
#     print("Category Cosine: ", cosine(combined_frequencies_p, combined_frequencies_q))

#    # Case 3 Completely different
#     print('-----------Completely different----------')
#     sample_p = np.array(['uswest(BA,BY)', '8,9', 'uswest(BA,BY)', 'BD3.Boydton3GSGO;BN3.Boydton3GSGO', 
#                         'SN5.SN5S', 'DefaultCustomer', 'DefaultCustomer', 'aeed5fdc-b28a-4bb9-847c-0a7bc6867c50', 
#                         'aeed5fdc-b28a-4bb9-847c-0a7bc6867c50', 'DefaultCustomer'])
#     repeat_time_p = 1
#     sample_p = sample_p.repeat(repeat_time_p)
#     np.random.shuffle(sample_p)

#     sample_q = np.array(['aff384ff-9720-42cd-95ad-836ef8652a65', '2018-09-09', 'NONDAP', 'NC', 
#                         'Organic', 'All', 'All', 'All'])
#     repeat_time_q = 1
#     sample_q = sample_q.repeat(repeat_time_q)
#     np.random.shuffle(sample_q)

#     expected_frequencies_p = np.array([0, 0.1, 0, 0.1, 0.3, 0, 0, 0, 0.1, 0.2, 0, 0.2])
#     expected_frequencies_q = np.array([0.125, 0, 0.375, 0, 0, 0.125, 0.125, 0.125, 0, 0, 0.125, 0])

#     expected_counts_p = np.array([0, 1, 0, 1, 3, 0, 0, 0, 1, 2, 0, 2]) * repeat_time_p
#     expected_counts_q = np.array([1, 0, 3, 0, 0, 1, 1, 1, 0, 0, 1, 0]) * repeat_time_q


#     combined_frequencies_p, combined_frequencies_q, combined_counts_p, combined_counts_q, _ = \
#         convert_freq_for_two_samples(sample_p, sample_q)

#     assert (combined_frequencies_p==expected_frequencies_p).all()
#     assert (combined_frequencies_q==expected_frequencies_q).all()
#     assert (combined_counts_p==expected_counts_p).all()
#     assert (combined_counts_q==expected_counts_q).all()
    
#     assert comp_cat_kl_div(sample_p, sample_q) > 0.4  # KL Divergence  for two samples
#     assert np.isclose(comp_cat_cos(sample_p, sample_q), 1)     # Cosine Similarity for two samples
#     chisq, p_val = comp_cat_chisq(sample_p, sample_q)  # Chisquare for two samples
#     assert p_val < 0.05 

#     assert jensenshannon(combined_frequencies_p, combined_frequencies_q) > 0.5# JS Divergence for two PMFs
#     assert np.isclose(minkowski(combined_frequencies_p, combined_frequencies_q, 1) , 2) # L1 for two PMFs
#     assert np.isclose(chebyshev(combined_frequencies_p, combined_frequencies_q), 0.375) # l_inf for two PMFs
#     assert np.isclose(cosine(combined_frequencies_p, combined_frequencies_q), 1) # Cosine for two PMFs

#     print("Category L-1: ", minkowski(combined_frequencies_p, combined_frequencies_q, 1))
#     print("Category KL-div: ", comp_cat_kl_div(sample_p, sample_q))
#     print("Category L-inf: ", comp_cat_l_inf(sample_p, sample_q))    # L-infinity for two samples
#     print("Category Chisq-p-val: ", p_val) 
#     print("Category JS-div: ", jensenshannon(combined_frequencies_p, combined_frequencies_q))
#     print("Category Cosine: ", cosine(combined_frequencies_p, combined_frequencies_q))


#     # Case 4 Slightly different
#     print('----------Slightly different----------')
#     sample_p = np.array(['uswest(BA,BY)', '8,9', 'uswest(BA,BY)', 'BD3.Boydton3GSGO;BN3.Boydton3GSGO', 
#                         'SN5.SN5S', 'DefaultCustomer', 'DefaultCustomer', 'aeed5fdc-b28a-4bb9-847c-0a7bc6867c50', 
#                         'aeed5fdc-b28a-4bb9-847c-0a7bc6867c50', 'DefaultCustomer'])
#     repeat_time_p = 1
#     sample_p = sample_p.repeat(repeat_time_p)
#     np.random.shuffle(sample_p)

#     sample_q = np.array(['uswest(BA,BY)', '8,9', 'uswest(BA,BY)', 'BD3.Boydton3GSGO;BN3.Boydton3GSGO', 
#                         'SN5.SN5S', 'DefaultCustomer', 'All', 'aeed5fdc-b28a-4bb9-847c-0a7bc6867c50', 
#                         'All', 'DefaultCustomer'])

#     repeat_time_q = 100
#     sample_q = sample_q.repeat(repeat_time_q)
#     np.random.shuffle(sample_q)

#     expected_frequencies_p = np.array([0.1, 0.0, 0.1, 0.3, 0.1, 0.2, 0.2])
#     expected_frequencies_q = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.2])

#     expected_counts_p = np.array([1, 0, 1, 3, 1, 2, 2]) * repeat_time_p
#     expected_counts_q = np.array([1, 2, 1, 2, 1, 1, 2]) * repeat_time_q


#     combined_frequencies_p, combined_frequencies_q, combined_counts_p, combined_counts_q, _ = \
#         convert_freq_for_two_samples(sample_p, sample_q)

#     assert (combined_frequencies_p==expected_frequencies_p).all()
#     assert (combined_frequencies_q==expected_frequencies_q).all()
#     assert (combined_counts_p==expected_counts_p).all()
#     assert (combined_counts_q==expected_counts_q).all()
    
#     assert comp_cat_kl_div(sample_p, sample_q) > 0.01  # KL Divergence  for two samples
#     chisq, p_val = comp_cat_chisq(sample_p, sample_q)  # Chisquare for two samples
#     assert p_val < 0.05 

#     # assert jensenshannon(combined_frequencies_p, combined_frequencies_q) > 0.5# JS Divergence for two PMFs
#     # assert np.isclose(minkowski(combined_frequencies_p, combined_frequencies_q, 1) , 2) # L1 for two PMFs
#     # assert np.isclose(chebyshev(combined_frequencies_p, combined_frequencies_q), 0.375) # l_inf for two PMFs
#     # assert np.isclose(cosine(combined_frequencies_p, combined_frequencies_q), 1) # Cosine for two PMFs

#     print("Category L-1: ", minkowski(combined_frequencies_p, combined_frequencies_q, 1))
#     print("Category KL-div: ", comp_cat_kl_div(sample_p, sample_q))
#     print("Category L-inf: ", comp_cat_l_inf(sample_p, sample_q))    # L-infinity for two samples
#     print("Category Chisq-p-val: ", p_val) 
#     print("Category JS-div: ", jensenshannon(combined_frequencies_p, combined_frequencies_q))
#     print("Category Cosine: ", cosine(combined_frequencies_p, combined_frequencies_q))


#     print('----------Seriously different----------')

#    # Case 3 Completely different
#     sample_p = np.array(['uswest(BA,BY)', '8,9', 'uswest(BA,BY)', 'BD3.Boydton3GSGO;BN3.Boydton3GSGO', 
#                         'SN5.SN5S', 'DefaultCustomer', 'DefaultCustomer', 'aeed5fdc-b28a-4bb9-847c-0a7bc6867c50', 
#                         'aeed5fdc-b28a-4bb9-847c-0a7bc6867c50', 'DefaultCustomer'])
#     repeat_time_p = 1
#     sample_p = sample_p.repeat(repeat_time_p)
#     np.random.shuffle(sample_p)

#     sample_q = np.array(['aff384ff-9720-42cd-95ad-836ef8652a65', '8,9', 'DefaultCustomer', 'DefaultCustomer', 
#                          'DefaultCustomer', 'uswest(BA,BY)', 'All', 'All', 'All', 'uswest(BA,BY)'])
#     repeat_time_q = 10
#     sample_q = sample_q.repeat(repeat_time_q)
#     np.random.shuffle(sample_q)

#     expected_frequencies_p = np.array([0.1, 0.0, 0.1, 0.3, 0.1, 0.2, 0.0, 0.2])
#     expected_frequencies_q = np.array([0.1, 0.3, 0.0, 0.3, 0.0, 0.0, 0.1, 0.2])

#     expected_counts_p = np.array([1, 0, 1, 3, 1, 2.0, 0, 2]) * repeat_time_p
#     expected_counts_q = np.array([1, 3, 0, 3, 0, 0.0, 1, 2]) * repeat_time_q


#     combined_frequencies_p, combined_frequencies_q, combined_counts_p, combined_counts_q, _= \
#         convert_freq_for_two_samples(sample_p, sample_q)

#     assert (combined_frequencies_p==expected_frequencies_p).all()
#     assert (combined_frequencies_q==expected_frequencies_q).all()
#     assert (combined_counts_p==expected_counts_p).all()
#     assert (combined_counts_q==expected_counts_q).all()
    
#     chisq, p_val = comp_cat_chisq(sample_p, sample_q)  # Chisquare for two samples
#     assert p_val < 0.05 

#     print("Category L-1: ", minkowski(combined_frequencies_p, combined_frequencies_q, 1))
#     print("Category KL-div: ", comp_cat_kl_div(sample_p, sample_q))
#     print("Category L-inf: ", comp_cat_l_inf(sample_p, sample_q))    # L-infinity for two samples
#     print("Category Chisq-p-val: ", p_val) 
#     print("Category JS-div: ", jensenshannon(combined_frequencies_p, combined_frequencies_q))
#     print("Category Cosine: ", cosine(combined_frequencies_p, combined_frequencies_q))




    #####################################
    # 08.15 Testing
    #####################################
    # from dyn_z_score_ineq import *

    # scale_range = np.arange(1, 11, 0.5)

    # dyn_z_ineq = DynamicZScoreIneq(is_numeric=True, scale_range=scale_range, verbose=True)
    # dyn_z_ineq.test_dynamic_z_score_syn_real(dyn_z_ineq.dir_names[100])
    
    # pool = Pool(10)
    # for dir_name in dyn_z_ineq.dir_names:
    #     pool.apply_async(dyn_z_ineq.test_dynamic_z_score_syn_real, args=(dir_name, ))
    # pool.close()
    # pool.join()


    # dyn_z_ineq = DynamicZScoreIneq(is_numeric=True, scale_range=scale_range, verbose=True)
    
    # dyn_z_ineq.test_dynamic_z_score_syn_real('~shares~asg.spaa~Solutions~AntiPhishing~Analytics~scanned_documents_+++.txt')

    # pool = Pool(10)
    # for dir_name in dyn_z_ineq.dir_names:
    #     pool.apply_async(dyn_z_ineq.test_dynamic_z_score_syn_real, args=(dir_name, ))
    # pool.close()
    # pool.join()
