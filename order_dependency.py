from re import I
import numpy as np
from numpy.core.fromnumeric import sort
import os
import matplotlib.pyplot as plt
from numpy.ma.core import count
import pandas as pd

from utils import *
from dists import *
from preprocessing import *
from multiprocessing import Pool


class OrderDep:
    """K-Clause"""

    def __init__(self, is_numeric, scale_range=np.arange(1, 11, 0.5),  verbose=False):

        self.is_numeric = is_numeric
        self.scale_range = scale_range.reshape(len(scale_range), 1)
        self.verbose = verbose

        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.DATA_POOL = '../data/numeric_pool'

            self.syn_base_dir = '../data/synthesis_0907/numeric'
            self.test_syn_base_dir = '../data/synthesis_0907/numeric_test'
            self.save_base_dir = '../result/order_dependency/numeric_result'
            self.verbose_dir = '../result/order_dependency/numeric/'

            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')


            self.dist_class = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist', 
                               'Min', 'Max', 'Mean', 'Median', 'Count', 
                               'Sum', 'Range', 'Skew', '2-moment', '3-moment', 'unique_ratio', 'complete_ratio']
            
            # single + two distribution metrics, remove Skewness, k-moment
            self.sel_metrics = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist', 
                               'Min', 'Max', 'Mean', 'Median', 'Count', 
                               'Sum', 'Range', 'unique_ratio', 'complete_ratio']

            # single distribution metrics, remove Skewness, k-moment
            # self.sel_metrics = ['Min', 'Max', 'Mean', 'Median', 'Count', 'Sum', 'Range', 'unique_ratio', 'complete_ratio']


        else:
            self.DATA_DIR = '../data/categorical data'
            self.DATA_POOL = '../data/category_pool'

            self.syn_base_dir = '../data/synthesis_0907/category'
            self.test_syn_base_dir = '../data/synthesis_0907/category_test'
            self.save_base_dir = '../result/order_dependency/category_result'
            self.verbose_dir = '../result/order_dependency/category/'

            self.sim_col_list = load_file('../result/sim_col_list_cat_09.pk')

            self.dist_class = ['L1', 'L-inf', 'Cosine', 'Chisquare', 'Count', 'JS_Div', 'KL_div', 
                               'str_len', 'char_len', 'digit_len', 'punc_len', 'unique_ratio', 'complete_ratio',
                               'dist_val_count', 
                               'pat_l_1', 'pat_l_inf', 'pat_cos', 'pat_p_val', 'pat_js_div', 'pat_kl_div']

            # single + two distribution metrics
            self.sel_metrics = None

            # single distribution metrics
            # self.sel_metrics = ['Count', 'str_len', 'char_len', 'digit_len', 'punc_len', 'unique_ratio', 'complete_ratio', 'dist_val_count']


        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]

        self.num_dist = len(self.dist_class)

        self.hist_wind_size = 29
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size
        self.pert_day = 29

        self.dist_cache = np.zeros((self.total_wind_size, self.num_dist))

        # self.target_precision    =          [0.999,  0.99,  0.9,   0.8,   0.7,   0.6,   0.5]
        # self.norm_scale_arr      = np.array([3.27,   2.575, 1.65,  1.29,  1.35,  0.85,  0.312]).reshape((7, 1))
        # self.chebyshev_scale_arr = np.array([31.623, 10.0,  3.162, 2.236, 1.826, 1.581, 1.414]).reshape((7, 1))
        # self.cantelli_scale_arr  = np.array([31.61,  9.95,  3.0,   2.0,   1.528, 1.225, 1.0]).reshape((7, 1))

        # scale                         1.0      1.5      2.0      2.5      3.0      3.5      4.0       4.5       5.0       5.5
        expected_precision_norm_list = [0.84134, 0.93319, 0.97725, 0.99379, 0.99865, 0.99977, 0.9999, 0.9999, 0.9999, 0.9999]
        expected_precision_norm_list.extend([0.9999]*89)
        expected_precision_norm = np.array(expected_precision_norm_list)
        
        self.expected_preci_loss_norm = 2.0 * (1.0 - expected_precision_norm)
        self.expected_preci_loss_cheby = 1.0 / (self.scale_range.flatten() **2)
        self.expected_preci_loss_cantelli = 1.0 / (1 + self.scale_range.flatten() **2)  # P(Z>k) <= 1/(1+k*k)

        self.budget_list = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

        loss_arr = []
        if self.is_numeric:
            
            for metric in self.dist_class:
                if metric in ['Count', 'Mean']:
                    loss_arr.append(self.expected_preci_loss_norm)
                elif metric in ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist']:
                    loss_arr.append(self.expected_preci_loss_cantelli)
                else:  # ['Min', 'Max', 'Median', 'Sum', 'Range', 'Skew', '2-moment', '3-moment', 'unique_ratio', 'complete_ratio']
                    loss_arr.append(self.expected_preci_loss_cheby)
        else:
            for metric in self.dist_class:
                if metric in ['str_len', 'char_len', 'digit_len', 'punc_len', 'unique_ratio', 'complete_ratio', 'Count', 'dist_val_count']:
                    loss_arr.append(self.expected_preci_loss_norm)
                elif metric in []:
                    loss_arr.append(self.expected_preci_loss_cheby)
                else:   # ['L1', 'L-inf', 'Cosine', 'Chisquare', 'JS_Div', 'KL_div', 'pat_l_1', 'pat_l_inf', 'pat_cos', 'pat_p_val', 'pat_js_div', 'pat_kl_div']
                    loss_arr.append(self.expected_preci_loss_cantelli)

        self.exp_preci_loss_arr = np.array(loss_arr).T

        self.removed_dist_idx = []

        # for verbose
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 5000)

        self.best_metric_index_list = None
        self.z_max_verbose = None
        self.z_min_verbose = None



    def pred_result2(self, dists, mu, sigma):
        z_max = mu + self.scale_range *  sigma
        z_min = mu - self.scale_range *  sigma

        result = np.zeros_like(z_max)

        self.z_max_verbose = z_max
        self.z_min_verbose = z_min

        # update for one-side test
        if self.is_numeric:

            # Cantelli's inequallity for single-side pair-wise distance
            result[:, 0] = dists[0] > z_max[:, 0] # EMD            0 is best
            result[:, 1] = dists[1] > z_max[:, 1] # JS divergence
            result[:, 2] = dists[2] > z_max[:, 2] # KL divergence
            result[:, 3] = dists[3] < z_min[:, 3] # KS p-value     1 is best
            result[:, 4] = dists[4] > z_max[:, 4] # Cohen_dist

            # Chebyshev' inequality
            result[:, 5] = (dists[5] > z_max[:, 5]) + (dists[5] < z_min[:, 5])        # Min
            result[:, 6] = (dists[6] > z_max[:, 6]) + (dists[6] < z_min[:, 6])        # Max
            result[:, 8] = (dists[8] > z_max[:, 8]) + (dists[8] < z_min[:, 8])        # Median
            result[:, 12] = (dists[12] > z_max[:, 12]) + (dists[12] < z_min[:, 12])   # Skew
            result[:, 13] = (dists[13] > z_max[:, 13]) + (dists[13] < z_min[:, 13])   # 2-moment
            result[:, 14] = (dists[14] > z_max[:, 14]) + (dists[14] < z_min[:, 14])   # 3-moment
            result[:, 10] = (dists[10] > z_max[:, 10]) + (dists[10] < z_min[:, 10])   # Sum
            result[:, 11] = (dists[11] > z_max[:, 11]) + (dists[11] < z_min[:, 11])   # Range
            result[:, 15] = (dists[15] > z_max[:, 15]) + (dists[15] < z_min[:, 15])   # unique_ratio
            result[:, 16] = (dists[16] > z_max[:, 16]) + (dists[16] < z_min[:, 16])   # complete ratio

            # CLT
            result[:, 7] = (dists[7] > z_max[:, 7]) + (dists[7] < z_min[:, 7])        # Mean
            result[:, 9] = (dists[9] > z_max[:, 9]) + (dists[9] < z_min[:, 9])        # Count

        else:
            # Cantelli's inequallity for single-side pair-wise distance
            result[:, 0] = dists[0] > z_max[:, 0] # L1
            result[:, 1] = dists[1] > z_max[:, 1] # Linf
            result[:, 2] = dists[2] > z_max[:, 2] # cos
            result[:, 3] = dists[3] < z_min[:, 3] # Chisquare

            result[:, 5] = dists[5] > z_max[:, 5] # JS divergence
            result[:, 6] = dists[6] > z_max[:, 6] # KL divergence

            result[:, 14] = dists[14] > z_max[:, 14] # Pattern L1
            result[:, 15] = dists[15] > z_max[:, 15] # Pattern Linf
            result[:, 16] = dists[16] > z_max[:, 16] # Pattern cos
            result[:, 17] = dists[17] < z_min[:, 17] # Pattern Chisquare
            result[:, 18] = dists[18] > z_max[:, 18] # Pattern JS divergence
            result[:, 19] = dists[19] > z_max[:, 19] # Pattern KL divergence

            # Chebyshev' inequality

            # CLT
            result[:, 4] = (dists[4] > z_max[:, 4]) + (dists[4] < z_min[:, 4])      # Count
            result[:, 13] = (dists[13] > z_max[:, 13]) + (dists[13] < z_min[:, 13]) # Distinct Value Count 

            result[:, 7] = (dists[7] > z_max[:, 7]) + (dists[7] < z_min[:, 7]) # str_len
            result[:, 8] = (dists[8] > z_max[:, 8]) + (dists[8] < z_min[:, 8]) # char_len
            result[:, 9] = (dists[9] > z_max[:, 9]) + (dists[9] < z_min[:, 9]) # digit_len
            result[:, 10] = (dists[10] > z_max[:, 10]) + (dists[10] < z_min[:, 10]) # punc_len
            result[:, 11] = (dists[11] > z_max[:, 11]) + (dists[11] < z_min[:, 11]) # unique_ratio
            result[:, 12] = (dists[12] > z_max[:, 12]) + (dists[12] < z_min[:, 12]) # complete_ratio
    
        return result


    def pred_result(self, dists, mu, sigma):
        """
        predict result according to one-sided test/ two-sided test
        dists: dists for current day
        mu: sample mean
        sigma: sample sigma
        return: 
        """

        z_max = mu + self.norm_scale_arr *  sigma
        z_min = mu - self.norm_scale_arr *  sigma
 
        result = np.zeros_like(z_max)

        self.z_max_verbose = z_max
        self.z_min_verbose = z_min

        # update for one-side test
        if self.is_numeric:
            
            # Cantelli's inequallity for single-side pair-wise distance
            z_max = mu + self.cantelli_scale_arr *  sigma
            z_min = mu - self.cantelli_scale_arr *  sigma
            result[:, 0] = dists[0] > z_max[:, 0] # EMD            0 is best
            result[:, 1] = dists[1] > z_max[:, 1] # JS divergence
            result[:, 2] = dists[2] > z_max[:, 2] # KL divergence
            result[:, 3] = dists[3] < z_min[:, 3] # KS p-value     1 is best
            result[:, 4] = dists[4] > z_max[:, 4] # Cohen_dist

            self.z_max_verbose[:, 0:5] =  z_max[:, 0:5]
            self.z_min_verbose[:, 0:5] =  z_min[:, 0:5]

            # Chebyshev' inequality
            z_max = mu + self.chebyshev_scale_arr *  sigma
            z_min = mu - self.chebyshev_scale_arr *  sigma
            result[:, 5] = (dists[5] > z_max[:, 5]) + (dists[5] < z_min[:, 5])        # Min
            result[:, 6] = (dists[6] > z_max[:, 6]) + (dists[6] < z_min[:, 6])        # Max
            result[:, 7] = (dists[7] > z_max[:, 7]) + (dists[7] < z_min[:, 7])        # Mean
            result[:, 8] = (dists[8] > z_max[:, 8]) + (dists[8] < z_min[:, 8])        # Median
            result[:, 9] = (dists[9] > z_max[:, 9]) + (dists[9] < z_min[:, 9])        # Count
            result[:, 10] = (dists[10] > z_max[:, 10]) + (dists[10] < z_min[:, 10])   # Sum
            result[:, 11] = (dists[11] > z_max[:, 11]) + (dists[11] < z_min[:, 11])   # Range
            result[:, 15] = (dists[15] > z_max[:, 15]) + (dists[15] < z_min[:, 15])   # unique_ratio
            result[:, 16] = (dists[16] > z_max[:, 16]) + (dists[16] < z_min[:, 16])   # complete ratio

            self.z_max_verbose[:, 5:9] =  z_max[:, 5:9]
            self.z_max_verbose[:, 10:12] =  z_max[:, 10:12]
            self.z_max_verbose[:, 15:17] =  z_max[:, 15:17]
            self.z_min_verbose[:, 5:9] =  z_min[:, 5:9]
            self.z_min_verbose[:, 10:12] =  z_min[:, 10:12]
            self.z_min_verbose[:, 15:17] =  z_min[:, 15:17]

            # CLT
            z_max = mu + self.norm_scale_arr *  sigma
            z_min = mu - self.norm_scale_arr *  sigma
            result[:, 12] = (dists[12] > z_max[:, 12]) + (dists[12] < z_min[:, 12])   # Skew
            result[:, 13] = (dists[13] > z_max[:, 13]) + (dists[13] < z_min[:, 13])   # 2-moment
            result[:, 14] = (dists[14] > z_max[:, 14]) + (dists[14] < z_min[:, 14])   # 3-moment

        else:

            # Cantelli's inequallity for single-side pair-wise distance
            z_max = mu + self.cantelli_scale_arr *  sigma
            z_min = mu - self.cantelli_scale_arr *  sigma 
            result[:, 0] = dists[0] > z_max[:, 0] # L1
            result[:, 1] = dists[1] > z_max[:, 1] # Linf
            result[:, 2] = dists[2] > z_max[:, 2] # cos
            result[:, 3] = dists[3] < z_min[:, 3] # Chisquare

            result[:, 5] = dists[5] > z_max[:, 5] # JS divergence
            result[:, 6] = dists[6] > z_max[:, 6] # KL divergence

            result[:, 14] = dists[14] > z_max[:, 14] # Pattern L1
            result[:, 15] = dists[15] > z_max[:, 15] # Pattern Linf
            result[:, 16] = dists[16] > z_max[:, 16] # Pattern cos
            result[:, 17] = dists[17] < z_min[:, 17] # Pattern Chisquare
            result[:, 18] = dists[18] > z_max[:, 18] # Pattern JS divergence
            result[:, 19] = dists[19] > z_max[:, 19] # Pattern KL divergence


            self.z_max_verbose[:, 0:4] =  z_max[:, 0:4]
            self.z_max_verbose[:, 5:7] =  z_max[:, 5:7]
            self.z_max_verbose[:, 14:20] =  z_max[:, 14:20]

            self.z_min_verbose[:, 0:4] =  z_min[:, 0:4]
            self.z_min_verbose[:, 5:7] =  z_min[:, 5:7]
            self.z_min_verbose[:, 14:20] =  z_min[:, 14:20]

            # Chebyshev' inequality
            z_max = mu + self.chebyshev_scale_arr *  sigma
            z_min = mu - self.chebyshev_scale_arr *  sigma
            result[:, 4] = (dists[4] > z_max[:, 4]) + (dists[4] < z_min[:, 4])      # Count
            result[:, 13] = (dists[13] > z_max[:, 13]) + (dists[13] < z_min[:, 13]) # Distinct Value Count 

            self.z_max_verbose[:, 4] =  z_max[:, 4]
            self.z_max_verbose[:, 13] =  z_max[:, 13]

            self.z_min_verbose[:, 4] =  z_min[:, 4]
            self.z_min_verbose[:, 13] =  z_min[:, 13]

            # CLT
            z_max = mu + self.norm_scale_arr *  sigma
            z_min = mu - self.norm_scale_arr *  sigma
            result[:, 7] = dists[7] > z_max[:, 7] # str_len
            result[:, 8] = dists[8] > z_max[:, 8] # char_len
            result[:, 9] = dists[9] > z_max[:, 9] # digit_len
            result[:, 10] = dists[10] > z_max[:, 10] # punc_len
            result[:, 11] = dists[11] > z_max[:, 11] # unique_ratio
            result[:, 12] = dists[12] > z_max[:, 12] # complete_ratio
    

        return result



    def comp_hist_dist_cache(self, df_list, col):
        # Compute history threshold

        hist_dist_cache = np.zeros((self.hist_wind_size, self.num_dist))  # (30, 16)

        for prev_day in range(0, self.hist_wind_size-1):    # prev_day 0~28
            curr_day = prev_day + 1                         # curr_day 1~29
            try:
                if self.is_numeric:
                    dists = comp_dist(df_list[curr_day][col], 
                                      df_list[prev_day][col], dtype='numeric')
                else:
                    dists = comp_dist(df_list[curr_day][col], 
                                      df_list[prev_day][col], dtype='category')
                                      
                hist_dist_cache[curr_day] = dists
            except:
                hist_dist_cache[curr_day] = hist_dist_cache[prev_day]

        return hist_dist_cache


    def get_similar_cols(self, dir_name, target_col, is_train=True):
        """
        dir_name: folder
        col: which col in dir_name
        return: 50 similar cols
        """
        if is_train:
            for sim_col_dict in self.sim_col_list:
                if dir_name == sim_col_dict['target_dir']:
                    if target_col == sim_col_dict['target_col']:
                        return sim_col_dict['similar_col'][0::2]      # 25 similar columns

        else:
            for sim_col_dict in self.sim_col_list:
                if dir_name == sim_col_dict['target_dir']:
                    if target_col == sim_col_dict['target_col']:
                        return sim_col_dict['similar_col'][1::2]      # 25 similar columns




    def parse_result(self):
        
        dir_names = [name for name in os.listdir(self.save_base_dir) \
                    if os.path.isfile(os.path.join(self.save_base_dir, name))]

        for i, dir_name in enumerate(dir_names):
            file_path = os.path.join(self.save_base_dir, dir_name)
            if i==0:  
                precision, recall_real, recall_syn = load_file(file_path)
            else:
                try:
                    precision_tmp, recall_real_tmp, recall_syn_tmp = load_file(file_path)
                    precision = np.vstack((precision, precision_tmp))
                    recall_real = np.vstack((recall_real, recall_real_tmp))
                    recall_syn = np.vstack((recall_syn, recall_syn_tmp))

                except:
                    print(dir_name)
                    pass

        # print("# of dirs: ", len(dir_names))
        # print('precision shape', precision.shape)
        # print('recall_real shape', recall_real.shape)
        # print('recall_syn shape', recall_syn.shape)

        precision = precision.mean(axis=0)
        recall_real = recall_real.mean(axis=0)
        recall_syn = recall_syn.mean(axis=0)

        print("average precision: ", precision)
        print("average recall_real: ", recall_real)
        print("average recall_syn: ", recall_syn)

        return precision, recall_real, recall_syn





    def test_syn_real(self, dir_name):
        "test_precision_z_score"

        # load data
        dir_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(dir_path)
        df_list.pop(self.pert_day)
        cols = df_list[0].columns.to_list()

        # load training synthetic data
        syn_data_path = os.path.join(self.syn_base_dir, dir_name)
        self.generated_sample_p = load_file(syn_data_path)

        # load testing synthetic data
        test_syn_data_path = os.path.join(self.test_syn_base_dir, dir_name)
        self.test_generated_sample_p = load_file(test_syn_data_path)

        # triple list for result storage
        precision_arr = []
        recall_real_arr =  []
        recall_syn_arr =  []

        # iterate all columns
        for i, col in enumerate(cols):

            # compute historical 30-day 
            self.dist_cache.fill(0)
            self.dist_cache[0:self.hist_wind_size] = self.comp_hist_dist_cache(df_list, col)

            df_dist_cache = pd.DataFrame(self.dist_cache[1:self.hist_wind_size])

            # increasing, decreasing, none
            
            order_state = []
            for metric_idx in range(df_dist_cache.shape[1]):

                tmp_series = df_dist_cache.iloc[:, metric_idx]
                if tmp_series.is_monotonic_increasing == True:
                    order_state.append(1.0)
                elif tmp_series.is_monotonic_decreasing == True:
                    order_state.append(-1.0)
                else:
                    order_state.append(0.0)
                    
#             print(order_state)
            ################ precision test ################
            y_pred_preci = []
            for j in range(self.pred_wind_size):  

                sample_p = df_list[j+self.hist_wind_size][col]
                sample_q = df_list[j+self.hist_wind_size-1][col]

                if self.is_numeric:
                    dists = comp_dist(sample_p, sample_q, dtype='numeric')
                else:
                    dists = comp_dist(sample_p, sample_q, dtype='category')

                self.dist_cache[j+self.hist_wind_size] = dists
                diff = dists - self.dist_cache[j+self.hist_wind_size-1]
                
                diff[diff<=0] = -1
                diff[diff>0]=1
                pred_tmp = []
                for _idx in range(len(diff)):
                    if((order_state[_idx]==0) or (diff[_idx] == order_state[_idx])):
                        pred_tmp.append(False)
                    else:
                        pred_tmp.append(True)
#                 print(pred_tmp)
#                 print("order_state:",order_state)
#                 print("diff:", diff)
                y_pred_preci.append(pred_tmp)

            ####################recall syn###################
            sim_col_list = self.get_similar_cols(dir_name, col, is_train=False) # 25 similar columns

            y_pred_real = []
            for k, item in enumerate(sim_col_list):
                # load similar data
                sim_dir = list(item.keys())[0]
                sim_col = item[sim_dir]
                sim_dir_path = os.path.join(self.DATA_POOL, sim_dir)
                try:
                    sim_df_list = load_file(sim_dir_path)
                except:
                    print(dir_name)
                    print(col)
                    return

                sample_p = sim_df_list[self.hist_wind_size][sim_col]
                sample_q = df_list[self.hist_wind_size-1][col]

                try:
                    if self.is_numeric:
                        dists = comp_dist(sample_p, sample_q, dtype='numeric')
                    else:
                        dists = comp_dist(sample_p, sample_q, dtype='category')
                except:
                    print('comp_dist exception')
                    print(dir_name)
                    print(col)
                    return

                diff = dists - self.dist_cache[28]

                diff[diff<=0] = -1
                diff[diff>0]=1
                pred_tmp = []
                for _idx in range(len(diff)):
                    if((order_state[_idx]==0) or (diff[_idx] == order_state[_idx])):
                        pred_tmp.append(False)
                    else:
                        pred_tmp.append(True)
#                 print(pred_tmp)
                
                y_pred_real.append(pred_tmp)
        ##################################################

            # previous day
            sample_q = df_list[self.hist_wind_size-1][col]

            y_pred_syn = []
            num_syn_sample = self.test_generated_sample_p[i].shape[0]

            for k in range(num_syn_sample):
                # current day
                sample_p = pd.Series(self.test_generated_sample_p[i][k])
                if self.is_numeric:
                    dists = comp_dist(sample_p, sample_q, dtype='numeric')
                else:
                    dists = comp_dist(sample_p, sample_q, dtype='category')

                diff = dists - self.dist_cache[28]
                
                diff[diff<=0] = -1
                diff[diff>0]=1
                pred_tmp = []
                for _idx in range(len(diff)):
                    if((order_state[_idx]==0) or (diff[_idx] == order_state[_idx])):
                        pred_tmp.append(False)
                    else:
                        pred_tmp.append(True)
#                 print(pred_tmp)

                y_pred_syn.append(pred_tmp)

            precision_arr.append(y_pred_preci)
            recall_real_arr.append(y_pred_real)
            recall_syn_arr.append(y_pred_syn)

        # save result
        save_path = os.path.join(self.save_base_dir, dir_name)
        save_file((np.array(precision_arr), np.array(recall_real_arr), np.array(recall_syn_arr)), save_path)


    def parse_result_all_metric(self):
        
        dir_names = [name for name in os.listdir(self.save_base_dir) \
                    if os.path.isfile(os.path.join(self.save_base_dir, name))]

        precision_real_list = []
        precision_syn_list = []
        recall_real_list = []
        recall_syn_list = []
        for i, dir_name in enumerate(dir_names):
            file_path = os.path.join(self.save_base_dir, dir_name)
            
            if i==0:
                precision, recall_real, recall_syn = load_file(file_path)
                num_metric = precision.shape[2]
            else:
                precision_tmp, recall_real_tmp, recall_syn_tmp = load_file(file_path)
                precision = np.vstack((precision, precision_tmp))
                recall_real = np.vstack((recall_real, recall_real_tmp))
                recall_syn = np.vstack((recall_syn, recall_syn_tmp))
#         print(precision)
#         print(precision.shape)
        
        precision = precision.sum(axis=2)
        recall_real = recall_real.sum(axis=2)
        recall_syn = recall_syn.sum(axis=2)
        
#         print(precision)
#         print(precision.shape)
        
        
        precision_real_list = []
        precision_syn_list = []
        recall_real_list = []
        recall_syn_list = []
        
        for t in range(1, num_metric+1):
            
            FP_real = (precision <= t).sum(axis=1)
            TP_real = (recall_real <= t).sum(axis=1)
            precision_real_o = pd.Series(TP_real / (TP_real + FP_real)).mean()
            
        
            TP_syn = (recall_syn <= t).sum(axis=1)
            FP_syn = (precision <= t).sum(axis=1)
            precision_syn_o = pd.Series(TP_syn / (TP_syn + FP_syn)).mean()
            
            if self.is_numeric:
                num_total_real = 25
                num_total_syn = 27
            else:
                num_total_real = 25
                num_total_syn = 33
                
            recall_real_o = pd.Series(TP_real / num_total_real).mean()   # num_total = TP + FN
            recall_syn_o = pd.Series(TP_syn / num_total_syn).mean()     # num_total = TP + FN
            
            precision_real_list.append(precision_real_o)
            precision_syn_list.append(precision_syn_o)
            recall_real_list.append(recall_real_o)
            recall_syn_list.append(recall_syn_o)
            

#         print("average precision real: ", precision_real_list)
#         print("average precision syn: ", precision_syn_list)
#         print("average recall_real: ", recall_real_list)
#         print("average recall_syn: ", recall_syn_list)

        return precision_real_list, precision_syn_list, recall_real_list, recall_syn_list


if __name__ == '__main__':

    scale_range = np.arange(1, 100, 1)

    #######################
    # Numeric
    #######################

    order_dep = OrderDep(is_numeric=True, scale_range=scale_range, verbose=True)
    # order_dep.test_syn_real(order_dep.dir_names[5])

    pool = Pool(40)
    for dir_name in order_dep.dir_names:
        pool.apply_async(order_dep.test_syn_real, args=(dir_name, ))
    pool.close()
    pool.join()

    order_dep.parse_result_all_metric()


    #######################
    # Category
    #######################

#     order_dep = OrderDep(is_numeric=False, scale_range=scale_range, verbose=True)
#     # order_dep.test_syn_real(order_dep.dir_names[7])

#     pool = Pool(40)
#     for dir_name in order_dep.dir_names:
#         pool.apply_async(order_dep.test_syn_real, args=(dir_name, ))
#     pool.close()
#     pool.join()

    # k_clause.parse_result()