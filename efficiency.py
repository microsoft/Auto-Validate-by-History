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
from stationary_checking import *
import global_efficiency as gloeff
import random

def warm_machine(num):
    """
    warm machine to avoid error at the beginning, 
    about 1 mimute when num=30 
    """
    
    for _ in range(num):
        mtx_a = np.random.rand(10000, 10000)
        mtx_b = np.random.rand(10000, 10000)
        mtx_a * mtx_b
        

class KClause:
    """KClause"""

    def __init__(self, is_numeric, scale_range=np.arange(1, 11, 0.5),  verbose=False):
        self.cnt = 0
        self.is_numeric = is_numeric
        self.scale_range = scale_range.reshape(len(scale_range), 1)
        self.verbose = verbose

        if is_numeric:
            self.DATA_DIR = '../data/numerical data'   # target directories
            self.DATA_POOL = '../data/numeric_pool'    # real similar columns 

            self.syn_base_dir = '../data/synthesis_0907/numeric'           # synthetic similar columns for training
            self.test_syn_base_dir = '../data/synthesis_0907/numeric_test' # synthetic similar columns for testing
            self.save_base_dir = '../result/k_clause_v4/numeric_efficiency'
            self.verbose_dir = '../result/k_clause_v4/numeric/'

            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')
            self.stat_dict = load_file('../result/process_time_series/numeric/ts_stat_dicts.pk')  # stationary dictionary


            self.dist_class = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist', 
                               'Min', 'Max', 'Mean', 'Median', 'Count', 
                               'Sum', 'Range', 'Skew', '2-moment', '3-moment', 'unique_ratio', 'complete_ratio']

            self.two_distri = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist']

            self.single_distri = ['Min', 'Max', 'Mean', 'Median', 'Count', 'Sum', 'Range', 'Skew', '2-moment', '3-moment', 'unique_ratio', 'complete_ratio']

            # delete 'Skew', '2-moment', '3-moment'
            # self.sel_metrics = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist', 
            #                     'Min', 'Max', 'Mean', 'Median', 'Count', 
            #                     'Sum', 'Range', 'unique_ratio', 'complete_ratio']

            # single distribution metrics, remove Skewness, k-moment
            self.sel_metrics = ['Min', 'Max', 'Mean', 'Median', 'Count', 'Sum', 'Range', 'unique_ratio', 'complete_ratio']

        else:
            self.DATA_DIR = '../data/categorical data'
            self.DATA_POOL = '../data/category_pool'

            self.syn_base_dir = '../data/synthesis_0907/category'
            self.test_syn_base_dir = '../data/synthesis_0907/category_test'
            self.save_base_dir = '../result/k_clause_v4/category_efficiency'
            self.verbose_dir = '../result/k_clause_v4/category/'

            self.sim_col_list = load_file('../result/sim_col_list_cat_09.pk')
            self.stat_dict = load_file('../result/process_time_series/category/ts_stat_dicts.pk')

            self.dist_class = ['L1', 'L-inf', 'Cosine', 'Chisquare', 'Count', 'JS_Div', 'KL_div', 
                               'str_len', 'char_len', 'digit_len', 'punc_len', 'unique_ratio', 'complete_ratio',
                               'dist_val_count', 
                               'pat_l_1', 'pat_l_inf', 'pat_cos', 'pat_p_val', 'pat_js_div', 'pat_kl_div']

            self.single_distri = ['Count', 'str_len', 'char_len', 'digit_len', 'punc_len', 'unique_ratio', 'complete_ratio', 'dist_val_count']
            
            # self.sel_metrics = self.single_distri
            self.sel_metrics = None


        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]

        self.num_dist = len(self.dist_class)

        self.hist_wind_size = 29
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size
        self.hist_days = 7

        self.dist_cache = np.zeros((self.total_wind_size, self.num_dist))
        self.stat_dist_cache = np.zeros((self.total_wind_size, self.num_dist))


        # scale                        1.0      1.5      2.0      2.5      3.0      3.5        4.0       4.5    5.0     5.5
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

        self.process_ts = ProcessTimeSeries(is_numeric=self.is_numeric)

        # for verbose
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 5000)

        self.best_metric_index_list = None
        self.z_max_verbose = None
        self.z_min_verbose = None



    def pred_result2(self, dists, mu, sigma):
        eps = 1e-5
        z_max = mu + self.scale_range *  sigma + eps
        z_min = mu - self.scale_range *  sigma - eps

        result = np.zeros_like(z_max)

        self.z_max_verbose = z_max
        self.z_min_verbose = z_min

        # update for one-side test
        if self.is_numeric:

            # Cantelli's inequallity for single-side pair-wise distance
            result[:, 0] = dists[0] > z_max[:, 0] # EMD            0 is best
            result[:, 1] = dists[1] > z_max[:, 1] # JS divergence
            result[:, 2] = dists[2] > z_max[:, 2] # KL divergence
            result[:, 4] = dists[4] > z_max[:, 4] # Cohen_dist
            result[:, 3] = dists[3] < z_min[:, 3] # KS p-value     1 is best

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


    def get_best_clause_on_syn_data(self, df_list, col, i, j, dir_name):
        """
        get_best_clause_on_syn_data
        df_list: 60-day data
        col: a specific column to be tested 
        i: ith col
        j: jth day
        return: best_clause
        """
        # get mu and sigma in previous 30 day
        slide_wind = self.stat_dist_cache[(j+self.hist_wind_size-self.hist_days):j+self.hist_wind_size]
        # slide_wind = self.stat_dist_cache[(j+1):j+self.hist_wind_size]
        mu, sigma = self.comp_mu_sigma(slide_wind)

        # previous day
        sample_q = df_list[j+self.hist_wind_size-1][col]

        num_syn = self.generated_sample_p[i].shape[0]
        y_pred = []

        
        for k in range(num_syn):
            # current day
            sample_p = pd.Series(self.generated_sample_p[i][k])

            exclude_start = time.time()
            if self.is_numeric:
                dists = comp_dist(sample_p, sample_q, dtype='numeric')
            else:
                dists = comp_dist(sample_p, sample_q, dtype='category')
            exclude_end = time.time()

            gloeff.set_value('clause_exclude_time', gloeff.get_value('clause_exclude_time') + exclude_end - exclude_start)


            self.dist_cache[j+self.hist_wind_size] = dists
            self.stationary_processing(dir_name, col, j+self.hist_wind_size)
            dists = self.stat_dist_cache[j+self.hist_wind_size]
            y_pred_tmp = self.pred_result2(dists, mu, sigma)
            y_pred.append(y_pred_tmp)

        best_clause_set_list = []
        for budget in self.budget_list:
            best_clause_set = self.get_best_k_clause(np.array(y_pred), budget)
            best_clause_set_list.append(best_clause_set)

        return best_clause_set_list


    def get_res_on_k_clause(self, y_pred, best_clause_set):

        res = False
        for (scale_idx, dist_idx) in best_clause_set:
            res = res or y_pred[scale_idx][dist_idx]   # requir all clause predicts 0 if it's normal 

        return res


    def comp_mu_sigma(self, slide_wind):
        mu = []
        sigma = []
        for i in range(self.num_dist):
            wind_ts = slide_wind[:, i]
            ts = wind_ts[~np.isnan(wind_ts)]
            mu.append(ts.mean())
            sigma.append(ts.std())

        return np.array(mu), np.array(sigma)

            
    def get_testing_precision_on_real_data(self, df_list, col, i, dir_name=None):
        """

        """

        y_pred = []
        for j in range(self.pred_wind_size):   # testing day

            # compute mu, sigma on previous 29 days
            slide_wind = self.stat_dist_cache[(j+self.hist_wind_size-self.hist_days):j+self.hist_wind_size]
            mu, sigma = self.comp_mu_sigma(slide_wind)

            # get best clause set
            gloeff.set_value('clause_exclude_time', 0)
            clause_start = time.time()

            best_clause_set_list = self.get_best_clause_on_syn_data(df_list, col, i, j, dir_name)

            clause_end = time.time()
            # exclude single-and-two distribution computing time
            gloeff.set_value('clause_time', gloeff.get_value('clause_time') + clause_end - clause_start - gloeff.get_value('clause_exclude_time'))

            # prediction
            sample_p = df_list[j+self.hist_wind_size][col]    # current day
            sample_q = df_list[j+self.hist_wind_size-1][col]  # previous day
            if self.is_numeric:
                dists = comp_dist(sample_p, sample_q, dtype='numeric')
            else:
                dists = comp_dist(sample_p, sample_q, dtype='category')

            self.dist_cache[j+self.hist_wind_size] = dists
            self.stationary_processing(dir_name, col, j+self.hist_wind_size) 
            dists = self.stat_dist_cache[j+self.hist_wind_size]

            y_pred_tmp = self.pred_result2(dists, mu, sigma)

            # compute precision
            res_list = []
            for best_clause_set in best_clause_set_list:
                res = self.get_res_on_k_clause(np.array(y_pred_tmp), best_clause_set)
                res_list.append(res)
            
            y_pred.append(res_list)
        
        # convert to precision
        y_pred = np.array(y_pred)

        num_fp = np.array(y_pred).sum(axis=0)
        num_total = y_pred.shape[0]         
        precision = 1 - (num_fp/num_total)

        return precision


    def get_stationary(self, dir_name, col, dist_list):
        """
        return staionary for verbose
        """

        use_log_list = []
        lag_list = []

        for dist in dist_list:
            # apply stationary transform to single distribution
            if dist in self.single_distri:
                use_log, lag = self.stat_dict[dir_name + '_' + col][dist] # get a stationary processing method
            else:
                use_log = False
                lag = 0

            use_log_list.append(use_log)
            lag_list.append(lag)

        return use_log_list, lag_list


    def get_recall_on_k_clause(self, y_pred, best_clause_set):
        
        res_list = []
        for i in range(y_pred.shape[0]):
            res = False
            for (scale_idx, dist_idx) in best_clause_set:
                res = res or y_pred[i][scale_idx][dist_idx]

            res_list.append(res)
            
        num_tp = np.array(res_list).sum(axis=0)         # (30, 20, 16)     (20, 16)
        num_total = y_pred.shape[0]         # 30
        recall = num_tp/num_total

        return recall


    def get_testing_recall_on_real_data(self, df_list, col, i, dir_name):
        """
        get_testing_recall_on_real_data
        """
        slide_wind = self.stat_dist_cache[(1+self.hist_wind_size-self.hist_days):self.hist_wind_size]
        # slide_wind = self.stat_dist_cache[1:self.hist_wind_size]
        mu, sigma = self.comp_mu_sigma(slide_wind)
        
        best_clause_set_list = self.get_best_clause_on_syn_data(df_list, col, i, 0, dir_name)
        # Testing recall
        sim_col_list = self.get_similar_cols(dir_name, col, is_train=False) # 25 similar columns

        y_pred = []
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
                return np.nan
            self.dist_cache[self.hist_wind_size] = dists
            self.stationary_processing(dir_name, col, self.hist_wind_size)
            dists = self.stat_dist_cache[self.hist_wind_size]
            y_pred_tmp = self.pred_result2(dists, mu, sigma)

            res_list = []
            for best_clause_set in best_clause_set_list:
                res = self.get_res_on_k_clause(np.array(y_pred_tmp), best_clause_set)
                res_list.append(res)

            y_pred.append(res_list)

            if self.verbose: 
                for budget_idx, best_clause_set in enumerate(best_clause_set_list):
                    dist_list = []
                    dist_val_list = []
                    scale_list = []
                    sigma_list = []
                    mu_list = []
                    reason = []
                    res = False
                    for (scale_idx, dist_idx) in best_clause_set:
                        res = res or y_pred_tmp[scale_idx][dist_idx]   # requir all clause predicts 0 if it's normal
                        dist_list.append(self.dist_class[dist_idx])
                        scale_list.append(self.scale_range.flatten()[scale_idx])
                        sigma_list.append(sigma[dist_idx])
                        mu_list.append(mu[dist_idx])
                        dist_val_list.append(dists[dist_idx])

                        if y_pred_tmp[scale_idx][dist_idx] == 0:
                            reason.append(self.dist_class[dist_idx])

                    if res == 0:
                        selected_clause = pd.DataFrame({'dist': dist_list,
                                                        'z': scale_list, 
                                                        'curr_dist': dist_val_list,
                                                        'z_max': np.array(mu_list) + np.array(sigma_list) * np.array(scale_list),
                                                        'z_min': np.array(mu_list) - np.array(sigma_list) * np.array(scale_list)})

                        file_name = dir_name + '@' + col + '@' + str(k) + '$' + '@'.join(reason)
                        save_path = os.path.join(self.verbose_dir, 'test_recall_on_real_data', 
                                                'precision_budget_'+str(self.budget_list[budget_idx]), file_name)
                        save_as_csv(selected_clause, save_path)

        
        y_pred = np.array(y_pred)
        num_tp = y_pred.sum(axis=0)         # (30, 20, 16)     (20, 16)
        num_total = y_pred.shape[0]         # 30
        recall = num_tp/num_total
        
        return recall


    def get_testing_recall_on_syn_data(self, df_list, col, i, dir_name):
        
        best_clause_set_list = self.get_best_clause_on_syn_data(df_list, col, i, 0, dir_name)
        
        slide_wind = self.stat_dist_cache[(1+self.hist_wind_size-self.hist_days):self.hist_wind_size]
        # slide_wind = self.stat_dist_cache[1:self.hist_wind_size]
        mu, sigma = self.comp_mu_sigma(slide_wind)

        # previous day
        sample_q = df_list[self.hist_wind_size-1][col]

        num_syn_sample = self.test_generated_sample_p[i].shape[0]
        y_pred = []
        for k in range(num_syn_sample):
            # current day
            sample_p = pd.Series(self.test_generated_sample_p[i][k])
            if self.is_numeric:
                dists = comp_dist(sample_p, sample_q, dtype='numeric')
            else:
                dists = comp_dist(sample_p, sample_q, dtype='category')

            self.dist_cache[self.hist_wind_size] = dists
            self.stationary_processing(dir_name, col, self.hist_wind_size)
            dists = self.stat_dist_cache[self.hist_wind_size]

            y_pred_tmp = self.pred_result2(dists, mu, sigma)

            res_list = []
            for best_clause_set in best_clause_set_list:
                res = self.get_res_on_k_clause(np.array(y_pred_tmp), best_clause_set)
                res_list.append(res)

            y_pred.append(res_list)

        y_pred = np.array(y_pred)

        num_tp = y_pred.sum(axis=0)
        num_total = y_pred.shape[0]
        recall = num_tp/num_total

        return recall


    def comp_hist_dist_cache(self, df_list, col):
        # Compute history threshold

        hist_dist_cache = np.zeros((self.hist_wind_size, self.num_dist))  # (30, 16)

        for prev_day in range(0, self.hist_wind_size-1):    # prev_day 0~28
            # print('----------------')
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


    def get_scale_dist_idx(self, idx):
        """
        recover z-index and dist-index from a linear flatten array
        """
        scale_idx = int(idx / len(self.dist_class))
        dist_idx = int(idx % len(self.dist_class))

        return scale_idx, dist_idx


    def diff_two_sets(self, covered_items, new_items):
        """
        incremental benefit
        if new clause can find the anomaly, but covered clause cannot, then count++
        covered_items: 30-day predictive result based on already selected clauses
        new_items: 30-day predictive result based on a new single clause
        """
        count = 0
        for i, val in enumerate(new_items):
            # at ith day, new clause is 1, but covered clause is 0
            if val == 1 and covered_items[i] == 0 :
                count += 1
        return count


    def update_pred_state(self, covered_items, pred_state):
        """
        update pred_state in an incremental way when considering new clause
        """

        updated_pred_state = np.empty((len(self.scale_range), len(self.dist_class)))
        updated_pred_state.fill(-np.inf)

        for i in range(len(self.scale_range)):
            for j in range(len(self.dist_class)):
                if j not in self.removed_dist_idx:
                    updated_pred_state[i, j] = self.diff_two_sets(covered_items, pred_state[:, i, j])

        return updated_pred_state


    def parse_result(self):
        
        dir_names = [name for name in os.listdir(self.save_base_dir) \
                    if os.path.isfile(os.path.join(self.save_base_dir, name))]

        # for dir_name in self.dir_names:
        #     if dir_name not in dir_names:
        #         print(dir_name)

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

    def compare_precision(self):
        precision, recall_real, recall_syn = self.parse_result()
        target_precision = 1-np.array(self.budget_list)
        plt.scatter(target_precision, precision)
        plt.ylabel("actual precision")
        plt.xlabel("target precision")
        plt.xlim(0.9, 1)
        # plt.ylim(0.9, 1)
        if self.is_numeric:
            plt.title('Numeric: Precision Comparison')
            plt.savefig('Numeric: Precision Comparison.jpg')
        else:
            plt.title('Category: Precision Comparison')
            plt.savefig('Category: Precision Comparison.jpg')


    def get_best_k_clause(self, pred_state, budget):

        pred_state_cpy = pred_state.copy()
        self.removed_dist_idx.clear()

        if self.sel_metrics is not None:
            del_metrics = set(self.dist_class).difference(set(self.sel_metrics))
            for del_metric in del_metrics:
                del_metric_idx = self.dist_class.index(del_metric)
                pred_state_cpy[:, :, del_metric_idx] = -np.inf
                self.removed_dist_idx.append(del_metric_idx)


        covered_items = np.zeros_like(pred_state_cpy[:, 0, 0])   # reset 30-day result to zero
        best_clause_set = []
        cost = 0.0

        for i in range(len(self.dist_class)):
            
            if i==0:
                W = pred_state_cpy.sum(axis=0) / self.exp_preci_loss_arr
            else:
                W = self.update_pred_state(covered_items, pred_state_cpy) / self.exp_preci_loss_arr # incremental benefits

            W = W.flatten()
            W_sort_idx = np.argsort(-W) # big -> small
            if W[W_sort_idx[0]] == 0:
                return best_clause_set

            scale_idx, dist_idx = self.get_scale_dist_idx(W_sort_idx[0])
            precision_loss = self.exp_preci_loss_arr[scale_idx, dist_idx]

            # if current clause is covered
            if (cost + precision_loss) < budget:
                best_clause_set.append((scale_idx, dist_idx))
                cost += precision_loss
                covered_items = (covered_items + pred_state_cpy[:, scale_idx, dist_idx]) >= 1

            # remove dist metric
            pred_state_cpy[:, :, dist_idx] = -np.inf
            self.removed_dist_idx.append(dist_idx)

        return best_clause_set



    def test_syn_real(self, dir_name):
        "test_precision_z_score"
        
        # warm_machine(150)

        # get column names
        dir_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(dir_path)
        cols = df_list[0].columns.to_list()

        # load training synthetic data
        syn_data_path = os.path.join(self.syn_base_dir, dir_name)
        self.generated_sample_p = load_file(syn_data_path)

        # load testing synthetic data
        test_syn_data_path = os.path.join(self.test_syn_base_dir, dir_name)
        self.test_generated_sample_p = load_file(test_syn_data_path)

        # iterate all columns
        for i, col in enumerate(cols):
            
            for num_iter in range(10):

                total_time_list = [col, num_iter]
                single_time_list = [col, num_iter]
                two_time_list = [col, num_iter]
                clause_time_list = [col, num_iter]

                for num_sel_samples in [1000, 2000, 3000, 4000, 5000]:
                
                    # sub-sampling
                    df_list = load_file(dir_path)
                    idx = random.sample(range(5000), num_sel_samples)
                    for num_day in range(len(df_list)):
                        df_list[num_day] = df_list[num_day].iloc[idx]

                    df_list.pop(29)                 # delete the 29th day

                    gloeff._init_dict()             # init all time to 0
                    start = time.time()

                    # init dist_cache & stat_dist_cache
                    self.dist_cache.fill(0)
                    self.stat_dist_cache.fill(np.nan)

                    # compute historical 29-day distance
                    self.dist_cache[0:self.hist_wind_size] = self.comp_hist_dist_cache(df_list, col)
                    self.stationary_processing(dir_name, col, 0)   # 0 represent processing historical data
 
                    # get precision
                    precision_list = self.get_testing_precision_on_real_data(df_list, col, i, dir_name)

                    end = time.time()

                    total_time_list.append(end-start)

                    single_time_list.append(gloeff.get_value('single_time'))
                    two_time_list.append(gloeff.get_value('two_time'))
                    clause_time_list.append(gloeff.get_value('clause_time'))

                gloeff.update_df('total_df', [total_time_list])
                gloeff.update_df('single_df', [single_time_list])
                gloeff.update_df('two_df', [two_time_list])
                gloeff.update_df('clause_df', [clause_time_list])

            save_path = os.path.join(self.save_base_dir, dir_name + '_' + col + '.pk')
            save_file(gloeff.glo_eff_df, save_path)


    def get_stats(self):

        target_path = os.path.join(self.verbose_dir, 'test_precision_on_real_data', 'precision_budget_0.001')
        file_names = [name for name in os.listdir(target_path) \
                          if os.path.isfile(os.path.join(target_path, name))]
        if self.is_numeric:
            self.dist_class.extend(['dist', 'div'])
        else:
            self.dist_class.extend(['dist', 'div', 'ratio', 'len', 'count', 'inf', '1', 'cos' , 'val'])

        count_list = np.zeros(len(self.dist_class))
        
        for file_name in file_names:

            res = file_name.split('_')

            if res[-1] in self.dist_class:
                idx = self.dist_class.index(res[-1])
                count_list[idx] += 1
        index = range(len(count_list))

        plt.figure()
        plt.bar(index, count_list)
        plt.xticks([i + 0.2 for i in index], self.dist_class, rotation=90)

        plt.title("total fail case: "+ str(len(file_names)))
        plt.tight_layout()
        if self.is_numeric:
            plt.savefig('num_bar_stats.jpg')
        else:
            plt.savefig('cat_bar_stats.jpg')

        plt.close()

        
    def stationary_processing(self, dir_name, col, j):
        """
        make the time series stationary
        j: ith day to update
            if j==0, then process historical 29-day data
            if j!=0, then process jth-day data
        """

        # iterate all distance metric
        for i in range(self.num_dist):
            
            # apply stationary transform to single distribution
            if self.dist_class[i] in self.single_distri:
                use_log, lag = self.stat_dict[dir_name + '_' + col][self.dist_class[i]] # get a stationary processing method
            else:
                use_log = False
                lag = 0

            # process historical 29-day sequence
            if j==0:
                # stationary after processing
                if use_log != -1:
                    dist_ts = self.dist_cache[1:self.hist_wind_size, i]
                    stat_dist_ts = self.process_ts.stationary_transform(dist_ts, lag=lag, use_log=use_log)
                    self.stat_dist_cache[(lag+1):self.hist_wind_size, i] = stat_dist_ts
                # non-stationary after processing
                else:
                    self.stat_dist_cache[:, i].fill(0)

            # process jth day sequence
            else:
                # stationary after processing
                if use_log != -1:
                    dist_ts = self.dist_cache[(j-lag):(j+1), i]
                    stat_dist_ts = self.process_ts.stationary_transform(dist_ts, lag=lag, use_log=use_log)
                    self.stat_dist_cache[j, i] = stat_dist_ts
                # non-stationary after processing
                else:
                    self.stat_dist_cache[:, i].fill(0)




if __name__ == '__main__':
    

    gloeff._init()

    scale_range = np.arange(1, 100, 1)

    # k_clause = KClause(is_numeric=True, scale_range=scale_range, verbose=False)
    # # k_clause.test_syn_real('~shares~searchDM~Rover~BEx~DataDrop~BExDataPipeline_ExploreMore_New_+++.txt')

    # selected_dir_names = ['~shares~BingEngagement~BingEngagement~Streams~Orders~Input~Merged_OrdersLineItemAttribute_+++.tsv',
    #                         '~shares~IndexQuality~Bingdex~Prod~IndexExplorer~NonUrlContainersL1TopN.+++.tsv',
    #                         '~shares~BingEngagement~BingEngagement~Streams~Orders~Input~Merged_Orders_+++.tsv',
    #                         '~shares~bingads.algo.prod.rap~shared~BiddingAgentV2~BidDeltas~BidDelta_Campaign_+++.tsv',
    #                         '~shares~bingads.algo.prod.rap~shared~BiddingAgent~BidDeltas~BidDelta_Listing_+++.tsv',
    #                         '~shares~bingads.algo.prod.rap~shared~BiddingAgentV2~BidDeltas~BidDelta_Listing_+++.tsv',
    #                         '~shares~autopilot~Silver~NeMo~Machines~+++~slbVips_+++.tsv',
    #                         '~shares~searchPlatform.IndexGenV2.SFS~SFS~GST~Archive~NPC_SET_+++.txt',
    #                         '~shares~searchDM~Rover~BEx~DataDrop~BExDataV2_+++.txt',
    #                         '~shares~searchPlatform.IndexGenV2.SFS~SFS~GST~InProgress~TaskSubmitter~NPC_SET_+++.txt',
    #                         '~shares~searchDM~Rover~BEx~DataDrop~Rewards_+++.txt',
    #                         '~shares~BingEngagement~BingEngagement~Streams~Orders~Input~Merged_OrdersLineItem_+++.tsv',   # 5 cols
    #                         '~shares~searchDM~LocalProjects~RoverPipelines~REDMOND_jeffj~395749_+++.txt']                 # 5 cols


    # for dir_name in selected_dir_names:
    #     k_clause.test_syn_real(dir_name)

    # pool = Pool(13)
    # for dir_name in selected_dir_names:
    #     pool.apply_async(k_clause.test_syn_real, args=(dir_name, ))
    # pool.close()
    # pool.join()


    ################################
    # category
    ################################
    k_clause = KClause(is_numeric=False, scale_range=scale_range, verbose=False)
    k_clause.test_syn_real('~shares~searchDM~LocalProjects~CompetitiveMetrics~++~mobileieCompetitiveDailyUserCount_+++.txt')
    
    # selected_dir_names = ['~shares~BingEngagement~BingEngagement~Streams~Orders~Input~Merged_OrdersLineItemAttribute_+++.tsv',
    #                     '~shares~BingEngagement~BingEngagement~Streams~Orders~Input~Merged_OrdersLineItem_+++.tsv',
    #                     '~shares~bingads.algo.pa~data~Team~ProductAds~DSAT~EntityDSATs.+++.tsv',
    #                     '~shares~bingads.marketplace.tools~VerticalSegmentMapping~RNRAMD~Data~Dev~QBR_BingQBR_+++.txt',
    #                     '~shares~searchDM~LocalProjects~ChitChat~++~ChitchatDomainQueries_+++.txt',
    #                     '~shares~bingads.marketplace.tools~Daily_IPM_WebsiteCountry_FormcodeClassification~Dev~QBR_BingQBR_+++.txt',
    #                     '~shares~searchDM~LocalProjects~RoverPipelines~REDMOND_jeffj~395749_+++.txt',
    #                     '~shares~searchDM~LocalProjects~RoverPipelines~FAREAST_v-wisun~337248_+++.txt',
    #                     '~shares~searchDM~LocalProjects~CompetitiveMetrics~++~mobileieCompetitiveDailyUserCount_+++.txt',
    #                     '~shares~searchDM~LocalProjects~UrlSearchEngineFeature~++~UrlFeature_+++.txt']


    # pool = Pool(13)
    # for dir_name in selected_dir_names:
    #     pool.apply_async(k_clause.test_syn_real, args=(dir_name, ))
    # pool.close()
    # pool.join()




