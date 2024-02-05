from math import dist
from re import I
import numpy as np
from numpy.core.fromnumeric import sort
import os
import matplotlib.pyplot as plt
from numpy.ma.core import count
import pandas as pd
from regex import W
from sklearn.preprocessing import scale
from sqlalchemy import true
from utils import *
from dists import *
from preprocessing import *
from multiprocessing import Pool
from stationary_checking import *
import copy

class RobustDiscovery:
    """FastRule"""

    def __init__(self, is_numeric, scale_range=np.arange(1, 11, 0.5)):
        self.cnt = 0
        self.is_numeric = is_numeric
        self.scale_range = scale_range.reshape(len(scale_range), 1)

        if is_numeric:
            self.DATA_DIR = '../data/numerical data'   # target directories
            self.DATA_POOL = '../data/numeric_pool'    # real similar columns 

            self.syn_base_dir = '../data/synthesis_0907/numeric'           # synthetic similar columns for training
            self.test_syn_base_dir = '../data/synthesis_0907/numeric_test' # synthetic similar columns for testing
            self.save_base_dir = '../result/robust_discovery/numeric_result'

            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')
            self.stat_dict = load_file('../result/process_time_series/numeric/ts_stat_dicts.pk')  # stationary dictionary


            self.dist_class = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist', 
                               'Min', 'Max', 'Mean', 'Median', 'Count', 
                               'Sum', 'Range', 'Skew', '2-moment', '3-moment', 'unique_ratio', 'complete_ratio']

            self.two_distri = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist']

            self.single_distri = ['Min', 'Max', 'Mean', 'Median', 'Count', 'Sum', 'Range', 'Skew', '2-moment', '3-moment', 'unique_ratio', 'complete_ratio']

            # single+two distribution metrics, delete 'Skew', '2-moment', '3-moment'
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
            self.save_base_dir = '../result/robust_discovery/category_result'

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
        self.hist_days = 29

        self.dist_cache = np.zeros((self.total_wind_size, self.num_dist))


    def pred_result(self, dists, mu, sigma):
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


    def comp_mu_sigma(self, slide_wind):
        mu = []
        sigma = []
        for i in range(self.num_dist):
            wind_ts = slide_wind[:, i]
            ts = wind_ts[~np.isnan(wind_ts)]
            mu.append(ts.mean())
            sigma.append(ts.std())

        return np.array(mu), np.array(sigma)


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



    # def generate_clause(self, history):

    #     clause_list = []
    #     mu, sigma = self.comp_mu_sigma(history)
    #     z_max = mu + self.scale_range *  sigma
    #     z_min = mu - self.scale_range *  sigma

    #     # if this clause can cover all range
    #     # iterate metric
    #     n_metric = history.shape[1]
    #     n_scale = self.scale_range.shape[0]
    #     for i in range(n_metric):
    #         # iterate scale
    #         for j in range(n_scale):
    #             # from small to large
    #             if(self.test_clause_coverage(z_max[j][i], z_min[j][i], history[:, i])):
    #                 clause_list.append([self.dist_class[i], i, mu[i], sigma[i], self.scale_range[j], z_max[j][i], z_min[j][i]])
    #                 break
    #     return clause_list

    def is_cover_history(hist_pred):
        if np.sum(hist_pred) == 0:
            return True
        else:
            return False

    def union_two_set(set1, set2):
        res = set1 and set2
        return res

    def get_min_idx(self, weight_arr):
        # row majar layout  
        m, n = weight_arr.shape
        min_idx = np.argmin(weight_arr)

        scale_idx = int(min_idx / n)
        metrix_idx = int(min_idx % n)

        return scale_idx, metrix_idx


    def find_clause(self, history_pred, syn_pred):

        # n_history x n_scale x n_metric
        n_metric = history_pred.shape[2]
        n_scale = history_pred.shape[1]

        alphas = np.arange(0.001, 1, 0.05)
        all_clause_list = []
        for alpha in alphas:
            beta = 1.0-alpha
            clause_list = []

            weight_arr = np.zeros((n_scale, n_metric))
            for i in range(n_metric):
                for j in range(n_scale):
                    g_precision = history_pred[:, j, i]
                    v_syn = syn_pred[:, j, i]
                    weight_arr[j][i] = self.compute_weight(alpha, beta, g_precision, v_syn)
            
            weight_arr_cpy = copy.deepcopy(weight_arr)
            scale_idx, metrix_idx = self.get_min_idx(weight_arr_cpy)
            clause_list.append([self.dist_class[metrix_idx], metrix_idx, scale_idx])
            weight_arr_cpy[:, metrix_idx] = np.inf

            g_prec, v_syn = self.compute_clause_coverage(clause_list, history_pred, syn_pred)

            while(not np.sum(g_prec)==0):
                incremental_weight = self.compute_incremental_weight(alpha, beta, g_prec, v_syn, history_pred, syn_pred, weight_arr_cpy)
                scale_idx, metrix_idx = self.get_min_idx(incremental_weight)
                clause_list.append([self.dist_class[metrix_idx], metrix_idx, scale_idx])
                weight_arr_cpy[:, metrix_idx] = np.inf

                g_prec, v_syn = self.compute_clause_coverage(clause_list, history_pred, syn_pred)

                if(len(clause_list)>=n_metric):
                    break

            all_clause_list.append(clause_list)

        return all_clause_list

    def compute_clause_coverage(self, clause_list, history_pred, syn_pred):
        all_g_precision = []
        all_v_syn = []
        for clause in clause_list:
            __, metrix_idx, scale_idx = clause
            all_g_precision.append(history_pred[:, scale_idx, metrix_idx])
            all_v_syn.append(syn_pred[:, scale_idx, metrix_idx])

        all_g_precision = np.array(all_g_precision)
        all_v_syn = np.array(all_v_syn)

        g_precision = np.all(all_g_precision, axis=0).astype(int)
        v_syn = np.all(all_v_syn, axis=0).astype(int)

        return g_precision, v_syn



    def parse_pred_result(self, all_clause_list, res):
        res_arr = []
        # increase threshold, we can find less number of clauses
        for clause_list in all_clause_list:  # a threshold
            res_list = []
            assert(len(clause_list) >= 0)

            for clause in clause_list:
                _, metrix_idx, scale_idx = clause
                res_list.append(res[scale_idx][metrix_idx])

            # only if one of clause predicts zero, the result will be zero
            if 0 in res_list:
                res_arr.append(0)
            else:
                res_arr.append(1)            
                
        return np.array(res_arr)
            
    def incremental_weight(prev_w):
        curr_w = 0
        incremental_w = curr_w - prev_w
        assert(incremental_w>=0)

        return incremental_w

    def compute_weight(self, alpha, beta, g_precision, v_syn):
        n_g = len(g_precision)
        n_v = len(v_syn)
        g_precision = np.where(g_precision, 0, 1)
        v_syn = np.where(v_syn, 0, 1)
        w = alpha * (1 - np.sum(g_precision)/n_g) + beta * (np.sum(v_syn)/n_v)
        return w

    def compute_incremental_weight(self, alpha, beta, prev_g_precision, prev_v_syn, history_pred, syn_pred, weight_arr_cpy):

        n_metric = weight_arr_cpy.shape[1]
        n_scale = weight_arr_cpy.shape[0]

        w = np.zeros_like(weight_arr_cpy)

        for i in range(n_metric):
            for j in range(n_scale):

                g_precision = history_pred[:, j, i]
                v_syn = syn_pred[:, j, i]
                n_g = len(g_precision)
                n_v = len(v_syn)

                all_g_precision = np.array([prev_g_precision, g_precision])
                all_g_precision = np.all(all_g_precision, axis=0).astype(int)
                cnt_array1 = np.where(all_g_precision, 0, 1)
                cnt_array2 = np.where(prev_g_precision, 0, 1)
                incre_g_precision = np.sum(cnt_array1) - np.sum(cnt_array2)

                all_v_syn = np.array([prev_v_syn, v_syn])
                all_v_syn = np.all(all_v_syn, axis=0).astype(int)
                cnt_array1 = np.where(all_v_syn, 0, 1)
                cnt_array2 = np.where(prev_v_syn, 0, 1)
                incre_v_syn = np.sum(cnt_array1)-np.sum(cnt_array2)

                w[j][i] = alpha * (1.0-np.sum(incre_g_precision)/n_g) + beta * (np.sum(incre_v_syn)/n_v)

        n_metric = weight_arr_cpy.shape[1]
        for i in range(n_metric):
            if(weight_arr_cpy[0][i] == np.inf):
                w[:, i] = np.inf

        return w

    def test_model(self, dir_name):
        "test_precision_z_score"

        # load data
        dir_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(dir_path)
        df_list.pop(29)                 # delete the 29th day
        cols = df_list[0].columns.to_list()

        # load training synthetic data
        syn_data_path = os.path.join(self.syn_base_dir, dir_name)
        self.generated_sample_p = load_file(syn_data_path)

        # load testing synthetic data
        test_syn_data_path = os.path.join(self.test_syn_base_dir, dir_name)
        self.test_generated_sample_p = load_file(test_syn_data_path)

        # iterate all columns
        for i, col in enumerate(cols):
            try:
                # historical 30 days -> clause
                self.dist_cache.fill(0)
                self.dist_cache[0:self.hist_wind_size] = self.comp_hist_dist_cache(df_list, col)
                history_dist = self.dist_cache[1:self.hist_wind_size]
                mu, sigma = self.comp_mu_sigma(history_dist)

                # previous day
                sample_q = df_list[self.hist_wind_size-1][col]
                num_syn_sample = self.generated_sample_p[i].shape[0]

                syn_res_arr = []
                for k in range(num_syn_sample):
                    # current day
                    sample_p = pd.Series(self.test_generated_sample_p[i][k])
                    if self.is_numeric:
                        dists = comp_dist(sample_p, sample_q, dtype='numeric')
                    else:
                        dists = comp_dist(sample_p, sample_q, dtype='category')
                    syn_res_arr.append(self.pred_result(dists, mu, sigma))
                syn_res_arr = np.array(syn_res_arr)                     # n_syn x n_scale x n_metric


                his_res_arr = []
                n_history = history_dist.shape[0]
                for k in range(n_history):                              # n_history x n_metric
                    his_res_arr.append(self.pred_result(history_dist[k], mu, sigma))
                his_res_arr = np.array(his_res_arr)                     # n_history x n_scale x n_metric


                all_clause_list= self.find_clause(his_res_arr, syn_res_arr)


                # prediction
                precision = []
                for j in range(self.pred_wind_size):   # testing day
                    slide_wind = self.dist_cache[(j+self.hist_wind_size-self.hist_days+1):j+self.hist_wind_size]
                    mu, sigma = self.comp_mu_sigma(slide_wind)

                    sample_p = df_list[j+self.hist_wind_size][col]    # current day
                    sample_q = df_list[j+self.hist_wind_size-1][col]  # previous day
                    if self.is_numeric:
                        dists = comp_dist(sample_p, sample_q, dtype='numeric')
                    else:
                        dists = comp_dist(sample_p, sample_q, dtype='category')
                    self.dist_cache[j+self.hist_wind_size] = dists

                    res = self.pred_result(dists, mu, sigma)
                    precision.append(self.parse_pred_result(all_clause_list, res))
                precision = np.array(precision)


                mu, sigma = self.comp_mu_sigma(history_dist)
                sim_col_list = self.get_similar_cols(dir_name, col, is_train=False) # 25 similar columns

                recall_real = []
                for k, item in enumerate(sim_col_list):
                    # load similar data
                    sim_dir = list(item.keys())[0]
                    sim_col = item[sim_dir]
                    sim_dir_path = os.path.join(self.DATA_POOL, sim_dir)
                    sim_df_list = load_file(sim_dir_path)

                    sample_p = sim_df_list[self.hist_wind_size][sim_col]
                    sample_q = df_list[self.hist_wind_size-1][col]

                    if self.is_numeric:
                        dists = comp_dist(sample_p, sample_q, dtype='numeric')
                    else:
                        dists = comp_dist(sample_p, sample_q, dtype='category')

                    res = self.pred_result(dists, mu, sigma)
                    recall_real.append(self.parse_pred_result(all_clause_list, res))
                recall_real = np.array(recall_real)



                # previous day
                sample_q = df_list[self.hist_wind_size-1][col]
                num_syn_sample = self.test_generated_sample_p[i].shape[0]
                recall_syn = []
                for k in range(num_syn_sample):
                    # current day
                    sample_p = pd.Series(self.test_generated_sample_p[i][k])
                    if self.is_numeric:
                        dists = comp_dist(sample_p, sample_q, dtype='numeric')
                    else:
                        dists = comp_dist(sample_p, sample_q, dtype='category')

                    res = self.pred_result(dists, mu, sigma)
                    recall_syn.append(self.parse_pred_result(all_clause_list, res))
                recall_syn = np.array(recall_syn)


                file_name = dir_name + '_' + col
                save_path = os.path.join(self.save_base_dir, file_name)
                save_file((precision, recall_real, recall_syn), save_path)

            except:
                print("Execution Failed: ", dir_name, col)

    def parse_result_update(self):
        res_path = os.path.join(self.save_base_dir)
        dir_names = [name for name in os.listdir(res_path) \
                        if os.path.isfile(os.path.join(res_path, name))]

        precision_real_arr = []
        precision_syn_arr = []
        recall_real_arr = []
        recall_syn_arr = []

        for dir_name in dir_names:
            file_path = os.path.join(res_path, dir_name)

            precision, recall_real, recall_syn = load_file(file_path)
            
            FP_real = np.sum(precision, axis=0)
            TP_real = np.sum(recall_real, axis=0)
            precision_real = TP_real / (TP_real + FP_real)
            recall_real = TP_real / recall_real.shape[0]

            TP_syn = np.sum(recall_syn, axis=0)
            FP_syn = np.sum(precision, axis=0)
            precision_syn = TP_syn / (TP_syn + FP_syn)

            recall_syn = TP_syn / recall_syn.shape[0]

            
            precision_real_arr.append(precision_real)
            precision_syn_arr.append(precision_syn)
            recall_real_arr.append(recall_real)
            recall_syn_arr.append(recall_syn)


        precision_real_arr = np.array(precision_real_arr)
        precision_syn_arr = np.array(precision_syn_arr)
        recall_real_arr = np.array(recall_real_arr)
        recall_syn_arr = np.array(recall_syn_arr)

        # print("precision_real_arr", np.mean(precision_real_arr, axis=0))
        # print("precision_syn_arr", np.mean(precision_syn_arr, axis=0))
        # print("recall_real_arr", np.mean(recall_real_arr, axis=0))
        # print("recall_syn_arr", np.mean(recall_syn_arr, axis=0))

        precision_real =  pd.DataFrame(precision_real_arr).mean(axis=0).to_numpy()
        precision_syn = pd.DataFrame(precision_syn_arr).mean(axis=0).to_numpy()
        recall_real = pd.DataFrame(recall_real_arr).mean(axis=0).to_numpy()
        recall_syn = pd.DataFrame(recall_syn_arr).mean(axis=0).to_numpy()

        return precision_real, precision_syn, recall_real, recall_syn




if __name__ == '__main__':

    ################################
    # numeric
    ################################

    # rd = RobustDiscovery(is_numeric=True, scale_range=np.arange(1, 100, 1))
    # rd.test_model(rd.dir_names[5])
                            
    # pool = Pool(30)
    # for dir_name in rd.dir_names:
    #     pool.apply_async(rd.test_model, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # rd.parse_result_update()

    ################################
    # category
    ################################

    rd = RobustDiscovery(is_numeric=False, scale_range=np.arange(1, 100, 1))
    # rd.test_model(rd.dir_names[5])
                            
    pool = Pool(40)
    for dir_name in rd.dir_names:
        pool.apply_async(rd.test_model, args=(dir_name, ))
    pool.close()
    pool.join()



