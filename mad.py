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


class MAD:

    def __init__(self, is_numeric, scale_range=np.arange(1, 11, 0.5),  verbose=False):
        self.cnt = 0
        self.is_numeric = is_numeric
        self.scale_range = scale_range.reshape(len(scale_range), 1)
        self.verbose = verbose
        self.use_mae = True

        if is_numeric:
            self.DATA_DIR = '../data/numerical data'   # target directories
            self.DATA_POOL = '../data/numeric_pool'    # real similar columns 

            self.syn_base_dir = '../data/synthesis_0907/numeric'           # synthetic similar columns for training
            self.test_syn_base_dir = '../data/synthesis_0907/numeric_test' # synthetic similar columns for testing
            self.save_base_dir = '../result/mad/numeric_result'

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
            self.save_base_dir = '../result/mad/category_result'

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
        self.stat_dist_cache = np.zeros((self.total_wind_size, self.num_dist))

        self.process_ts = ProcessTimeSeries(is_numeric=self.is_numeric)

        # for verbose
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 5000)

        self.best_metric_index_list = None
        self.z_max_verbose = None
        self.z_min_verbose = None

        self.clause_limit = 1


    def pred_result(self, dists, mu, sigma):

        z_score = np.abs((dists - mu) / (sigma + 1e-5))
        max_z_score = np.max(z_score)
        threshold = np.arange(0, 1000, 1)
        result = max_z_score > threshold

        return result


    def comp_mad(self, ts):
        median = np.median(ts)
        median_list = []

        for i in range(len(ts)):
            median_list.append((np.abs(ts[i] - median)))
        
        median_dist = np.median(median_list)
        return median, median_dist
        

    def comp_mu_sigma(self, slide_wind, mad=True):
        if mad == False:
            mu = []
            sigma = []
            for i in range(self.num_dist):
                wind_ts = slide_wind[:, i]
                ts = wind_ts[~np.isnan(wind_ts)]
                mu.append(ts.mean())
                sigma.append(ts.std())

            return np.array(mu), np.array(sigma)

        else:
            median_list = []
            media_scale_list = []
            for i in range(self.num_dist):
                wind_ts = slide_wind[:, i]
                ts = wind_ts[~np.isnan(wind_ts)]
                median, median_sigma = self.comp_mad(ts)
                median_list.append(median)
                media_scale_list.append(median_sigma)

            return np.array(median_list), np.array(media_scale_list)





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



    def parse_result_update(self):
        
        dir_names = [name for name in os.listdir(self.save_base_dir) \
                        if os.path.isfile(os.path.join(self.save_base_dir, name))]

        if self.is_numeric:
            num_recall_syn = 27
        else:
            num_recall_syn = 33

        for i, dir_name in enumerate(dir_names):
            file_path = os.path.join(self.save_base_dir, dir_name)
            if i==0:
                precision, recall_real, recall_syn = load_file(file_path)
            else:
                precision_tmp, recall_real_tmp, recall_syn_tmp = load_file(file_path)

                precision = np.vstack((precision, precision_tmp))
                recall_real = np.vstack((recall_real, recall_real_tmp))
                recall_syn = np.vstack((recall_syn, recall_syn_tmp))

        TP_real = recall_real.sum(axis=1)
        FP_real = precision.sum(axis=1)
        precision_real = TP_real / (TP_real + FP_real)

        TP_syn = recall_syn.sum(axis=1)
        FP_syn = precision.sum(axis=1)
        precision_syn = TP_syn / (TP_syn + FP_syn)

        recall_real = TP_real / 25   # num_total = TP + FN
        recall_syn = TP_syn / num_recall_syn     # num_total = TP + FN

        precision_real = precision_real.mean(axis=0)
        precision_syn = precision_syn.mean(axis=0)
        recall_real = recall_real.mean(axis=0)
        recall_syn = recall_syn.mean(axis=0)

        # print("average precision real: ", precision_real)
        # print("average precision syn: ", precision_syn)
        # print("average recall_real: ", recall_real)
        # print("average recall_syn: ", recall_syn)

        return precision_real, precision_syn, recall_real, recall_syn




    def test_syn_real(self, dir_name):
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

        # triple list for result storage
        precision_arr = []
        recall_real_arr =  []
        recall_syn_arr =  []
        recall_syn_org_arr =  []

        # iterate all columns
        for i, col in enumerate(cols):

            # init dist_cache & stat_dist_cache
            self.dist_cache.fill(0)
            self.stat_dist_cache.fill(np.nan)

            # compute historical 29-day distance
            self.dist_cache[0:self.hist_wind_size] = self.comp_hist_dist_cache(df_list, col)

            # get precision
            y_pred_prec = []
            for j in range(self.pred_wind_size):   # testing day

                # compute mu, sigma on previous 29 days
                slide_wind = self.dist_cache[(1+j+self.hist_wind_size-self.hist_days):j+self.hist_wind_size]
                mu, sigma = self.comp_mu_sigma(slide_wind)


                # prediction
                sample_p = df_list[j+self.hist_wind_size][col]    # current day
                sample_q = df_list[j+self.hist_wind_size-1][col]  # previous day
                if self.is_numeric:
                    dists = comp_dist(sample_p, sample_q, dtype='numeric')
                else:
                    dists = comp_dist(sample_p, sample_q, dtype='category')

                self.dist_cache[j+self.hist_wind_size] = dists
                y_pred_tmp = self.pred_result(dists, mu, sigma)

                # compute precision
                y_pred_prec.append(y_pred_tmp)
            
            y_pred_prec = np.array(y_pred_prec)




            # get reall on real data
            slide_wind = self.dist_cache[(1+self.hist_wind_size-self.hist_days):self.hist_wind_size]
            mu, sigma = self.comp_mu_sigma(slide_wind)
            
            # Testing recall
            sim_col_list = self.get_similar_cols(dir_name, col, is_train=False) # 25 similar columns

            y_pred_recall_real = []
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
                y_pred_tmp = self.pred_result(dists, mu, sigma)


                y_pred_recall_real.append(y_pred_tmp)

            y_pred_recall_real = np.array(y_pred_recall_real)



            # # get recall on synthetic data
            slide_wind = self.dist_cache[(1+self.hist_wind_size-self.hist_days):self.hist_wind_size]
            mu, sigma = self.comp_mu_sigma(slide_wind)

            # previous day
            sample_q = df_list[self.hist_wind_size-1][col]

            num_syn_sample = self.test_generated_sample_p[i].shape[0]
            y_pred_reacall_syn = []
            for k in range(num_syn_sample):
                # current day
                sample_p = pd.Series(self.test_generated_sample_p[i][k])
                if self.is_numeric:
                    dists = comp_dist(sample_p, sample_q, dtype='numeric')
                else:
                    dists = comp_dist(sample_p, sample_q, dtype='category')

                self.dist_cache[self.hist_wind_size] = dists

                y_pred_tmp = self.pred_result(dists, mu, sigma)

                y_pred_reacall_syn.append(y_pred_tmp)

            y_pred_reacall_syn = np.array(y_pred_reacall_syn)

            precision_arr.append(y_pred_prec)
            recall_real_arr.append(y_pred_recall_real)
            recall_syn_arr.append(y_pred_reacall_syn)

        # save result
        save_path = os.path.join(self.save_base_dir, dir_name)
        save_file((np.array(precision_arr), np.array(recall_real_arr), np.array(recall_syn_arr)), save_path)


        


if __name__ == '__main__':

    ################################
    # numeric
    ################################

    scale_range = np.arange(1, 100, 1)
    mad = MAD(is_numeric=True, scale_range=scale_range, verbose=True)
                            
    pool = Pool(40)
    for dir_name in mad.dir_names:
        pool.apply_async(mad.test_syn_real, args=(dir_name, ))
    pool.close()
    pool.join()

    ################################
    # category
    ################################
    mad = MAD(is_numeric=False, scale_range=scale_range, verbose=True)

    pool = Pool(40)
    for dir_name in mad.dir_names:
        pool.apply_async(mad.test_syn_real, args=(dir_name, ))
    pool.close()
    pool.join()


