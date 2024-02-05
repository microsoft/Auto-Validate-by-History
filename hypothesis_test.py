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

def save_as_csv(data, res_path):
    dir_path = os.path.dirname(res_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    data.to_csv(res_path, index=False, sep='\t')


class HypoTest:
    """dynamic scale test"""

    def __init__(self, is_numeric,  verbose=False):

        self.is_numeric = is_numeric
        self.verbose = verbose

        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.DATA_POOL = '../data/numeric_pool'    # similar columns for real recall test is here
            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')
            self.save_base_dir = '../result/hypo_test/numeric_result'
            self.syn_base_dir = '../data/synthesis_0907/numeric'
            self.test_syn_base_dir = '../data/synthesis_0907/numeric_test'
            self.verbose_dir = '../result/hypo_test/numeric/'

        else:
            self.DATA_DIR = '../data/categorical data'
            self.DATA_POOL = '../data/category_pool'   # similar columns for real recall test is here
            self.sim_col_list = load_file('../result/sim_col_list_cat_09.pk')
            self.save_base_dir = '../result/hypo_test/category_result'
            self.syn_base_dir = '../data/synthesis_0907/category'
            self.test_syn_base_dir = '../data/synthesis_0907/category_test'
            self.verbose_dir = '../result/hypo_test/category/'
        
        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]

        self.hist_wind_size = 29
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size
        self.pert_day = 29
        self.threshold = np.arange(0, 1.05, 0.05)


    def get_testing_precision_on_real_data(self, df_list, col, i, dir_name=None):

        y_pred = []
        for j in range(self.pred_wind_size):   # testing day

            # prediction
            sample_p = df_list[j+self.hist_wind_size][col]    # current day
            sample_q = df_list[j+self.hist_wind_size-1][col]  # previous day
            sample_p = sample_p.dropna()
            sample_q = sample_q.dropna()

            if self.is_numeric:
                _, p_val = ks_2samp(sample_p, sample_q)        # KS distance
                y_pred_tmp = p_val < self.threshold
            else:
                sample_p = sample_p.astype(str)
                sample_q = sample_q.astype(str)
                _, p_val = comp_cat_chisq(sample_p, sample_q)  # Chisquare for two sample
                y_pred_tmp = p_val < self.threshold

            y_pred.append(y_pred_tmp)
        
        # convert to precision
        y_pred = np.array(y_pred)

        num_fp = np.array(y_pred).sum(axis=0)   
        num_total = y_pred.shape[0]         # 30
        precision = 1 - (num_fp/num_total)  # (20, 16)

        return precision


    def get_testing_recall_on_real_data(self, df_list, col, i, dir_name):
        """
        get_testing_recall_on_real_data
        """

        # Testing recall
        sim_col_list = self.get_similar_cols(dir_name, col, is_train=False) # 25 similar columns

        y_pred = []
        for k, item in enumerate(sim_col_list):
            # load similar data
            sim_dir = list(item.keys())[0]
            sim_col = item[sim_dir]
            sim_dir_path = os.path.join(self.DATA_POOL, sim_dir)
            sim_df_list = load_file(sim_dir_path)

            sample_p = sim_df_list[self.hist_wind_size][sim_col]
            sample_q = df_list[self.hist_wind_size-1][col]
            sample_p = sample_p.dropna()
            sample_q = sample_q.dropna()
            try:
                if self.is_numeric:
                    _, p_val = ks_2samp(sample_p, sample_q)        # KS distance
                    y_pred_tmp = p_val < self.threshold
                else:
                    sample_p = sample_p.astype(str)
                    sample_q = sample_q.astype(str)
                    _, p_val = comp_cat_chisq(sample_p, sample_q)  # Chisquare for two sample
                    y_pred_tmp = p_val < self.threshold
            except:
                return np.nan
            
            y_pred.append(y_pred_tmp)

        y_pred = np.array(y_pred)
        num_tp = y_pred.sum(axis=0)         # (30, 20, 16)     (20, 16)
        num_total = y_pred.shape[0]         # 30
        recall = num_tp/num_total
        
        return recall


    def get_testing_recall_on_syn_data(self, df_list, col, i, dir_name):

        # previous day
        sample_q = df_list[self.hist_wind_size-1][col]
        sample_q = sample_q.dropna()

        num_syn_sample = self.test_generated_sample_p[i].shape[0]
        y_pred = []
        for k in range(num_syn_sample):
            # current day
            sample_p = pd.Series(self.test_generated_sample_p[i][k])
            sample_p = sample_p.dropna()

            if self.is_numeric:
                _, p_val = ks_2samp(sample_p, sample_q)        # KS distance
                y_pred_tmp = p_val < self.threshold
            else:
                sample_p = sample_p.astype(str)
                sample_q = sample_q.astype(str)
                _, p_val = comp_cat_chisq(sample_p, sample_q)  # Chisquare for two sample
                y_pred_tmp = p_val < self.threshold
            
            y_pred.append(y_pred_tmp)

        y_pred = np.array(y_pred)

        num_tp = y_pred.sum(axis=0)
        num_total = y_pred.shape[0]
        recall = num_tp/num_total

        return recall


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

        precision = precision.mean(axis=0)
        recall_real = recall_real.mean(axis=0)
        recall_syn = recall_syn.mean(axis=0)

        # print("average precision: ", precision)
        # print("average recall_real: ", recall_real)
        # print("average recall_syn: ", recall_syn)

        return precision, recall_real, recall_syn


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

                TP_real = recall_real * 25
                FP_real = (1 - precision) * self.pred_wind_size
                precision_real = TP_real / (TP_real + FP_real)

                TP_syn = recall_syn * num_recall_syn
                FP_syn = (1 - precision) * self.pred_wind_size
                precision_syn = TP_syn / (TP_syn + FP_syn)

            else:
                precision_tmp, recall_real_tmp, recall_syn_tmp = load_file(file_path)
                # if i==32:
                #     print(precision_tmp)

                TP_real = recall_real_tmp * 25
                FP_real = (1 - precision_tmp) * self.pred_wind_size
                precision_real_tmp = TP_real / (TP_real + FP_real)

                TP_syn = recall_syn_tmp * num_recall_syn
                FP_syn = (1 - precision_tmp) * self.pred_wind_size
                precision_syn_tmp = TP_syn / (TP_syn + FP_syn)

                precision_real = np.vstack((precision_real, precision_real_tmp))
                precision_syn = np.vstack((precision_syn, precision_syn_tmp))
                recall_real = np.vstack((recall_real, recall_real_tmp))
                recall_syn = np.vstack((recall_syn, recall_syn_tmp))
        
        precision_real = pd.DataFrame(precision_real).mean(axis=0).to_numpy()
        precision_syn = pd.DataFrame(precision_syn).mean(axis=0).to_numpy()
        recall_real = recall_real.mean(axis=0)
        recall_syn = recall_syn.mean(axis=0)

        # print("average precision real: ", precision_real)
        # print("average precision syn: ", precision_syn)
        # print("average recall_real: ", recall_real)
        # print("average recall_syn: ", recall_syn)

        return precision_real, precision_syn, recall_real, recall_syn

    def hypo_test_syn_real(self, dir_name):
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
        
            # get precision
            precision_list = self.get_testing_precision_on_real_data(df_list, col, i, dir_name)

            # get reall on real data
            recall_real_list = self.get_testing_recall_on_real_data(df_list, col, i, dir_name)

            # get recall on synthetic data
            recall_syn_list = self.get_testing_recall_on_syn_data(df_list, col, i, dir_name)

            precision_arr.append(precision_list)
            recall_real_arr.append(recall_real_list)
            recall_syn_arr.append(recall_syn_list)

        # save result
        save_path = os.path.join(self.save_base_dir, dir_name)
        save_file((np.array(precision_arr), np.array(recall_real_arr), np.array(recall_syn_arr)), save_path)



if __name__ == '__main__':

    #######################
    # Numeric
    #######################
    hypo = HypoTest(is_numeric=True, verbose=True)
    
    pool = Pool(2)
    for dir_name in hypo.dir_names:
        pool.apply_async(hypo.hypo_test_syn_real, args=(dir_name, ))
    pool.close()
    pool.join()

    hypo.parse_result_update()

    #######################
    # Category
    #######################
    # hypo = HypoTest(is_numeric=False, verbose=True)    
      
    # pool = Pool(2)
    # for dir_name in hypo.dir_names:
    #     pool.apply_async(hypo.hypo_test_syn_real, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # hypo.parse_result_update()



