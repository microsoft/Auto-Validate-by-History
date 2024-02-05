import numpy as np
from numpy.core.fromnumeric import sort
import os
import matplotlib.pyplot as plt
import pandas as pd
from sympy import appellf1
from utils import *
from multiprocessing import Pool
from dists import *


class TFDV:

    def __init__(self, is_numeric, threshold=None,  verbose=False):
        
        self.is_numeric = is_numeric
        self.threshold = threshold
        self.verbose = verbose
    
        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.DATA_POOL = '../data/numeric_pool'    # similar columns for real recall test is here
            self.dist_class = ['JS_div']  
            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')
            self.save_base_dir = '../result/tfdv/numeric'
            self.test_syn_base_dir = '../data/synthesis_0907/numeric_test'

        else:
            self.DATA_DIR = '../data/categorical data'
            self.DATA_POOL = '../data/category_pool'   # similar columns for real recall test is here
            self.dist_class = ['L-inf']
            self.sim_col_list = load_file('../result/sim_col_list_cat_09.pk')
            self.save_base_dir = '../result/tfdv/category'
            self.test_syn_base_dir = '../data/synthesis_0907/category_test'

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                            if os.path.isfile(os.path.join(self.DATA_DIR, name))]

        self.hist_wind_size = 29
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size +self.pred_wind_size

        self.pert_day = 29


    def test_tfdv(self, dir_name):
        """
        test precison, reall on real data, recall on synthetical data
        """

        dir_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(dir_path)
        df_list.pop(self.pert_day)
        cols = df_list[0].columns.to_list()

        # real similar cols
        sim_col_dicts = [i for i in self.sim_col_list if i['target_dir'] == dir_name]

        # load testing synthetic data
        test_syn_data_path = os.path.join(self.test_syn_base_dir, dir_name)
        self.test_generated_sample_p = load_file(test_syn_data_path)


        FP_list = []
        TP_real_list = []
        TP_syn_list = []
        # iterate all columns
        for col_idx, col in enumerate(cols):

            y_pred = []
            # predict 30 days
            for i in range(self.pred_wind_size): 

                sample_p = df_list[self.hist_wind_size+i][col]    # 29
                sample_q = df_list[self.hist_wind_size+i-1][col]  # 28

                if self.is_numeric:
                    sample_p = sample_p.dropna()
                    sample_q = sample_q.dropna()
                    try:
                        dists = compute_js_divergence(sample_p, sample_q)
                        y_pred.append(list(dists > self.threshold))
                    except:
                        y_pred.append(y_pred[i-1])

                else:
                    sample_p = sample_p.dropna().astype(str)
                    sample_q = sample_q.dropna().astype(str)
                    try:
                        dists = comp_cat_l_inf(sample_p, sample_q)
                        y_pred.append(list(dists > self.threshold))
                    except:
                        y_pred.append(y_pred[i-1])

            y_pred = np.array(y_pred)


            fp = y_pred.sum(axis=0)         # (30, 20)  -> (20, )
            FP_list.append(fp)


            ##################### real ######################
            sim_col_dict = sim_col_dicts[col_idx]
  
            target_col = sim_col_dict['target_col']
            assert(target_col == col)
            
            similar_col_list = sim_col_dict['similar_col'][1::2]   # 50% x 50 days

            y_pred = []
            for j, item in enumerate(similar_col_list): 

                sim_dir = list(item.keys())[0]      
                sim_col = item[sim_dir]

                sim_dir_path = os.path.join(self.DATA_POOL, sim_dir)
                sim_df_list = load_file(sim_dir_path)
                sample_p = sim_df_list[self.hist_wind_size][sim_col]
                sample_q = df_list[self.hist_wind_size-1][target_col]

                if self.is_numeric:
                    sample_p = sample_p.dropna()
                    sample_q = sample_q.dropna()
                    try:
                        dists = compute_js_divergence(sample_p, sample_q)
                        y_pred.append(list(dists > self.threshold))
                    except:
                        y_pred.append(y_pred[j-1])

                else:
                    sample_p = sample_p.dropna().astype(str)
                    sample_q = sample_q.dropna().astype(str)
                    try:
                        dists = comp_cat_l_inf(sample_p, sample_q)
                        y_pred.append(list(dists > self.threshold))
                    except:                                    # np.histogram 
                        y_pred.append(y_pred[j-1])

            y_pred = np.array(y_pred)
            tp = y_pred.sum(axis=0)  # (50, 20, 16)  ==>   (20, 16)
            TP_real_list.append(tp)

            ##################### syn ######################
            y_pred = []
            num_syn_sample = self.test_generated_sample_p[col_idx].shape[0]
            for k in range(num_syn_sample):

                sample_p = pd.Series(self.test_generated_sample_p[col_idx][k])
                sample_q = df_list[self.hist_wind_size-1][col]

                if self.is_numeric:
                    sample_p = sample_p.dropna()
                    sample_q = sample_q.dropna()
                    try:
                        dists = compute_js_divergence(sample_p, sample_q)
                        y_pred.append(list(dists > self.threshold))
                    except:
                        y_pred.append(y_pred[k-1])

                else:
                    sample_p = sample_p.dropna().astype(str)
                    sample_q = sample_q.dropna().astype(str)
                    try:
                        dists = comp_cat_l_inf(sample_p, sample_q)
                        y_pred.append(list(dists > self.threshold))
                    except:                                    # np.histogram 
                        y_pred.append(y_pred[k-1])

            y_pred = np.array(y_pred)
            
            tp = y_pred.sum(axis=0)  # (50, 20, 16)  ==>   (20, 16)
            TP_syn_list.append(tp)


        save_path = os.path.join(self.save_base_dir, dir_name)
        save_file((FP_list, TP_real_list, TP_syn_list), save_path)






    def test_recall_on_real_data(self, dir_name):
        """test recall tfdv"""

        sim_col_dicts = [i for i in self.sim_col_list if i['target_dir'] == dir_name]

        dir_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(dir_path)
        df_list.pop(self.pert_day)

        recall_list = []
        for sim_col_dict in sim_col_dicts:

            target_col = sim_col_dict['target_col']
            similar_col_list = sim_col_dict['similar_col'][1::2]   # 50% x 50 days

            y_pred = []
            for j, item in enumerate(similar_col_list): 

                sim_dir = list(item.keys())[0]      
                sim_col = item[sim_dir]

                sim_dir_path = os.path.join(self.DATA_POOL, sim_dir)
                sim_df_list = load_file(sim_dir_path)
                sample_p = sim_df_list[self.hist_wind_size][sim_col]
                sample_q = df_list[self.hist_wind_size-1][target_col]

                if self.is_numeric:
                    sample_p = sample_p.dropna()
                    sample_q = sample_q.dropna()
                    try:
                        dists = compute_js_divergence(sample_p, sample_q)
                        y_pred.append(list(dists > self.threshold))
                    except:
                        y_pred.append(y_pred[j-1])

                else:
                    sample_p = sample_p.dropna().astype(str)
                    sample_q = sample_q.dropna().astype(str)
                    try:
                        dists = comp_cat_l_inf(sample_p, sample_q)
                        y_pred.append(list(dists > self.threshold))
                    except:                                    # np.histogram 
                        y_pred.append(y_pred[j-1])

            y_pred = np.array(y_pred)
            
            num_tp = y_pred.sum(axis=0)  # (50, 20, 16)  ==>   (20, 16)
            num_total = y_pred.shape[0]
            recall = num_tp/num_total

            recall_list.append(recall)

        return np.array(recall_list)



    def test_recall_on_syn_data(self, dir_name):

        # load syn data
        dir_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(dir_path)
        df_list.pop(self.pert_day)
        cols = df_list[0].columns.to_list()

        # load testing synthetic data
        test_syn_data_path = os.path.join(self.test_syn_base_dir, dir_name)
        self.test_generated_sample_p = load_file(test_syn_data_path)

        recall_list = []
        for i, col in enumerate(cols):

            y_pred = []
            num_syn_sample = self.test_generated_sample_p[i].shape[0]
            for k in range(num_syn_sample):

                sample_p = pd.Series(self.test_generated_sample_p[i][k])
                sample_q = df_list[self.hist_wind_size-1][col]

                if self.is_numeric:
                    sample_p = sample_p.dropna()
                    sample_q = sample_q.dropna()
                    try:
                        dists = compute_js_divergence(sample_p, sample_q)
                        y_pred.append(list(dists > self.threshold))
                    except:
                        y_pred.append(y_pred[k-1])

                else:
                    sample_p = sample_p.dropna().astype(str)
                    sample_q = sample_q.dropna().astype(str)
                    try:
                        dists = comp_cat_l_inf(sample_p, sample_q)
                        y_pred.append(list(dists > self.threshold))
                    except:                                    # np.histogram 
                        y_pred.append(y_pred[k-1])

            y_pred = np.array(y_pred)
            
            num_tp = y_pred.sum(axis=0)  # (50, 20, 16)  ==>   (20, 16)
            num_total = y_pred.shape[0]
            recall = num_tp/num_total

            recall_list.append(recall)

        return np.array(recall_list) 



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
                FP_list, TP_real_list, TP_syn_list = load_file(file_path)
                FP_arr = np.array(FP_list)
                TP_real_arr = np.array(TP_real_list)
                TP_syn_arr = np.array(TP_syn_list)

            else:
                FP_list_tmp, TP_real_list_tmp, TP_syn_list_tmp = load_file(file_path)
                FP_arr_tmp = np.array(FP_list_tmp)
                TP_real_arr_tmp = np.array(TP_real_list_tmp)
                TP_syn_arr_tmp = np.array(TP_syn_list_tmp)

                FP_arr = np.vstack((FP_arr, FP_arr_tmp))
                TP_real_arr = np.vstack((TP_real_arr, TP_real_arr_tmp))
                TP_syn_arr = np.vstack((TP_syn_arr, TP_syn_arr_tmp))
        
    
        precision_real = TP_real_arr / (TP_real_arr + FP_arr)
        precision_syn = TP_syn_arr / (TP_syn_arr + FP_arr)
        recall_real = TP_real_arr / 25.0
        recall_syn = TP_syn_arr / num_recall_syn

        precision_real = pd.DataFrame(precision_real).mean(axis=0).to_numpy()
        precision_syn = pd.DataFrame(precision_syn).mean(axis=0).to_numpy()
        recall_real = pd.DataFrame(recall_real).mean(axis=0).to_numpy()
        recall_syn = pd.DataFrame(recall_syn).mean(axis=0).to_numpy()

        # print("average precision_real: ", precision_real)
        # print("average precision_syn: ", precision_syn)
        # print("average recall_real: ", recall_real)
        # print("average recall_syn: ", recall_syn)

        return precision_real, precision_syn, recall_real, recall_syn


if __name__ == '__main__':

    
    #################################
    # Numeric
    #################################
    threshold = np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 
                          0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                          0.9, 1.0, 10, 100])
            
    tfdv = TFDV(is_numeric=True, threshold=threshold, verbose=False)

    pool = Pool(10)
    for dir_name in tfdv.dir_names:
        pool.apply_async(tfdv.test_tfdv, args=(dir_name, ))
    pool.close()
    pool.join()

    # tfdv.parse_result_update()


    # #################################
    # # Category
    # #################################
    # threshold = np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 
    #                       0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
    #                       0.9, 1.0, 10, 100, 100])

    # tfdv = TFDV(is_numeric=False, threshold=threshold, verbose=False)

    # pool = Pool(10)
    # for dir_name in tfdv.dir_names:
    #     pool.apply_async(tfdv.test_tfdv, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # tfdv.parse_result_update()