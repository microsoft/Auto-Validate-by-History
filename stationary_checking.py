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
from statsmodels.tsa.stattools import adfuller



class StationaryChecking:
    """Stationary Checking"""

    def __init__(self, is_numeric,  verbose=False):

        self.is_numeric = is_numeric
        self.verbose = verbose

        if is_numeric:
            self.DATA_DIR = '../data/numerical data'

            # self.save_base_dir = '../result/stationary_checking/numeric copy'
            self.save_base_dir = '../result/k_clause_v4_unittest/numeric_result'
            self.verbose_dir = '../result/stationary_checking/numeric/'

            self.sim_col_list = load_file('../result/sim_col_list_num.pk')

            self.dist_class = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist', 
                               'Min', 'Max', 'Mean', 'Median', 'Count', 
                               'Sum', 'Range', 'Skew', '2-moment', '3-moment', 'unique_ratio', 'complete_ratio']

        else:
            self.DATA_DIR = '../data/categorical data'

            self.save_base_dir = '../result/stationary_checking/category copy'
            self.verbose_dir = '../result/stationary_checking/category/'

            self.sim_col_list = load_file('../result/sim_col_list_cat.pk')

            self.dist_class = ['L1', 'L-inf', 'Cosine', 'Chisquare', 'Count', 'JS_Div', 'KL_div', 
                               'str_len', 'char_len', 'digit_len', 'punc_len', 'unique_ratio', 'complete_ratio',
                               'dist_val_count', 
                               'pat_l_1', 'pat_l_inf', 'pat_cos', 'pat_p_val', 'pat_js_div', 'pat_kl_div']

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]

        self.dir_col_names = [name for name in os.listdir(self.save_base_dir) \
                                if os.path.isdir(os.path.join(self.save_base_dir, name))]

        self.num_dist = len(self.dist_class)

        self.hist_wind_size = 30
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

    
    def parse_result(self):

        dir_col_names = [name for name in os.listdir(self.save_base_dir) \
                                if os.path.isdir(os.path.join(self.save_base_dir, name))]

        for i, dir_name in enumerate(dir_col_names[0:1000]):
            if i == 0:
                stat_path1 = os.path.join(self.save_base_dir, dir_name, 'stationary_stats1.pk')
                stat_arr1 = load_file(stat_path1).to_numpy()

                stat_path5 = os.path.join(self.save_base_dir, dir_name, 'stationary_stats5.pk')
                stat_arr5 = load_file(stat_path5).to_numpy()

                stat_path10 = os.path.join(self.save_base_dir, dir_name, 'stationary_stats10.pk')
                stat_arr10 = load_file(stat_path10).to_numpy()

            else:
                stat_path_tmp1 = os.path.join(self.save_base_dir, dir_name, 'stationary_stats1.pk')
                stat_arr_tmp1 = load_file(stat_path_tmp1).to_numpy()

                stat_path_tmp5 = os.path.join(self.save_base_dir, dir_name, 'stationary_stats5.pk')
                stat_arr_tmp5 = load_file(stat_path_tmp5).to_numpy()

                stat_path_tmp10 = os.path.join(self.save_base_dir, dir_name, 'stationary_stats10.pk')
                stat_arr_tmp10 = load_file(stat_path_tmp10).to_numpy()

                stat_arr1 = np.vstack((stat_arr1, stat_arr_tmp1))
                stat_arr5 = np.vstack((stat_arr5, stat_arr_tmp5))
                stat_arr10 = np.vstack((stat_arr10, stat_arr_tmp10))

        stat_res1 = stat_arr1.sum(axis=0)
        stat_res5 = stat_arr5.sum(axis=0)
        stat_res10 = stat_arr10.sum(axis=0)

        index = range(self.num_dist)
        plt.figure()
        plt.bar(index, stat_res1)
        plt.xticks([i + 0.2 for i in index], self.dist_class, rotation=90)
        plt.title("non-stationary metric count - 0.01 , total # of columns: 1000")
        plt.tight_layout()
        if self.is_numeric:
            plt.savefig('num_statinonary_1.jpg')
        else:
            plt.savefig('cat_statinonary_1.jpg')
        plt.close()

        plt.figure()
        plt.bar(index, stat_res5)
        plt.xticks([i + 0.2 for i in index], self.dist_class, rotation=90)
        plt.title("non-stationary metric count - 0.05 , total # of columns: 1000")
        plt.tight_layout()
        if self.is_numeric:
            plt.savefig('num_statinonary_5.jpg')
        else:
            plt.savefig('cat_statinonary_5.jpg')
        plt.close()

        plt.figure()
        plt.bar(index, stat_res10)
        plt.xticks([i + 0.2 for i in index], self.dist_class, rotation=90)
        plt.title("non-stationary metric count - 0.1 , total # of columns: 1000")
        plt.tight_layout()
        if self.is_numeric:
            plt.savefig('num_statinonary_10.jpg')
        else:
            plt.savefig('cat_statinonary_10.jpg')
        plt.close()

    def plot_dist_curve(self, dir_name):

        # dir_name = '~shares~asg.spaa~Solutions~AntiPhishing~Analytics~daily_malware_sourceid_+++.txt_C1' 
        dist_path = os.path.join(self.save_base_dir, dir_name, 'dist_cache')
        if os.path.exists(dist_path):

            dist_df = pd.read_csv(dist_path, sep='\t')[1:]

            dist_df.plot(figsize=(16, 14), subplots=True)
            plt.tight_layout()

            save_path = os.path.join(self.save_base_dir, dir_name, 'dist_cache.jpg')
            plt.savefig(save_path)
            plt.close()



class ProcessTimeSeries:
    """ProcessingTimeSeries"""

    def __init__(self, is_numeric,  verbose=False):

        self.is_numeric = is_numeric
        self.verbose = verbose

        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.dist_class = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist', 
                               'Min', 'Max', 'Mean', 'Median', 'Count', 
                               'Sum', 'Range', 'Skew', '2-moment', '3-moment', 'unique_ratio', 'complete_ratio']

            self.save_base_dir = '../result/process_time_series/numeric'

        else:
            self.DATA_DIR = '../data/categorical data'
            self.dist_class = ['L1', 'L-inf', 'Cosine', 'Chisquare', 'Count', 'JS_Div', 'KL_div', 
                               'str_len', 'char_len', 'digit_len', 'punc_len', 'unique_ratio', 'complete_ratio',
                               'dist_val_count', 
                               'pat_l_1', 'pat_l_inf', 'pat_cos', 'pat_p_val', 'pat_js_div', 'pat_kl_div']

            self.save_base_dir = '../result/process_time_series/category'

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]

        self.num_dist = len(self.dist_class)

        self.hist_wind_size = 29
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

    
    def stationary_checking(self, dir_name):
        "test_precision_z_score"

        # load data
        dir_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(dir_path)
        cols = df_list[0].columns.to_list()

        # iterate all columns
        for i, col in enumerate(cols):

            dist_cache = np.zeros((self.hist_wind_size, self.num_dist))
            # iterate 60 day
            for prev_day in range(0, self.hist_wind_size-1):    # prev_day 0~27
                curr_day = prev_day + 1                       # curr_day 1~28
                try:
                    if self.is_numeric:
                        dists = comp_dist(df_list[curr_day][col], 
                                            df_list[prev_day][col], dtype='numeric')
                    else:
                        dists = comp_dist(df_list[curr_day][col], 
                                            df_list[prev_day][col], dtype='category')
                                        
                    dist_cache[curr_day] = dists
                except:
                    print(dir_name)
                    print(col)
                    print(curr_day)
                    break

            is_stationary_arr = np.zeros(self.num_dist)
            dist_dict = {}

            # iterate all distance metrics
            for i in range(self.num_dist):
                stat_flag = False                 # used for log switch
                dist_ts = dist_cache[1:, i]       # get 1-dim time series

                ################## use_log=False ##################
                for lag in range(11):
                    ts = self.stationary_transform(dist_ts, lag=lag, use_log=False)
                    adf_res = adfuller(ts)        # ADF test

                    adf_val = adf_res[0]          # ADF Statistic
                    crit_val = adf_res[4]['5%']
                    is_stationary = ~(adf_val > crit_val)   # True: Stationary;   False: Non Stationary
                    if is_stationary:
                        is_stationary_arr[i] = 1
                        dist_dict.update({self.dist_class[i]: [False, lag]})   # [use_log, lag]
                        stat_flag = True
                        break

                ################## use_log=True ##################
                if not stat_flag: 
                    for lag in range(11):
                        ts = self.stationary_transform(dist_ts, lag=lag, use_log=True)
                        if ts is not None:
                            adf_res = adfuller(ts)        # ADF test

                            adf_val = adf_res[0]          # ADF Statistic
                            crit_val = adf_res[4]['5%']   # critical value
                            is_stationary = ~(adf_val > crit_val)   # True: Stationary;   False: Non Stationary
                            if is_stationary:
                                is_stationary_arr[i] = 1
                                dist_dict.update({self.dist_class[i]: [True, lag]})  # [use_log, lag]
                                stat_flag = True
                                break

                ################## non-stationary ##################  
                if not stat_flag: 
                    dist_dict.update({self.dist_class[i]: [-1, -1]})


            save_path = os.path.join(self.save_base_dir, 'is_stationary_arr', dir_name+'_'+col)
            save_file(is_stationary_arr, save_path)

            ts_stat_dict = {dir_name+'_'+col: dist_dict}     # time series stationary keys 
            save_path = os.path.join(self.save_base_dir, 'ts_stat_dict', dir_name+'_'+col)
            save_file(ts_stat_dict, save_path)

            # for unittest
            # save_path = os.path.join('../result/k_clause_v4_unittest/numeric_result', 'is_stationary_arr', dir_name+'_'+col)
            # save_file(is_stationary_arr, save_path)

            # ts_stat_dict = {dir_name+'_'+col: dist_dict}     # time series stationary keys 
            # save_path = os.path.join('../result/k_clause_v4_unittest/numeric_result', 'ts_stat_dict', dir_name+'_'+col)
            # save_file(ts_stat_dict, save_path)



    def stationary_transform(self, ts, lag, use_log=False):
        """
        make time series stationary by using difference and log transform
        ts: origial time series
        lag: the order of difference, eg 0, 1, 2, ... , 9
        use_log: True/False
        return, the time series processed
        """

        ts_copy = ts.copy()

        if use_log:
            ts_copy = self.log_diff(ts_copy, lag)
        else:
            ts_copy = self.diff(ts_copy, lag)
        
        return ts_copy


    def diff(self, ts, lag):
        """
        lag: 1, 2, ..., 10
        """
        if lag == 0:
            return ts
        else:
            ts_series = pd.Series(ts).copy()
            ts_series = ts_series.diff(lag)[lag:]
            return ts_series.to_numpy()


    def log_diff(self, ts, lag=1):
        """
        log tranformation, and then differentiate
        lag: 0, 1, 2, ..., 10
        """
        # if False in (ts >= 0):    # add bias to time series
        #     ts += np.abs(ts.min())
        
        # if 0.0 in ts:           # for the ADF test
        #     ts += 1e-5

        if False in (ts > 0):
             return None

        ts_log = np.log(ts)

        if lag == 0:
            return ts_log
        else:
            ts_series = pd.Series(ts_log).copy()
            ts_series = ts_series.diff(lag)[lag:]
            return ts_series.to_numpy()


    def parse_one_result(self, lag, use_log=False):
        if use_log:
            dir_path = os.path.join(self.save_base_dir, 'log-lag-'+str(lag))
        else:
            dir_path = os.path.join(self.save_base_dir, 'lag-'+str(lag))

        dir_col_names = [name for name in os.listdir(dir_path) \
                                if os.path.isdir(os.path.join(dir_path, name))]

        for i, dir_name in enumerate(dir_col_names[0:1000]):

            if use_log:
                stat_path1 = os.path.join(self.save_base_dir, 'log-lag-'+str(lag), dir_name, 'stationary_stats1.pk')
                stat_path5 = os.path.join(self.save_base_dir, 'log-lag-'+str(lag), dir_name, 'stationary_stats5.pk')
                stat_path10 = os.path.join(self.save_base_dir, 'log-lag-'+str(lag), dir_name, 'stationary_stats10.pk')
            else:
                stat_path1 = os.path.join(self.save_base_dir, 'lag-'+str(lag), dir_name, 'stationary_stats1.pk')
                stat_path5 = os.path.join(self.save_base_dir, 'lag-'+str(lag), dir_name, 'stationary_stats5.pk')
                stat_path10 = os.path.join(self.save_base_dir, 'lag-'+str(lag), dir_name, 'stationary_stats10.pk')

            if i == 0:
                stat_arr1 = load_file(stat_path1).to_numpy()
                stat_arr5 = load_file(stat_path5).to_numpy()
                stat_arr10 = load_file(stat_path10).to_numpy()
            else:
                stat_arr_tmp1 = load_file(stat_path1).to_numpy()
                stat_arr_tmp5 = load_file(stat_path5).to_numpy()
                stat_arr_tmp10 = load_file(stat_path10).to_numpy()

                stat_arr1 = np.vstack((stat_arr1, stat_arr_tmp1))
                stat_arr5 = np.vstack((stat_arr5, stat_arr_tmp5))
                stat_arr10 = np.vstack((stat_arr10, stat_arr_tmp10))

        stat_res1 = stat_arr1.sum(axis=0)
        stat_res5 = stat_arr5.sum(axis=0)
        stat_res10 = stat_arr10.sum(axis=0)

        index = range(self.num_dist)
        # plt.figure()
        # plt.bar(index, stat_res1)
        # plt.xticks([i + 0.2 for i in index], self.dist_class, rotation=90)
        # title = "non-stat metric count lag-{}, use-log:{}, 0.01".format(lag, use_log)
        # plt.title(title)
        # plt.tight_layout()
        # if self.is_numeric:
        #     plt.savefig('num_'+title+'.jpg')
        # else:
        #     plt.savefig('cat_'+title+'.jpg')
        # plt.close()

        plt.figure()
        plt.bar(index, stat_res5)
        plt.xticks([i + 0.2 for i in index], self.dist_class, rotation=90)
        title = "non-stat metric count lag-{}, use-log:{}, 0.05".format(lag, use_log)
        plt.title(title)
        plt.tight_layout()
        if self.is_numeric:
            plt.savefig('num_'+title+'.jpg')
        else:
            plt.savefig('cat_'+title+'.jpg')
        plt.close()

        # plt.figure()
        # plt.bar(index, stat_res10)
        # plt.xticks([i + 0.2 for i in index], self.dist_class, rotation=90)
        # title = "non-stat metric count lag-{}, use-log:{}, 0.1".format(lag, use_log)
        # plt.title(title)
        # plt.tight_layout()
        # if self.is_numeric:
        #     plt.savefig('num_'+title+'.jpg')
        # else:
        #     plt.savefig('cat_'+title+'.jpg')
        # plt.close()

    def parse_stat(self):

        dir_path = os.path.join(self.save_base_dir, 'is_stationary_arr')
        dir_col_names = [name for name in os.listdir(dir_path) \
                            if os.path.isfile(os.path.join(dir_path, name))]
        stat_res_arr = []
        for dir_col_name in dir_col_names:

            file_path = os.path.join(self.save_base_dir, 'is_stationary_arr', dir_col_name)
            stat_res = load_file(file_path)
            stat_res_arr.append(stat_res)

        stat_res_arr = np.array(stat_res_arr)
        stats = stat_res_arr.sum(axis=0)

        index = range(self.num_dist)
        plt.figure()
        plt.bar(index, stats)
        plt.xticks([i + 0.2 for i in index], self.dist_class, rotation=90)
        title = "stat metric count, 0.05"
        plt.title(title)
        plt.tight_layout()
        if self.is_numeric:
            plt.savefig('num_'+title+'.jpg')
        else:
            plt.savefig('cat_'+title+'.jpg')
        plt.close()


        # merge ts_stat_dict for dyn_z_score_v4
        dir_path = os.path.join(self.save_base_dir, 'ts_stat_dict')
        dir_col_names = [name for name in os.listdir(dir_path) \
                            if os.path.isfile(os.path.join(dir_path, name))]
        ts_stat_dicts = {}
        for dir_col_name in dir_col_names:
            file_path = os.path.join(self.save_base_dir, 'ts_stat_dict', dir_col_name)
            ts_stat_dict = load_file(file_path)
            ts_stat_dicts.update(ts_stat_dict)

        save_path = os.path.join(self.save_base_dir, 'ts_stat_dicts.pk')

        save_file(ts_stat_dicts, save_path)


    def debug(self):
        dir_path = os.path.join(self.save_base_dir, 'is_stationary_arr')
        dir_col_names = [name for name in os.listdir(dir_path) \
                            if os.path.isfile(os.path.join(dir_path, name))]

        for dir_name in self.dir_names:
            # load data
            dir_path = os.path.join(self.DATA_DIR, dir_name)
            df_list = load_file(dir_path)
            cols = df_list[0].columns.to_list()

            # iterate all columns
            for i, col in enumerate(cols):

                dir_col_name = dir_name + '_' + col
                if dir_col_name not in dir_col_names:
                    print(dir_col_name)


if __name__ == '__main__':

    #####################
    # StationaryChecking
    #####################
    # sc = StationaryChecking(is_numeric=True)

    # dir_name = '~shares~asg.spaa~Solutions~AntiPhishing~Analytics~daily_malware_sourceid_+++.txt'
    # sc.stationary_checking(dir_name)
    # sc.parse_result()
    # pool = Pool(15)
    # for dir_name in sc.dir_names:
    #     pool.apply_async(sc.stationary_checking, args=(dir_name, ))
    # pool.close()
    # pool.join()


    # pool = Pool(30)
    # for dir_name in sc.dir_col_names[0:1000]:
    #     pool.apply_async(sc.plot_dist_curve, args=(dir_name, ))
    # pool.close()
    # pool.join()


    # sc = StationaryChecking(is_numeric=False)
    # dir_name = '~shares~DeepLinks~DeeplinksData~MobileDashboard~Defects~Mobile_defect_CCR_+++.txt'
    # sc.stationary_checking(dir_name)
    # for dir_name in sc.dir_names:
    #     try:
    #         sc.stationary_checking(dir_name)
    #     except:
    #         print(dir_name)
    #         continue
    # sc.parse_result()

    # pool = Pool(30)
    # for dir_name in sc.dir_col_names[0:1000]:
    #     pool.apply_async(sc.plot_dist_curve, args=(dir_name, ))
    # pool.close()
    # pool.join()


    # pool = Pool(30)
    # for dir_name in sc.dir_names:
    #     pool.apply_async(sc.stationary_checking, args=(dir_name, ))
    # pool.close()
    # pool.join()


    #####################
    # ProcessTimeSeries
    #####################
    # numeric
    # process_ts = ProcessTimeSeries(is_numeric=True)
    # process_ts.stationary_checking('~shares~searchDM~LocalProjects~CompetitiveMetrics~++~ie11DailyPreRender_+++.txt')

    # for dir_name in process_ts.dir_names:
    #     process_ts.stationary_checking(dir_name)

    # pool = Pool(30)
    # for dir_name in process_ts.dir_names:
    #     pool.apply_async(process_ts.stationary_checking, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # process_ts.stationary_checking('~shares~searchDM~LocalProjects~BingForSchool~Weekly~WeeklyBFSUUMetrics_+++.txt')
    # process_ts.stationary_checking(process_ts.dir_names[0])
    # process_ts.parse_stat()
    # process_ts.debug()

    # # process_ts.parse_result()


    # # category
    process_ts = ProcessTimeSeries(is_numeric=False)
    # process_ts.parse_result()
    # process_ts.stationary_checking(process_ts.dir_names[0])
    # # # process_ts.stationary_checking('~shares~autopilot~Silver~NeMo~Machines~+++~azDataCenters_+++.tsv')
    
    # pool = Pool(30)
    # for dir_name in process_ts.dir_names:
    #     pool.apply_async(process_ts.stationary_checking, args=(dir_name, ))
    # pool.close()
    # pool.join()

    process_ts.parse_stat()