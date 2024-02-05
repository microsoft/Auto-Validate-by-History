
from pickle import load
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.ma.core import count
import pandas as pd
from utils import *
from dists import *
from preprocessing import *
from multiprocessing import Pool
import time

# anomaly detection
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

class ExtractFeature:
    def __init__(self, is_numeric):
        self.is_numeric = is_numeric
            
        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.DATA_POOL = '../data/numeric_pool'
            self.save_base_dir = '../data/ml_baseline/numeric'
            self.syn_base_dir = '../data/synthesis_0907/numeric'           # synthetic similar columns for training
            self.test_syn_base_dir = '../data/synthesis_0907/numeric_test' # synthetic similar columns for testing
            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')

        else:
            self.DATA_DIR = '../data/categorical data'
            self.DATA_POOL = '../data/category_pool'
            self.save_base_dir = '../data/ml_baseline/category'
            self.syn_base_dir = '../data/synthesis_0907/category'
            self.test_syn_base_dir = '../data/synthesis_0907/category_test'
            self.sim_col_list = load_file('../result/sim_col_list_cat_09.pk')

            


        self.hist_wind_size = 30
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]


    
    def extract_feature(self, sample_p):
        # load real data
        row_count_p = len(sample_p)   # row number
        sample_p = sample_p.dropna()

        if self.is_numeric:
            min_val = sample_p.min()
            max_val = sample_p.max()
            mean_val = sample_p.mean()
            median_val = sample_p.median()
            sum_val = sample_p.sum()
            range_val = max_val - min_val
            # skewness = skew(sample_p)             # check ok
            # moment2 = moment(sample_p, moment=2)
            # moment3 = moment(sample_p, moment=3)  # check ok

            unique_ratio = len(pd.unique(sample_p)) / len(sample_p)
            complete_ratio = len(sample_p) / row_count_p     # update

            features = [min_val, max_val, mean_val, median_val, row_count_p, 
                        sum_val, range_val, unique_ratio, complete_ratio]
        else:
            sample_p = sample_p.astype(str)
            str_len = int(np.mean(sample_p.apply(comp_string_len)))
            char_len = int(np.mean(sample_p.apply(comp_char_len)))
            digit_len = int(np.mean(sample_p.apply(comp_digit_len)))
            punc_len = int(np.mean(sample_p.apply(comp_punc_len)))
            unique_ratio = len(pd.unique(sample_p)) / len(sample_p)
            complete_ratio = len(sample_p) / row_count_p     # update

            dist_val_count = len(pd.unique(sample_p))

            features = [row_count_p, str_len, char_len, digit_len, punc_len, unique_ratio, complete_ratio,
                        dist_val_count]

        return np.array(features)

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


    def generate_data_syn(self, dir_name):

        # load data
        dir_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(dir_path)
        cols = df_list[0].columns.to_list()

        # load training synthetic data
        syn_data_path = os.path.join(self.syn_base_dir, dir_name)
        self.generated_sample_p = load_file(syn_data_path)

        # load testing synthetic data
        test_syn_data_path = os.path.join(self.test_syn_base_dir, dir_name)
        self.test_generated_sample_p = load_file(test_syn_data_path)

        
        for i, col in enumerate(cols):

            file_name = dir_name + '_' + col

            feature_arr = []
            for j in range(self.total_wind_size):
                sample_p = df_list[j][col]
                features = self.extract_feature(sample_p)
                feature_arr.append(features)

            feature_arr = np.array(feature_arr)
            save_path = os.path.join(self.save_base_dir, file_name, 'precision.pk')
            save_file(feature_arr, save_path)

            # generate data for recall on real data
            feature_arr = []
            sim_col_list = self.get_similar_cols(dir_name, col, is_train=False) # 25 similar columns
            for k, item in enumerate(sim_col_list):
                # load similar data
                sim_dir = list(item.keys())[0]
                sim_col = item[sim_dir]
                sim_dir_path = os.path.join(self.DATA_POOL, sim_dir)
                sim_df_list = load_file(sim_dir_path)

                sample_p = sim_df_list[self.hist_wind_size][sim_col]
                features = self.extract_feature(sample_p)
                feature_arr.append(features)

            feature_arr = np.array(feature_arr)
            save_path = os.path.join(self.save_base_dir, file_name, 'recall_real.pk')
            save_file(feature_arr, save_path)

            # generate data for recall on synthetic data
            feature_arr = []
            num_syn_sample = self.generated_sample_p[i].shape[0]
            for k in range(num_syn_sample):
                sample_p = pd.Series(self.generated_sample_p[i][k])
                features = self.extract_feature(sample_p)
                feature_arr.append(features)

            num_syn_sample = self.test_generated_sample_p[i].shape[0]
            for k in range(num_syn_sample):
                sample_p = pd.Series(self.test_generated_sample_p[i][k])
                features = self.extract_feature(sample_p)
                feature_arr.append(features)

            feature_arr = np.array(feature_arr)
            save_path = os.path.join(self.save_base_dir, file_name, 'recall_syn.pk')
            save_file(feature_arr, save_path)


class AnomalyDetection:

    def __init__(self, is_numeric, model_name):
        self.is_numeric = is_numeric
        self.model_name = model_name
            
        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.DATA_POOL = '../data/numeric_pool'
            self.save_base_dir = '../data/ml_baseline/numeric'
            self.res_save_base_dir = '../result/ml_baseline/numeric'
            self.syn_base_dir = '../data/synthesis_0907/numeric'           # synthetic similar columns for training
            self.test_syn_base_dir = '../data/synthesis_0907/numeric_test' # synthetic similar columns for testing
            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')

        else:
            self.DATA_DIR = '../data/categorical data'
            self.DATA_POOL = '../data/category_pool'
            self.save_base_dir = '../data/ml_baseline/category'
            self.res_save_base_dir = '../result/ml_baseline/category'
            self.syn_base_dir = '../data/synthesis_0907/category'
            self.test_syn_base_dir = '../data/synthesis_0907/category_test'
            self.sim_col_list = load_file('../result/sim_col_list_cat_09.pk')

        self.hist_wind_size = 30
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]

        self.threshold = np.arange(0, 1, 0.01)

    
    def test_model(self, dir_name):

        # load data
        dir_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(dir_path)
        cols = df_list[0].columns.to_list()

        precision_arr = []
        recall_real_arr = []
        recall_syn_arr = []
        for col in cols:
            outliers_fraction = 0.1
            if self.model_name == 'isof':
                model = IsolationForest(contamination=outliers_fraction, random_state=42)
            elif self.model_name == 'svm': 
                model = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)
            else:
                model = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction, novelty=True)

            # test precision
            file_name = file_name = dir_name + '_' + col
            file_path = os.path.join(self.save_base_dir, file_name, 'precision.pk')

            X = load_file(file_path)

            scaler = StandardScaler()
            scaler = scaler.fit(X)
            std_X = scaler.transform(X)

            X_train = std_X[:30]
            X_test = std_X[30:]

            model = model.fit(X_train)
            y_pred_tmp = model.decision_function(X_test)
            precision_list = []
            for t in self.threshold:
                y_pred = y_pred_tmp < t
                num_fp = y_pred.sum()
                num_total = y_pred.shape[0]
                precision = 1 - (num_fp/num_total)
                precision_list.append(precision)

            # test reall on real data
            file_path = os.path.join(self.save_base_dir, file_name, 'recall_real.pk')
            X_test_rc_real = load_file(file_path)
            X_test_rc_real = scaler.transform(X_test_rc_real)
            y_pred_tmp = model.decision_function(X_test_rc_real)
            recall_real_list = []
            for t in self.threshold:
                y_pred = y_pred_tmp < t
                num_tp = y_pred.sum()         # (30, 20, 16)     (20, 16)
                num_total = y_pred.shape[0]         # 30
                recall = num_tp/num_total
                recall_real_list.append(recall)

            # test reall on synthetic data
            file_path = os.path.join(self.save_base_dir, file_name, 'recall_syn.pk')
            X_test_rc_syn = load_file(file_path)
            X_test_rc_syn = scaler.transform(X_test_rc_syn)
            X_test_rc_syn = X_test_rc_syn[27:]
            y_pred_tmp = model.decision_function(X_test_rc_syn)

            recall_syn_list = []
            for t in self.threshold:
                y_pred = y_pred_tmp < t
                num_tp = y_pred.sum()         # (30, 20, 16)     (20, 16)
                num_total = y_pred.shape[0]         # 30
                recall = num_tp/num_total
                recall_syn_list.append(recall)

            precision_arr.append(precision_list)
            recall_real_arr.append(recall_real_list)
            recall_syn_arr.append(recall_syn_list)

        precision_arr = np.array(precision_arr)
        recall_real_arr = np.array(recall_real_arr)
        recall_syn_arr = np.array(recall_syn_arr)

        save_path = os.path.join(self.res_save_base_dir,  self.model_name, file_name)
        save_file((precision_arr, recall_real_arr, recall_syn_arr), save_path)


    def parse_result(self):
        res_path = os.path.join(self.res_save_base_dir, self.model_name)
        dir_names = [name for name in os.listdir(res_path) \
                    if os.path.isfile(os.path.join(res_path, name))]

        for i, dir_name in enumerate(dir_names):
            file_path = os.path.join(res_path, dir_name)
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
        res_path = os.path.join(self.res_save_base_dir, self.model_name)
        dir_names = [name for name in os.listdir(res_path) \
                        if os.path.isfile(os.path.join(res_path, name))]

        if self.is_numeric:
            num_recall_syn = 27
        else:
            num_recall_syn = 33

        for i, dir_name in enumerate(dir_names):
            file_path = os.path.join(res_path, dir_name)
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


class SupervisedModel:
    def __init__(self, is_numeric, model_name):
        self.is_numeric = is_numeric
        self.model_name = model_name
            
        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.DATA_POOL = '../data/numeric_pool'
            self.save_base_dir = '../data/ml_baseline/numeric'
            self.res_save_base_dir = '../result/ml_baseline/numeric'
            self.syn_base_dir = '../data/synthesis_0907/numeric'           # synthetic similar columns for training
            self.test_syn_base_dir = '../data/synthesis_0907/numeric_test' # synthetic similar columns for testing
            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')

        else:
            self.DATA_DIR = '../data/categorical data'
            self.DATA_POOL = '../data/category_pool'
            self.save_base_dir = '../data/ml_baseline/category'
            self.res_save_base_dir = '../result/ml_baseline/category'
            self.syn_base_dir = '../data/synthesis_0907/category'
            self.test_syn_base_dir = '../data/synthesis_0907/category_test'
            self.sim_col_list = load_file('../result/sim_col_list_cat_09.pk')

        self.hist_wind_size = 30
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]

        self.threshold = np.arange(-1, 1, 0.01)


    def test_model(self, dir_name):
        # load data
        dir_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(dir_path)
        cols = df_list[0].columns.to_list()

        # load training synthetic data
        syn_data_path = os.path.join(self.syn_base_dir, dir_name)
        self.generated_sample_p = load_file(syn_data_path)
        num_syn = self.generated_sample_p[0].shape[0]

        # # load testing synthetic data
        test_syn_data_path = os.path.join(self.test_syn_base_dir, dir_name)
        self.test_generated_sample_p = load_file(test_syn_data_path)

        precision_arr = []
        recall_real_arr = []
        recall_syn_arr = []
        for col in cols:

            # create model
            if self.model_name == 'lr':
                model = LogisticRegression()
            elif self.model_name == 'rf': 
                model = RandomForestClassifier(max_depth=2, random_state=42)
            elif self.model_name == 'svc': 
                model = svm.SVC(kernel='rbf', C=0.8, probability=True, random_state=42)
            elif self.model_name == 'xgb':
                model = XGBClassifier()
            # elif self.model_name == 'km':
            #     pass

            # prepare data
            file_name = file_name = dir_name + '_' + col
            file_path = os.path.join(self.save_base_dir, file_name, 'precision.pk')

            X = load_file(file_path)

            # scaling
            scaler = StandardScaler()
            scaler = scaler.fit(X)
            std_X = scaler.transform(X)

            X_train_pre = std_X[:30]
            X_test_pre = std_X[30:]

            file_path = os.path.join(self.save_base_dir, file_name, 'recall_syn.pk')
            X_rc_syn = load_file(file_path)
            X_rc_syn = scaler.transform(X_rc_syn)
            X_train_rc_syn = X_rc_syn[:num_syn]
            X_test_rc_syn = X_rc_syn[num_syn:]

            X_train = np.vstack((X_train_pre, X_train_rc_syn))
            y_train = np.hstack((np.zeros(30), np.ones(num_syn)))

            model = model.fit(X_train, y_train)

            # test precision
            y_test_tmp = model.predict_proba(X_test_pre)[:, 1]
            precision_list = []
            for t in self.threshold:
                y_pred = y_test_tmp > t
                num_fp = y_pred.sum()
                num_total = y_pred.shape[0]
                precision = 1 - (num_fp/num_total)
                precision_list.append(precision)

            # test recall on synthetic data
            y_test_tmp = model.predict_proba(X_test_rc_syn)[:, 1]
            recall_syn_list = []
            for t in self.threshold:
                y_pred = y_test_tmp > t
                num_tp = y_pred.sum()         # (30, 20, 16)     (20, 16)
                num_total = y_pred.shape[0]         # 30
                recall = num_tp/num_total
                recall_syn_list.append(recall)

            # test recall on real data
            file_path = os.path.join(self.save_base_dir, file_name, 'recall_real.pk')
            X_test_rc_real = load_file(file_path)
            X_test_rc_real = scaler.transform(X_test_rc_real)
            y_test_tmp = model.predict_proba(X_test_rc_real)[:, 1]
            recall_real_list = []
            for t in self.threshold:
                y_pred = y_test_tmp > t
                num_tp = y_pred.sum()         # (30, 20, 16)     (20, 16)
                num_total = y_pred.shape[0]         # 30
                recall = num_tp/num_total
                recall_real_list.append(recall)

            precision_arr.append(precision_list)
            recall_real_arr.append(recall_real_list)
            recall_syn_arr.append(recall_syn_list)

        precision_arr = np.array(precision_arr)
        recall_real_arr = np.array(recall_real_arr)
        recall_syn_arr = np.array(recall_syn_arr)

        save_path = os.path.join(self.res_save_base_dir,  self.model_name, file_name)
        save_file((precision_arr, recall_real_arr, recall_syn_arr), save_path)

    def parse_result(self):
        res_path = os.path.join(self.res_save_base_dir, self.model_name)
        dir_names = [name for name in os.listdir(res_path) \
                    if os.path.isfile(os.path.join(res_path, name))]

        for i, dir_name in enumerate(dir_names):
            file_path = os.path.join(res_path, dir_name)
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

        print("# of dirs: ", len(dir_names))

        precision = precision.mean(axis=0)
        recall_real = recall_real.mean(axis=0)
        recall_syn = recall_syn.mean(axis=0)

        # print("average precision: ", precision)
        # print("average recall_real: ", recall_real)
        # print("average recall_syn: ", recall_syn)

        return precision, recall_real, recall_syn


    def parse_result_update(self):
        res_path = os.path.join(self.res_save_base_dir, self.model_name)
        dir_names = [name for name in os.listdir(res_path) \
                        if os.path.isfile(os.path.join(res_path, name))]

        if self.is_numeric:
            num_recall_syn = 27
        else:
            num_recall_syn = 33

        for i, dir_name in enumerate(dir_names):
            file_path = os.path.join(res_path, dir_name)
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


if __name__ == '__main__':

    ################################
    # Feature Extraction
    ################################
    # extr_feat = ExtractFeature(is_numeric=True)
    # # extr_feat.generate_data_syn(extr_feat.dir_names[100])

    # pool = Pool(48)
    # for dir_name in extr_feat.dir_names:
    #     pool.apply_async(extr_feat.generate_data_syn, args=(dir_name, ))
    # pool.close()
    # pool.join()


    # extr_feat = ExtractFeature(is_numeric=False)
    # # extr_feat.generate_data_syn(extr_feat.dir_names[100])

    # pool = Pool(48)
    # for dir_name in extr_feat.dir_names:
    #     pool.apply_async(extr_feat.generate_data_syn, args=(dir_name, ))
    # pool.close()
    # pool.join()


    ################################
    # AnomalyDetection
    ################################

    # ad = AnomalyDetection(is_numeric=True, model_name='lof')
    # ad.parse_result_update()

    # ad.test_model(ad.dir_names[100])

    # pool = Pool(15)
    # for dir_name in ad.dir_names:
    #     pool.apply_async(ad.test_model, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # for model_name in ['isof', 'svm', 'lof']:
    #     ad = AnomalyDetection(is_numeric=True, model_name=model_name)
    #     pool = Pool(48)
    #     for dir_name in ad.dir_names:
    #         pool.apply_async(ad.test_model, args=(dir_name, ))
    #     pool.close()
    #     pool.join()

    # for model_name in ['isof', 'svm', 'lof']:
    #     ad = AnomalyDetection(is_numeric=False, model_name=model_name)
    #     pool = Pool(48)
    #     for dir_name in ad.dir_names:
    #         pool.apply_async(ad.test_model, args=(dir_name, ))
    #     pool.close()
    #     pool.join()

    # ad = AnomalyDetection(is_numeric=False, model_name='svm')
    # ad.parse_result()
    

    ####################
    # SupervisedModel
    ####################

    sm = SupervisedModel(is_numeric=True, model_name='xgb')
    # sm.test_model(sm.dir_names[100])


    # sm = SupervisedModel(is_numeric=True, model_name='xgb')
    # pool = Pool(48)
    # for dir_name in sm.dir_names:
    #     pool.apply_async(sm.test_model, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # sm = SupervisedModel(is_numeric=False, model_name='xgb')
    # pool = Pool(48)
    # for dir_name in sm.dir_names:
    #     pool.apply_async(sm.test_model, args=(dir_name, ))
    # pool.close()
    # pool.join()
    

    sm.parse_result_update()