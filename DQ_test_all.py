
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
# import xgboost
from xgboost import XGBClassifier


from collections import namedtuple, Counter as count
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from hyperloglog import HyperLogLog
from pyod.models.knn import KNN
from abc import abstractmethod
from dabl import detect_types
from nltk.util import ngrams
from sklearn import metrics
from pathlib import Path
from enum import IntEnum
import pandas as pd
import numpy as np
import copy
import time
from sklearn.metrics import roc_auc_score

np.random.seed(42)


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

            unique_ratio = len(pd.unique(sample_p)) / len(sample_p)
            complete_ratio = len(sample_p) / row_count_p

            features = [min_val, max_val, mean_val, median_val, row_count_p, 
                            sum_val, range_val, unique_ratio, complete_ratio]
        else:
            sample_p = sample_p.astype(str)
            str_len = int(np.mean(sample_p.apply(comp_string_len)))
            char_len = int(np.mean(sample_p.apply(comp_char_len)))
            digit_len = int(np.mean(sample_p.apply(comp_digit_len)))
            punc_len = int(np.mean(sample_p.apply(comp_punc_len)))
            unique_ratio = len(pd.unique(sample_p)) / len(sample_p)
            complete_ratio = len(sample_p) / row_count_p

            dist_val_count = len(pd.unique(sample_p))

            features = [row_count_p, str_len, char_len, digit_len, punc_len, 
                                unique_ratio, complete_ratio, dist_val_count]

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



def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r took %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


class Quality(IntEnum):
    GOOD = 0
    BAD = 1


class Learner:
    @abstractmethod
    def fit(history):
        pass

    @abstractmethod
    def predict(X):
        pass


class KNNLearner(Learner):
    def __init__(self):
        # https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.knn
        self.clf = None

    def fit(self, history):
        learner = KNN(contamination=0.01,
                      n_neighbors=5,
                      method='mean',
                      metric='euclidean',
                      algorithm='ball_tree')
        self.clf = Pipeline([
            ('scaler', MinMaxScaler()),
            ('learner', learner)
        ]).fit(history)

        return self

    def predict(self, X):
        assert self.clf is not None, ".fit first"
        
        return self.clf.predict(X)

    def predict_proba(self, X):
        assert self.clf is not None, ".fit first"
        return self.clf.predict_proba(X)[:, 1]
        # return self.clf.decision_function(X)




class DataProfiler:
    class __DP:
        def __init__(self):
            self.analyzer = {
                "Completeness": lambda x: self.completeness(x),
                "Uniqueness": lambda x: self.uniqueness(x),
                "ApproxCountDistinct": lambda x: self.approx_count_distinct(x),
                "Mean": lambda x: np.mean(x),
                "Minimum": lambda x: np.min(x),
                "Maximum": lambda x: np.max(x),
                "StandardDeviation": lambda x: np.std(x),
                "Sum": lambda x: np.sum(x),
                "Count": lambda x: x.shape[0],
                "FrequentRatio": lambda x: 1.*max(count(x).values())/x.shape[0],
                "PeculiarityIndex": lambda x: self.peculiarity(x),
            }

            self.dtype_checking = {
                "int64": True,
                "float64": True
            }


        def completeness(self, x):
            return 1. - np.sum(pd.isna(x)) / x.shape[0]

        def uniqueness(self, x):
            tmp = [i for i in count(x).values() if i == 1]
            return 1. * np.sum(tmp) / x.shape[0]

        def count_distinct(self, x):
            return 1. * len(count(x).keys()) / x.shape[0]

        def approx_count_distinct(self, x):
            hll = HyperLogLog(.01)
            for idx, val in x.items():
                hll.add(str(val))
            return len(hll)

        # TODO: count sketch, using deterministic count for small data
#        def count_sketch(self, matrix, sketch_size=50):
#            m, n = matrix.shape[0], 1
#            res = np.zeros([m, sketch_size])
#            hashedIndices = np.random.choice(sketch_size, replace=True)
#            print(hashedIndices)
#            randSigns = np.random.choice(2, n, replace=True) * 2 - 1 # a n-by-1{+1, -1} vector
#            matrix = matrix * randSigns
#            for i in range(sketch_size):
#                res[:, i] = np.sum(matrix[:, hashedIndices == i], 1)
#            return res

        def peculiarity(self, x):
            def _peculiarity_index(word, count2grams, count3grams):
                t = []
                for xyz in ngrams(str(word), 3):
                    xy, yz = xyz[:2], xyz[1:]
                    cxy, cyz = count2grams.get(xy, 0), count2grams.get(yz, 0)
                    cxyz = count3grams.get(xyz, 0)
                    t.append(.5* (np.log(cxy) + np.log(cyz) - np.log(cxyz)))
                return np.sqrt(np.mean(np.array(t)**2))

            aggregated_string = " ".join(map(str, x))
            c2gr = count(ngrams(aggregated_string, 2))
            c3gr = count(ngrams(aggregated_string, 3))
            return x.apply(lambda y: _peculiarity_index(y, c2gr, c3gr)).max()

    instance = None

    def __init__(self):
        if not DataProfiler.instance:
            DataProfiler.instance = DataProfiler.__DP()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def _compute_for_column(self, column, *analyzers):
        return [self.instance.analyzer[name](column) for name in analyzers]

    # @timeit
    def compute_for(self, batch, return_labels=False):
        profile, labels = [], []
        generic_metrics = ["Completeness", "Uniqueness",
                           "ApproxCountDistinct", "FrequentRatio"]
        numeric_metrics = ["Mean", "Minimum", "Maximum",
                           "StandardDeviation", "Sum"]

        is_free_string = detect_types(batch)['free_string']
        for col, dtype in zip(batch.columns, batch.dtypes):
            # For every column, compute generic metrics,
            # add additional numeric metrics for numeric columns
            metrics = copy.deepcopy(generic_metrics)
            if self.dtype_checking.get(dtype, False):
                metrics.extend(numeric_metrics)
            if dtype == 'object': # Dummy check for likely-to-be-strings
                metrics.append("PeculiarityIndex")
            # print(col, dtype, metrics)
            # We assume the data schema to be stable, column order unchanged,
            # no additional validation for feature order happens, optional
            column_profile = self._compute_for_column(batch[col], *metrics)
            profile.extend(column_profile)
            labels.extend([f'{col}_{m}' for m in metrics])
        return profile if not return_labels else (profile, labels)


    # def compute_features():
    #     my_num_metrics = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist', 
    #                             'Min', 'Max', 'Mean', 'Median', 'Count', 
    #                             'Sum', 'Range', 'unique_ratio', 'complete_ratio']

    #     my_cat_metrics =  ['L1', 'L-inf', 'Cosine', 'Chisquare', 'Count', 'JS_Div', 'KL_div', 
    #                            'str_len', 'char_len', 'digit_len', 'punc_len', 'unique_ratio', 'complete_ratio',
    #                            'dist_val_count', 
    #                            'pat_l_1', 'pat_l_inf', 'pat_cos', 'pat_p_val', 'pat_js_div', 'pat_kl_div']

    #     profile =0

    #     return profile


class DataQualityValidatior:
    def __init__(self):
        self.clf = KNNLearner()
        self.history = []

    def add(self, batch):
        self.history.append(batch)
        return self

    def test(self, batch):
        # print(len(self.history))
        # re-fit the model from scratch
        self.clf.fit(self.history)
        decision = self.clf.predict_proba([batch])

        return decision


Batch = namedtuple('Batch', 'id clean dirty')

def get_batch_fnames():
    folder = Path('partitions/')
    batches = []
    for day in range(1, 54):
        fclean = folder / f'clean/FBPosts_clean_{day}.tsv'
        fdirty = folder / f'dirty/FBPosts_dirty_{day}.tsv'
        assert fclean.exists()
        assert fdirty.exists()
        batches.append(Batch(day, fclean, fdirty))
    return iter(batches)


def good_or_bad(batch):
    return np.random.choice([Quality.GOOD, Quality.BAD], p=[9./10, 1./10])


class DQTest3:
    def __init__(self, is_numeric):
        self.is_numeric = is_numeric
            
        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.DATA_POOL = '../data/numeric_pool'
            self.save_base_dir = '../data/ml_baseline/numeric'
            self.res_save_base_dir = '../result/DQTest3_all_feature/numeric'
            self.syn_base_dir = '../data/synthesis_0907/numeric'           # synthetic similar columns for training
            self.test_syn_base_dir = '../data/synthesis_0907/numeric_test' # synthetic similar columns for testing
            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')

        else:
            self.DATA_DIR = '../data/categorical data'
            self.DATA_POOL = '../data/category_pool'
            self.save_base_dir = '../data/ml_baseline/category'
            self.res_save_base_dir = '../result/DQTest3_all_feature/category'
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


        for col in cols:

            # create model
            dqv = DataQualityValidatior()
            try:
                ############
                file_name = file_name = dir_name + '_' + col
                file_path = os.path.join(self.save_base_dir, file_name, 'precision.pk')
                X_data = load_file(file_path)

                for i in range(30):
                    profile = X_data[i]
                    dqv.add(profile)

                # precision
                precision_arr = []
                for i in range(30, 60):
                    profile = X_data[i]
                    res = dqv.test(profile)
                    precision_arr.append(res) 
                precision_arr = np.array(precision_arr).flatten()

                # recall real
                file_path = os.path.join(self.save_base_dir, file_name, 'recall_real.pk')
                X_test_rc_real = load_file(file_path)
                m = X_test_rc_real.shape[0]
                recall_real_arr = []
                for i in range(m):
                    profile = X_test_rc_real[i]
                    res = dqv.test(profile)
                    recall_real_arr.append(res) 
                recall_real_arr = np.array(recall_real_arr).flatten()

                # recall syn
                file_path = os.path.join(self.save_base_dir, file_name, 'recall_syn.pk')
                X_rc_syn = load_file(file_path)
                X_test_rc_syn = X_rc_syn[num_syn:]
                m = X_test_rc_syn.shape[0]
                recall_syn_arr = []
                for i in range(m):
                    profile = X_test_rc_syn[i]
                    res = dqv.test(profile)
                    recall_syn_arr.append(res) 
                recall_syn_arr = np.array(recall_syn_arr).flatten()

                file_name = dir_name + '_' + col
                save_path = os.path.join(self.res_save_base_dir, file_name)
                save_file((precision_arr, recall_real_arr, recall_syn_arr), save_path)

            except:
                print("Execution Failed: ", dir_name, col)
                continue

    def parse_result_update(self):
        res_path = os.path.join(self.res_save_base_dir)
        dir_names = [name for name in os.listdir(res_path) \
                        if os.path.isfile(os.path.join(res_path, name))]

        precision_real_arr = []
        precision_syn_arr = []
        recall_real_arr = []
        recall_syn_arr = []
        auc_real_arr = []
        auc_syn_arr = []

        for dir_name in dir_names:
            file_path = os.path.join(res_path, dir_name)

            precision, recall_real, recall_syn = load_file(file_path)

            y_true_real = np.concatenate((np.zeros(len(precision)), np.ones(len(recall_real))))
            y_true_syn = np.concatenate((np.zeros(len(precision)), np.ones(len(recall_syn))))
            y_pred_real = np.concatenate((precision, recall_real))
            y_pred_syn = np.concatenate((precision, recall_syn))

            fpr, tpr, _ = metrics.roc_curve(y_true_real, y_pred_real)
            auc = metrics.auc(fpr, tpr)
            auc_real_arr.append(auc)

            fpr, tpr, _ = metrics.roc_curve(y_true_syn, y_pred_syn)
            auc = metrics.auc(fpr, tpr)
            auc_syn_arr.append(auc)


            threshold = np.arange(0, 1.0, 0.02).reshape((1, -1))

            precision = precision.reshape((-1, 1))
            recall_real = recall_real.reshape((-1, 1))
            recall_syn = recall_syn.reshape((-1, 1))

            precision = precision > threshold
            recall_real = recall_real > threshold
            recall_syn = recall_syn > threshold




            TP_real = np.sum(recall_real, axis=0)
            FP_real = np.sum(precision, axis=0)
            precision_real = TP_real / (TP_real + FP_real)

            TP_syn = np.sum(recall_syn, axis=0)
            FP_syn = np.sum(precision, axis=0)
            precision_syn = TP_syn / (TP_syn + FP_syn)

            recall_real = TP_real / recall_real.shape[0]
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

        precision_real =  np.mean(precision_real_arr, axis=0)
        precision_syn = np.mean(precision_syn_arr, axis=0)
        recall_real = np.mean(recall_real_arr, axis=0)
        recall_syn = np.mean(recall_syn_arr, axis=0)
        auc_real = np.mean(auc_real_arr)
        auc_syn = np.mean(auc_syn_arr)


        return precision_real, precision_syn, recall_real, recall_syn, auc_real, auc_syn



if __name__ == '__main__':

    ####################
    # Numerical
    ####################

    # dq = DQTest3(is_numeric=True)
    # dq.test_model(dq.dir_names[100])


    pool = Pool(5)
    for dir_name in dq.dir_names:
        pool.apply_async(dq.test_model, args=(dir_name, ))
    pool.close()
    pool.join()


    dq = DQTest3(is_numeric=False)
    # dq.test_model(dq.dir_names[100])

    pool = Pool(40)
    for dir_name in dq.dir_names:
        pool.apply_async(dq.test_model, args=(dir_name, ))
    pool.close()
    pool.join()

