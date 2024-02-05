import requests
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import azureml.core
from azureml.core import Dataset, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.datadrift import DataDriftDetector, AlertConfiguration

import datetime
from datetime import datetime, timedelta

import os
from multiprocessing import Pool
from utils import *
from tqdm import tqdm



def save_as_csv(data, res_path):
    dir_path = os.path.dirname(res_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    data.to_csv(res_path, index=False, sep='\t')


class CSVDataGenerator:
    
    def __init__(self, is_numeric, verbose=False):

        self.is_numeric = is_numeric
        self.verbose = verbose
        

        # baseline data： 2021-5-1  ~  2021-6-28
        self.date_time = [x.strftime('%Y-%m-%d') + ' ' + \
                            x.strftime('%H:%M:%S') for x  \
                            in list(pd.date_range(start='2021-05-01', \
                            end='2021-07-03'))]

        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.DATA_POOL = '../data/numeric_pool' 
            self.PERT_DATA_DIR = '../data/synthesis_0907/numeric_test'
            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')
            self.save_base_dir = '../result/drift_detector0106/numeric'

        else:
            self.DATA_DIR = '../data/categorical data'
            self.DATA_POOL = '../data/category_pool'
            self.PERT_DATA_DIR = '../data/synthesis_0907/category_test'
            self.sim_col_list = load_file('../result/sim_col_list_cat_09.pk')
            self.save_base_dir = '../result/drift_detector0106/category'

        self.hist_wind_size = 30
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]


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
                        return sim_col_dict['similar_col'][0::2]      # 27 similar columns

        else:
            for sim_col_dict in self.sim_col_list:
                if dir_name == sim_col_dict['target_dir']:
                    if target_col == sim_col_dict['target_col']:
                        return sim_col_dict['similar_col'][1::2]      # 27 similar columns



    def generate_new(self, dir_name):
    
        file_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(file_path)
        # df_list.pop(29)                 # delete the 29th day
        cols = df_list[0].columns.to_list()
        try:
            pert_file_path = os.path.join(self.PERT_DATA_DIR, dir_name)  # synthetical perturbed samples
            pert_samples = load_file(pert_file_path)
        except:
            print(dir_name)
            return

        for j, col in enumerate(cols):
    
            # prepare data for precision test
            total_data_time = []
            total_value = []
            for i, day_value in enumerate(df_list):
                count = day_value.shape[0]
                curr_date_time = [self.date_time[i]] * count
                total_data_time.extend(curr_date_time)

                curr_val = day_value[col].to_list()
                total_value.extend(curr_val)

            data_for_precision = pd.DataFrame({'timestamp': total_data_time, 'value': total_value})

            save_path_name = dir_name + '_' + col
            save_path = os.path.join(self.save_base_dir, save_path_name, 'data_for_precision.csv')
            save_as_csv(data_for_precision, save_path)


            # used for recall dataset generation
            hist_data_time = []
            hist_value = []
            for i, day_value in enumerate(df_list[:30]):
                count = day_value.shape[0]
                curr_date_time = [self.date_time[i]] * count
                hist_data_time.extend(curr_date_time)

                curr_val = day_value[col].to_list()
                hist_value.extend(curr_val)


            # prepare data for recall test on real data    50% real 
            sim_col_list = self.get_similar_cols(dir_name, col, is_train=False) # 25 similar columns

            total_data_time = hist_data_time.copy()
            total_value = hist_value.copy()
            for k, item in enumerate(sim_col_list):

                # load similar data
                sim_dir = list(item.keys())[0]
                sim_col = item[sim_dir]
                sim_dir_path = os.path.join(self.DATA_POOL, sim_dir)
                sim_df_list = load_file(sim_dir_path)

                day_value = sim_df_list[self.hist_wind_size][sim_col].to_list()
                count = len(day_value)
                curr_date_time = [self.date_time[self.hist_wind_size+k]] * count
                total_data_time.extend(curr_date_time)
                total_value.extend(day_value)

            data_for_recall_real = pd.DataFrame({'timestamp': total_data_time, 'value': total_value})
            save_path = os.path.join(self.save_base_dir, save_path_name, 'data_for_recall_real.csv')
            save_as_csv(data_for_recall_real, save_path)


            # prepare data for recall test on synthetical data
            total_data_time = hist_data_time.copy()
            total_value = hist_value.copy()
            for k, day_value in enumerate(pert_samples[j]):
                
                count = day_value.shape[0]
                curr_date_time = [self.date_time[self.hist_wind_size+k]] * count
                total_data_time.extend(curr_date_time)
                total_value.extend(day_value.tolist())

            data_for_recall_syn = pd.DataFrame({'timestamp': total_data_time, 'value': total_value})   # 30 day

            save_path = os.path.join(self.save_base_dir, save_path_name, 'data_for_recall_syn.csv')
            save_as_csv(data_for_recall_syn, save_path)

            

class AzureDriftDetector():
    def __init__(self, is_numeric, verbose=False):
    
        self.is_numeric = is_numeric
        self.verbose = verbose

        self.date_train = [x.strftime('%Y-%m-%d') + ' ' + \
                            x.strftime('%H:%M:%S') for x  \
                            in list(pd.date_range(start='2021-05-01', \
                            end='2021-06-29'))]

        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')
            self.save_base_dir = '../data/drift_detector0106/numeric'
            self.datastore_dir = 'drift_detector0106/numeric'
            self.res_save_base_dir = '../result/drift_detector0106/numeric'
            self.precision_prefix = "num_preci_0112_"
            self.recall_syn_prefix = "num_rc_syn_0112_"
            self.recall_real_prefix = "num_rc_real_0112_"

            self.cpu_cluster1 = 'dezhantu1'
            self.cpu_cluster2 = 'japan-cluster-2'

            self.dir_col_names = [name for name in os.listdir(self.save_base_dir) \
                                    if os.path.isdir(os.path.join(self.save_base_dir, name))]

        else:
            self.DATA_DIR = '../data/categorical data'
            self.sim_col_list = load_file('../result/sim_col_list_cat_09.pk')
            self.save_base_dir = '../data/drift_detector0106/category/'
            self.datastore_dir = 'drift_detector0106/category'
            self.res_save_base_dir = '../result/drift_detector0106/category'
            self.precision_prefix = "cat_preci_0112_"
            self.recall_syn_prefix = "cat_rc_syn_0112_"
            self.recall_real_prefix = "cat_rc_real_0112_"

            self.cpu_cluster1 = 'cluster-korea-1'
            self.cpu_cluster2 = 'korea-cluster-2'

            self.dir_col_names = [name for name in os.listdir(self.save_base_dir) \
                                    if os.path.isdir(os.path.join(self.save_base_dir, name))]

        self.dir_to_num_dict = {}
        self.num_to_dir_dict = {}
        for i, dir_col_name in enumerate(self.dir_col_names):
            self.dir_to_num_dict[dir_col_name] = i
            self.num_to_dir_dict[i] = dir_col_name

        self.hist_wind_size = 30
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]

        self.ws = None
        self.dstore = None

        # upload dataset
        # self.ws = Workspace.from_config()
        # self.dstore = self.ws.get_default_datastore()
        self.ws = None;
        self.dstore = None;
        # upload data
        # self.dstore.upload('../data/drift_detector/', 'drift_detector810', overwrite=True)

        self.threshold = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).reshape((11, 1))


        # CPU Cluster
        # compute_name = 'cpu-cluster'

        # if compute_name in self.ws.compute_targets:
        #     compute_target = self.ws.compute_targets[compute_name]
        #     if compute_target and type(compute_target) is AmlCompute:
        #         print('found compute target. just use it. ' + compute_name)
        # else:
        #     print('creating a new compute target...')
        #     provisioning_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D3_V2', min_nodes=0, max_nodes=2)

        #     # create the cluster
        #     compute_target = ComputeTarget.create(self.ws, compute_name, provisioning_config)

        #     # can poll for a minimum number of nodes and for a specific timeout.
        #     # if no min node count is provided it will use the scale settings for the cluster
        #     compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

        #     # For a more detailed view of current AmlCompute status, use get_status()
        #     print(compute_target.get_status().serialize())


    def parse_precision(self):
        """
        result:
        {'drift_type': 'DatasetBased',
        'metrics': [{'schema_version': '0.1',
            'start_date': datetime.datetime(2021, 5, 30, 0, 0),
            'end_date': datetime.datetime(2021, 5, 31, 0, 0),
            'baseline_dataset_id': 'caf05173-82b1-4ccb-bed2-90576f963468',
            'target_dataset_id': '9317ec02-ee96-44cc-96d0-66c87d8c67ab',
            'column_metrics': [{'value': [{'name': 'wasserstein_distance',
                'value': 331217.5015933104},
            {'name': 'energy_distance', 'value': 211.12220826352126},
            {'name': 'datadrift_contribution', 'value': 3000.0},
            {'name': 'baseline_min', 'value': -1360.0},
            {'name': 'baseline_max', 'value': 14580202.0},
            {'name': 'baseline_mean', 'value': 917160.4618204024},
            {'name': 'target_min', 'value': -525.0},
            {'name': 'target_max', 'value': 16660055.0},
            {'name': 'target_mean', 'value': 1218238.3268156424}]}],
        """

        # get precision dir name
        res_dir_col_names = [name for name in os.listdir(self.res_save_base_dir) \
                                if os.path.isdir(os.path.join(self.res_save_base_dir, name)) and \
                                    os.path.exists(os.path.join(self.res_save_base_dir, name, 'precision.pk')) ]
        
        FP_list = []
        for dir_col_name in res_dir_col_names:

            file_path = os.path.join(self.res_save_base_dir, dir_col_name, 'precision.pk')
            res = load_file(file_path)[0]['metrics'][1:]

            total_num = len(res)
            if total_num != 30:
                print('ddd')
            
            val_list = []
            for i in range(total_num):
                val = res[i]['dataset_metrics'][0]['value']
                val_list.append(val)

            val_arr = np.array(val_list)
            y_pred = val_arr > self.threshold 

            fp = y_pred.sum(axis=1)

            FP_list.append(fp)

        # all_precision = np.array(all_precision)

        # save_path = os.path.join(self.res_save_base_dir, 'azure_drift_precision.pk')
        # save_file(FP_list, save_path)

        return FP_list


    def parse_recall(self):
        """
            parse recall
        """

        # get precision dir name

        res_dir_col_names = [name for name in os.listdir(self.res_save_base_dir) \
                                if os.path.isdir(os.path.join(self.res_save_base_dir, name)) and \
                                    os.path.exists(os.path.join(self.res_save_base_dir, name, 'recall_on_real.pk')) ]
        
        all_recall = []
        for dir_col_name in res_dir_col_names:

            file_path = os.path.join(self.res_save_base_dir, dir_col_name, 'recall_on_real.pk')
            res = load_file(file_path)[0]['metrics'][1:11]

            total_num = len(res)
            
            val_list = []
            for i in range(total_num):
                val = res[i]['dataset_metrics'][0]['value']
                val_list.append(val)

            val_arr = np.array(val_list)
            y_pred = val_arr > self.threshold 

            tp = y_pred.sum(axis=1)
            recall = tp / total_num

            all_recall.append(recall)

        all_recall_real = np.array(all_recall)

        save_path = os.path.join(self.res_save_base_dir, 'azure_drift_recall_on_real.pk')
        save_file(all_recall_real, save_path)


        # syn
        res_dir_col_names = [name for name in os.listdir(self.res_save_base_dir) \
                                if os.path.isdir(os.path.join(self.res_save_base_dir, name)) and \
                                    os.path.exists(os.path.join(self.res_save_base_dir, name, 'recall_on_syn.pk')) ]
        
        all_recall = []
        for dir_col_name in res_dir_col_names:
    
            file_path = os.path.join(self.res_save_base_dir, dir_col_name, 'recall_on_syn.pk')
            res = load_file(file_path)[0]['metrics'][1:]

            total_num = len(res)
            
            val_list = []
            for i in range(total_num):
                val = res[i]['dataset_metrics'][0]['value']
                val_list.append(val)

            val_arr = np.array(val_list)
            y_pred = val_arr > self.threshold 

            tp = y_pred.sum(axis=1)
            recall = tp / total_num

            all_recall.append(recall)

        all_recall_syn = np.array(all_recall)

        save_path = os.path.join(self.res_save_base_dir, 'azure_drift_recall_on_syn.pk')
        save_file(all_recall_syn, save_path)

        return all_recall_real, all_recall_syn


    def  parse_result_update(self):
        """
        result:
        {'drift_type': 'DatasetBased',
        'metrics': [{'schema_version': '0.1',
            'start_date': datetime.datetime(2021, 5, 30, 0, 0),
            'end_date': datetime.datetime(2021, 5, 31, 0, 0),
            'baseline_dataset_id': 'caf05173-82b1-4ccb-bed2-90576f963468',
            'target_dataset_id': '9317ec02-ee96-44cc-96d0-66c87d8c67ab',
            'column_metrics': [{'value': [{'name': 'wasserstein_distance',
                'value': 331217.5015933104},
            {'name': 'energy_distance', 'value': 211.12220826352126},
            {'name': 'datadrift_contribution', 'value': 3000.0},
            {'name': 'baseline_min', 'value': -1360.0},
            {'name': 'baseline_max', 'value': 14580202.0},
            {'name': 'baseline_mean', 'value': 917160.4618204024},
            {'name': 'target_min', 'value': -525.0},
            {'name': 'target_max', 'value': 16660055.0},
            {'name': 'target_mean', 'value': 1218238.3268156424}]}],
        """

        # get precision dir name
        prec_dir_col_names = [name for name in os.listdir(self.res_save_base_dir) \
                                if os.path.isdir(os.path.join(self.res_save_base_dir, name)) and \
                                    os.path.exists(os.path.join(self.res_save_base_dir, name, 'precision.pk')) ]

        real_dir_col_names = [name for name in os.listdir(self.res_save_base_dir) \
                                if os.path.isdir(os.path.join(self.res_save_base_dir, name)) and \
                                    os.path.exists(os.path.join(self.res_save_base_dir, name, 'recall_on_real.pk')) ]

        syn_dir_col_names = [name for name in os.listdir(self.res_save_base_dir) \
                                if os.path.isdir(os.path.join(self.res_save_base_dir, name)) and \
                                    os.path.exists(os.path.join(self.res_save_base_dir, name, 'recall_on_syn.pk')) ]

        FP_list = []
        TP_real_list = []
        TP_syn_list = []

        for dir_col_name in prec_dir_col_names:
            if dir_col_name in real_dir_col_names and dir_col_name in syn_dir_col_names:
                

                file_path = os.path.join(self.res_save_base_dir, dir_col_name, 'precision.pk')
                res = load_file(file_path)[0]['metrics'][1:]

                total_num = len(res)
                
                val_list = []
                for i in range(total_num):
                    val = res[i]['dataset_metrics'][0]['value']
                    val_list.append(val)

                val_arr = np.array(val_list)
                y_pred = val_arr > self.threshold 

                fp = y_pred.sum(axis=1)
                FP_list.append(fp)



                ################## real #####################
   
                file_path = os.path.join(self.res_save_base_dir, dir_col_name, 'recall_on_real.pk')
                res = load_file(file_path)[0]['metrics'][1:11]

                total_num = len(res)
                
                val_list = []
                for i in range(total_num):
                    val = res[i]['dataset_metrics'][0]['value']
                    val_list.append(val)

                val_arr = np.array(val_list)
                y_pred = val_arr > self.threshold 

                tp = y_pred.sum(axis=1)
                TP_real_list.append(tp)

                ################## syn #####################
                
            
                file_path = os.path.join(self.res_save_base_dir, dir_col_name, 'recall_on_syn.pk')
                res = load_file(file_path)[0]['metrics'][1:]

                total_num = len(res)
                
                val_list = []
                for i in range(total_num):
                    val = res[i]['dataset_metrics'][0]['value']
                    val_list.append(val)

                val_arr = np.array(val_list)
                y_pred = val_arr > self.threshold 

                tp = y_pred.sum(axis=1)
                TP_syn_list.append(tp)

        if self.is_numeric:
            num_syn = 27
        else:
            num_syn = 33

        FP_arr = np.array(FP_list)
        TP_real_arr = np.array(TP_real_list)
        TP_syn_arr = np.array(TP_syn_list)

        precision_real = TP_real_arr / (TP_real_arr + FP_arr)
        precision_syn = TP_syn_arr / (TP_syn_arr + FP_arr)
        recall_real = TP_real_arr / 25.0
        recall_syn = TP_syn_arr / num_syn

        precision_real = pd.DataFrame(precision_real).mean(axis=0).to_numpy()
        precision_syn = pd.DataFrame(precision_syn).mean(axis=0).to_numpy()
        recall_real = recall_real.mean(axis=0)
        recall_syn = recall_syn.mean(axis=0)

        # print("average precision real: ", precision_real)
        # print("average precision syn: ", precision_syn)
        # print("average recall_real: ", recall_real)
        # print("average recall_syn: ", recall_syn)

        return precision_real, precision_syn, recall_real, recall_syn




    def register_all_data(self):
        for dir_name in tqdm(self.dir_names[:10]):
            file_path = os.path.join(self.DATA_DIR, dir_name)
            df_list = load_file(file_path)
            cols = df_list[0].columns.to_list()

            for col in cols:
                dir_name_col = dir_name + '_' + col

                ###########################
                # set target dataset
                ###########################
                file_path = os.path.join(self.datastore_dir, dir_name_col, 'data_for_precision.csv')
                dstore_path = self.dstore.path(file_path)
                try:
                    target = Dataset.Tabular.from_delimited_files(dstore_path, separator='\t')
                except:
                    print(file_path)
                    print("could not load dataset")
                    continue

                target_name = self.precision_prefix + str(self.dir_to_num_dict[dir_name_col])
                target = target.with_timestamp_columns('timestamp')
                target = target.register(self.ws, target_name, create_new_version=True)
                target = Dataset.get_by_name(self.ws, target_name)

                ###########################
                ## data_for_recall_real
                ###########################
                file_path = os.path.join(self.datastore_dir, dir_name_col, 'data_for_recall_real.csv')
                dstore_path = self.dstore.path(file_path)
                try:
                    target = Dataset.Tabular.from_delimited_files(dstore_path, separator='\t')
                except:
                    print(file_path)
                    print("could not load dataset")
                    return

                target_name = self.recall_real_prefix + str(self.dir_to_num_dict[dir_name_col])
                target = target.with_timestamp_columns('timestamp')
                target = target.register(self.ws, target_name, create_new_version=True)
                target = Dataset.get_by_name(self.ws, target_name)

                ###########################
                ## data_for_recall_syn
                ###########################
                file_path = os.path.join(self.datastore_dir, dir_name_col, 'data_for_recall_syn.csv')
                dstore_path = self.dstore.path(file_path)
                try:
                    target = Dataset.Tabular.from_delimited_files(dstore_path, separator='\t')
                except:
                    print(file_path)
                    print("could not load dataset")
                    return

                target_name = self.recall_syn_prefix + str(self.dir_to_num_dict[dir_name_col])
                target = target.with_timestamp_columns('timestamp')
                target = target.register(self.ws, target_name, create_new_version=True)
                target = Dataset.get_by_name(self.ws, target_name)

        
    def register_all_monitors(self):

        for dir_name in tqdm(self.dir_names[250:]):
            file_path = os.path.join(self.DATA_DIR, dir_name)
            df_list = load_file(file_path)
            cols = df_list[0].columns.to_list()

            for col in cols:
                dir_name_col = dir_name + '_' + col
                
                ###############################
                # precision test
                ###############################
                target_name = self.precision_prefix + str(self.dir_to_num_dict[dir_name_col])
                try:
                    target = Dataset.get_by_name(self.ws, target_name)
                except:
                    print(target_name)
                    print('does not exist!')
                    continue

                # set baseline dataset
                baseline = target.time_before(datetime(2021, 5, 30))

                file_path = os.path.join(self.save_base_dir, dir_name_col, 'data_for_precision.csv')
                df = pd.read_csv(file_path, sep='\t')
                feature_list = df.columns.to_list()
                feature_list.remove('timestamp')

                # creat_monitor
                monitor_name = self.precision_prefix +  str(self.dir_to_num_dict[dir_name_col])
                try:
                    monitor = DataDriftDetector.create_from_datasets(self.ws, monitor_name, baseline, target,
                                                                compute_target='westus13',         # compute target for scheduled pipeline and backfills
                                                                frequency='Day',                     # how often to analyze target data
                                                                feature_list=feature_list,                    # list of features to detect drift on
                                                                drift_threshold=0.5,                 # threshold from 0 to 1 for email alerting
                                                                latency=0,                            # SLA in hours for target data to arrive in the dataset
                                                                alert_config=None)            # email addresses to send alert
                except:
                    print("!!!!!create pre_monitor exception!!!!!")
                    print(dir_name_col)
                    monitor = DataDriftDetector.get_by_name(self.ws, monitor_name)
                    monitor.delete()
                    monitor = DataDriftDetector.create_from_datasets(self.ws, monitor_name, baseline, target,
                                                                compute_target='westus13',         # compute target for scheduled pipeline and backfills
                                                                frequency='Day',                     # how often to analyze target data
                                                                feature_list=feature_list,                    # list of features to detect drift on
                                                                drift_threshold=0.5,                 # threshold from 0 to 1 for email alerting
                                                                latency=0,                            # SLA in hours for target data to arrive in the dataset
                                                                alert_config=None)            # email addresses to send alert   


                ###############################
                # data_for_recall_real
                ###############################
                file_path = os.path.join(self.save_base_dir, dir_name_col, 'data_for_recall_real.csv')
                df = pd.read_csv(file_path, sep='\t')
                feature_list = df.columns.to_list()
                feature_list.remove('timestamp')

                target_name = self.recall_real_prefix + str(self.dir_to_num_dict[dir_name_col])
                try:
                    target = Dataset.get_by_name(self.ws, target_name)
                except:
                    print(target_name)
                    print('does not exist!')
                    continue

                # baseline dataset
                baseline = target.time_before(datetime(2021, 5, 30))
                
                # creat_monitor
                monitor_name = self.recall_real_prefix + str(self.dir_to_num_dict[dir_name_col])
                try:
                    monitor = DataDriftDetector.create_from_datasets(self.ws, monitor_name, baseline, target,
                                                                compute_target='westus14',         # compute target for scheduled pipeline and backfills
                                                                frequency='Day',                     # how often to analyze target data
                                                                feature_list=feature_list,                    # list of features to detect drift on
                                                                drift_threshold=0.5,                 # threshold from 0 to 1 for email alerting
                                                                latency=0,                            # SLA in hours for target data to arrive in the dataset
                                                                alert_config=None)            # email addresses to send alert
                except:
                    print("!!!!!create rc_real_monitor exception!!!!!")
                    print(dir_name_col)
                    monitor = DataDriftDetector.get_by_name(self.ws, monitor_name)
                    monitor.delete()
                    monitor = DataDriftDetector.create_from_datasets(self.ws, monitor_name, baseline, target,
                                                                compute_target='westus14',         # compute target for scheduled pipeline and backfills
                                                                frequency='Day',                     # how often to analyze target data
                                                                feature_list=feature_list,                    # list of features to detect drift on
                                                                drift_threshold=0.5,                 # threshold from 0 to 1 for email alerting
                                                                latency=0,                            # SLA in hours for target data to arrive in the dataset
                                                                alert_config=None)            # email addresses to send alert    


                ##########################
                # data_for_recall_syn
                ###########################
                file_path = os.path.join(self.save_base_dir, dir_name_col, 'data_for_recall_syn.csv')
                df = pd.read_csv(file_path, sep='\t')
                feature_list = df.columns.to_list()
                feature_list.remove('timestamp')

                target_name = self.recall_syn_prefix + str(self.dir_to_num_dict[dir_name_col])
                try:
                    target = Dataset.get_by_name(self.ws, target_name)
                except:
                    print(target_name)
                    print('does not exist!')
                    continue

                # baseline dataset
                baseline = target.time_before(datetime(2021, 5, 30))

                # creat_monitor
                monitor_name = self.recall_syn_prefix + str(self.dir_to_num_dict[dir_name_col])
                try:
                    monitor = DataDriftDetector.create_from_datasets(self.ws, monitor_name, baseline, target,
                                                            compute_target='westus15',         # compute target for scheduled pipeline and backfills
                                                            frequency='Day',                     # how often to analyze target data
                                                            feature_list=feature_list,                    # list of features to detect drift on
                                                            drift_threshold=0.5,                 # threshold from 0 to 1 for email alerting
                                                            latency=0,                            # SLA in hours for target data to arrive in the dataset
                                                            alert_config=None)            # email addresses to send alert
                except:
                    monitor = DataDriftDetector.get_by_name(self.ws, monitor_name)
                    monitor.delete()
                    monitor = DataDriftDetector.create_from_datasets(self.ws, monitor_name, baseline, target,
                                                                compute_target='westus15',         # compute target for scheduled pipeline and backfills
                                                                frequency='Day',                     # how often to analyze target data
                                                                feature_list=feature_list,                    # list of features to detect drift on
                                                                drift_threshold=0.5,                 # threshold from 0 to 1 for email alerting
                                                                latency=0,                            # SLA in hours for target data to arrive in the dataset
                                                                alert_config=None)            # email addresses to send alert


    def test_precision(self, dir_name=None):

        file_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(file_path)
        cols = df_list[0].columns.to_list()
        
        for col in cols:
            dir_name_col = dir_name + '_' + col

            # get monitor
            monitor_name = self.precision_prefix +  str(self.dir_to_num_dict[dir_name_col])
            try:
                monitor = DataDriftDetector.get_by_name(self.ws, monitor_name)
            except:
                print(monitor_name)
                print('does not exist! get monitor failed!')
                continue

            # backfill
            backfill_start_date = datetime(2021, 5, 30)
            backfill_end_date = datetime(2021, 6, 29)
            backfill = monitor.backfill(backfill_start_date, backfill_end_date)

            # wait for computing
            status = backfill.wait_for_completion(wait_post_processing=True)

            # get result
            start_time = datetime(year=2021, month=5, day=30)
            end_time = datetime(year=2021, month=6, day=29)
            results, metrics = monitor.get_output(start_time, end_time)

            save_path = os.path.join(self.res_save_base_dir, dir_name_col, 'precision.pk')
            save_file(metrics, save_path)


    def test_recall_on_real(self, dir_name=None):
    
        file_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(file_path)
        cols = df_list[0].columns.to_list()
        
        for col in cols:
            dir_name_col = dir_name + '_' + col

            # get monitor
            monitor_name = self.recall_real_prefix + str(self.dir_to_num_dict[dir_name_col])
            try:
                monitor = DataDriftDetector.get_by_name(self.ws, monitor_name)
            except:
                print(monitor_name)
                print('does not exist! get monitor failed!')
                continue

            # backfill
            backfill_start_date = datetime(2021, 5, 30)
            backfill_end_date = datetime(2021, 6, 24)               #num
            backfill = monitor.backfill(backfill_start_date, backfill_end_date)

            # wait for computing
            status = backfill.wait_for_completion(wait_post_processing=True)

            # predict
            start_time = datetime(year=2021, month=5, day=30)
            end_time = datetime(year=2021, month=6, day=24)
            results, metrics = monitor.get_output(start_time, end_time)

            save_path = os.path.join(self.res_save_base_dir, dir_name_col, 'recall_on_real.pk')
            save_file(metrics, save_path)


    def test_recall_on_syn(self, dir_name=None):
        
        file_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(file_path)
        cols = df_list[0].columns.to_list()
        
        for col in cols:
            dir_name_col = dir_name + '_' + col
            
            # get monitor
            monitor_name = self.recall_syn_prefix + str(self.dir_to_num_dict[dir_name_col])
            try:
                monitor = DataDriftDetector.get_by_name(self.ws, monitor_name)
            except:
                print(monitor_name)
                print('does not exist! get monitor failed!')
                continue

            # backfill
            backfill_start_date = datetime(2021, 5, 30)
            # backfill_end_date = datetime(2021, 6, 26)             #num
            backfill_end_date = datetime(2021, 6, 16)                #cat
            backfill = monitor.backfill(backfill_start_date, backfill_end_date)

            # wait for computing
            status = backfill.wait_for_completion(wait_post_processing=True)

            backfill_start_date = datetime(2021, 6, 16)
            backfill_end_date = datetime(2021, 7, 2)                #cat
            backfill = monitor.backfill(backfill_start_date, backfill_end_date)

            # wait for computing
            status = backfill.wait_for_completion(wait_post_processing=True)


            # predict
            start_time = datetime(year=2021, month=5, day=30)
            # end_time = datetime(year=2021, month=6, day=26)    #num
            end_time = datetime(year=2021, month=7, day=2)      #cat
            results, metrics = monitor.get_output(start_time, end_time)

            save_path = os.path.join(self.res_save_base_dir, dir_name_col, 'recall_on_syn.pk')
            save_file(metrics, save_path)
    

if __name__ == '__main__':

    ##########################################
    # Data Generation For Azure Drift Detector
    ##########################################

    # generator = CSVDataGenerator(True)
    # generator.generate_new(generator.dir_names[0])

    # pool = Pool(31)
    # for dir_name in generator.dir_names:
    #     pool.apply_async(generator.generate_new, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # generator = CSVDataGenerator(False)
    # generator.generate_new(generator.dir_names[0])

    # pool = Pool(31)
    # for dir_name in generator.dir_names:
    #     pool.apply_async(generator.generate_new, args=(dir_name, ))
    # pool.close()
    # pool.join()


    ##################################################
    # Register all data from blodstorage to workspace
    ##################################################
    # azure_drift_detector = AzureDriftDetector(True)
    # azure_drift_detector.register_all_data()

    # azure_drift_detector = AzureDriftDetector(False)
    # azure_drift_detector.register_all_data()


    ##################################################
    # Register all monitors
    ##################################################
    # azure_drift_detector = AzureDriftDetector(True)
    # azure_drift_detector.register_all_monitors()

    # azure_drift_detector = AzureDriftDetector(False)
    # azure_drift_detector.register_all_monitors()


    ##########################################
    # Test
    ##########################################

    # num_dirs = 50

    # # numeric
    azure_drift_detector = AzureDriftDetector(True)
    # azure_drift_detector.test_precision(azure_drift_detector.dir_names[0])
    # azure_drift_detector.test_recall_on_real(azure_drift_detector.dir_names[0])
    # azure_drift_detector.test_recall_on_syn(azure_drift_detector.dir_names[0])

    azure_drift_detector.parse_result_update()

    # pool = Pool(4)
    # for dir_name in azure_drift_detector.dir_names[50:100]:
    #     pool.apply_async(azure_drift_detector.test_precision, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # for dir_name in azure_drift_detector.dir_names[0:50]:
    #     azure_drift_detector.test_precision(dir_name)


    # pool = Pool(4)
    # for dir_name in azure_drift_detector.dir_names[50:100]:
    #     pool.apply_async(azure_drift_detector.test_recall_on_real, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # for dir_name in azure_drift_detector.dir_names[0:50]:
    #     azure_drift_detector.test_recall_on_real(dir_name)



    # pool = Pool(10)
    # for dir_name in azure_drift_detector.dir_names[150:200]:
    #     pool.apply_async(azure_drift_detector.test_recall_on_syn, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # for dir_name in azure_drift_detector.dir_names[0:50]:
    #     azure_drift_detector.test_recall_on_syn(dir_name)


    ###################################################


    azure_drift_detector = AzureDriftDetector(False)
    # azure_drift_detector.test_precision(azure_drift_detector.dir_names[0])
    # azure_drift_detector.test_recall_on_real(azure_drift_detector.dir_names[0])
    # azure_drift_detector.test_recall_on_syn(azure_drift_detector.dir_names[0])

    azure_drift_detector.parse_result_update()


    # pool = Pool(2)
    # for dir_name in azure_drift_detector.dir_names[0:50]:
    #     pool.apply_async(azure_drift_detector.test_precision, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # for dir_name in azure_drift_detector.dir_names[0:50]:
    #     azure_drift_detector.test_precision(dir_name)



    # pool = Pool(2)
    # for dir_name in azure_drift_detector.dir_names[0:50]:
    #     pool.apply_async(azure_drift_detector.test_recall_on_real, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # for dir_name in azure_drift_detector.dir_names[100:150]:
    #     azure_drift_detector.test_recall_on_real(dir_name)


    # pool = Pool(2)
    # for dir_name in azure_drift_detector.dir_names[200:]:
    #     pool.apply_async(azure_drift_detector.test_recall_on_syn, args=(dir_name, ))
    # pool.close()
    # pool.join()

    # for dir_name in azure_drift_detector.dir_names[200:]:
    #     azure_drift_detector.test_recall_on_syn(dir_name)