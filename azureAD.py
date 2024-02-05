# To start sending requests to the Anomaly Detector API, paste your Anomaly Detector resource access key below,
# and replace the endpoint variable with the endpoint for your region or your on-premise container endpoint. 
# Endpoint examples:
# https://westus2.api.cognitive.microsoft.com/anomalydetector/v1.0/timeseries/last/detect
# http://127.0.0.1:5000/anomalydetector/v1.0/timeseries/last/detect
apikey = '17e1163275034d31a3efd9c766ba2389'
endpoint_latest = 'https://dv-test.cognitiveservices.azure.com/anomalydetector/v1.0/timeseries/last/detect'



import requests
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import library to display results
import matplotlib.pyplot as plt

from multiprocessing import Pool
import os
from utils import *
from tqdm import tqdm

from bokeh.plotting import figure,output_notebook, show
from bokeh.palettes import Blues4
from bokeh.models import ColumnDataSource,Slider

import datetime
from bokeh.io import push_notebook
from dateutil import parser
from ipywidgets import interact, widgets, fixed
output_notebook()


def save_as_json(data, res_path):
    dir_path = os.path.dirname(res_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    with open(res_path, 'w') as file:
        json.dump(data, file)


def detect(endpoint, apikey, request_data):
    headers = {'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key': apikey}
    response = requests.post(endpoint, data=json.dumps(request_data), headers=headers)
    if response.status_code == 200:
        return json.loads(response.content.decode("utf-8"))
    else:
        print(response.status_code)
        raise Exception(response.text)


class JsonDataGenerator:

    def __init__(self, is_numeric, verbose=False):

        self.is_numeric = is_numeric
        self.verbose = verbose

        self.date_list = [x.strftime('%Y-%m-%d') + 'T' + \
                          x.strftime('%H:%M:%S')+'Z' for x  \
                          in list(pd.date_range(start='2020-05-01', \
                          end='2020-06-29'))]

        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.DATA_POOL = '../data/numeric_pool'    # real similar columns 
            self.sim_col_list = load_file('../result/sim_col_list_num_09.pk')
            self.save_base_dir = '../data/azureAD0106/numeric'

        else:
            self.DATA_DIR = '../data/categorical data'
            self.DATA_POOL = '../data/category_pool'
            self.sim_col_list = load_file('../result/sim_col_list_cat_09.pk')
            self.save_base_dir = '../data/azureAD0106/category'

        self.hist_wind_size = 30
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]


    def get_sim_col_count(self, dir_name):

        for sim_col_dict in self.sim_col_list:
            if dir_name == sim_col_dict['target_dir']:
                sim_col_list = sim_col_dict['similar_col'][1::2]      # 25 similar columns
                count_list = []
                for item in sim_col_list:
                    # load similar data
                    sim_dir = list(item.keys())[0]
                    sim_dir_path = os.path.join(self.DATA_POOL, sim_dir)
                    sim_df_list = load_file(sim_dir_path)
                    count_list.append(sim_df_list[self.hist_wind_size].shape[0])
                return count_list


    def generate_data_for_precision(self):
        """
        generate json data for precision test
        """

        for dir_name in self.dir_names:
            file_path = os.path.join(self.DATA_DIR, dir_name)
            df_list = load_file(file_path)
            series = []
            for i in range(len(df_list)):
                value = {'timestamp': self.date_list[i], 'value': df_list[i].shape[0]}
                series.append(value)
                
            json_dict = {'granularity':"daily", 'series':series}
            save_path = os.path.join(self.save_base_dir, 'precision', dir_name)
            save_as_json(json_dict, save_path)


    def generate_data_for_reall(self):
        for dir_name in tqdm(self.dir_names):
            count_list = self.get_sim_col_count(dir_name)

            file_path = os.path.join(self.DATA_DIR, dir_name)
            df_list = load_file(file_path)

            # history 30-day data
            series = []
            for i in range(self.hist_wind_size):
                value = {'timestamp': self.date_list[i], 
                         'value': df_list[i].shape[0]}
                series.append(value)

            # No.31 recall day
            try:
                json_dict_list = []
                for count_val in count_list:

                    series_tmp = series.copy()
                    series_tmp.append(
                                    {'timestamp': self.date_list[self.hist_wind_size], 
                                    'value': count_val})
                    json_dict = {'granularity':"daily", 'series':series_tmp}
                    json_dict_list.append(json_dict)

                save_path = os.path.join(self.save_base_dir, 'recall', dir_name)
                save_file(json_dict_list, save_path)
            except:
                print(dir_name)



class AzureAD:
    def __init__(self, is_numeric, verbose=False):
    
        self.is_numeric = is_numeric
        self.verbose = verbose

        if is_numeric:
            self.DATA_DIR = '../data/azureAD0106/numeric'
            self.save_base_dir = '../result/azureAD0106/numeric'

        else:
            self.DATA_DIR = '../data/azureAD0106/category'
            self.save_base_dir = '../result/azureAD0106/category'

        self.hist_wind_size = 30
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

        # self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
        #                   if os.path.isfile(os.path.join(self.DATA_DIR, name))]


    def test_precision(self):
        dir_path = os.path.join(self.DATA_DIR, 'precision')
        dir_names = [name for name in os.listdir(dir_path) \
                          if os.path.isfile(os.path.join(dir_path, name))]

        for dir_name in tqdm(dir_names):
            precison_list = []
            sample_data = json.load(open(dir_path + '/' + dir_name))
            
            points = sample_data['series']
            skip_point = 30
            
            for sensitivity in [0, 20, 40, 60, 80, 100]:
                anom_count = 0
                for i in range(skip_point, len(points)+1):
                    single_sample_data = {}
                    single_sample_data['series'] = points[i-skip_point:i]
                    single_sample_data['granularity'] = 'daily'
                    single_sample_data['maxAnomalyRatio'] = 0.25
                    single_sample_data['sensitivity'] = sensitivity
                    single_point = detect(endpoint_latest, apikey, single_sample_data)
                    if single_point['isAnomaly'] == True:
                        anom_count = anom_count + 1
                precision = 1 - anom_count/skip_point
                precison_list.append(precision)
            
            save_file(np.array(precison_list), self.save_base_dir + '/precision/'+ dir_name)



    def test_recall(self):
        dir_path = os.path.join(self.DATA_DIR, 'recall')
        dir_names = [name for name in os.listdir(dir_path) \
                    if os.path.isfile(os.path.join(dir_path, name))]

        for dir_name in tqdm(dir_names):
            recall_list = []

            samples = load_file(dir_path + '/' + dir_name)
            total_num = len(samples)
            for sensitivity in [0, 20, 40, 60, 80, 100]:
                anom_count = 0
                
                for sample_data in samples:
            
                    points = sample_data['series']
                    single_sample_data = {}
                    single_sample_data['series'] = points
                    single_sample_data['granularity'] = 'daily'
                    single_sample_data['maxAnomalyRatio'] = 0.25
                    single_sample_data['sensitivity'] = sensitivity
                    single_point = detect(endpoint_latest, apikey, single_sample_data)

                    if single_point['isAnomaly'] == True:
                        anom_count = anom_count + 1
                
                recall = anom_count/total_num
                recall_list.append(recall)
            
            save_file(np.array(recall_list), self.save_base_dir + '/recall/'+ dir_name)


    def parse_result_update(self):
        
        ############################
        # precision
        ############################
        data_dir = os.path.join(self.save_base_dir, 'precision')

        dir_names = [name for name in os.listdir(data_dir) \
                    if os.path.isfile(os.path.join(data_dir, name))]

        azure_precision = load_file(os.path.join(data_dir, dir_names[0]))
        for dir_name in dir_names[1:]:
            precision_tmp = load_file(os.path.join(data_dir, dir_name))
            azure_precision = np.vstack((azure_precision, precision_tmp))

        ############################
        # recall
        ############################
        data_dir = os.path.join(self.save_base_dir, 'recall')

        dir_names = [name for name in os.listdir(data_dir) \
                    if os.path.isfile(os.path.join(data_dir, name))]

        azure_recall = load_file(os.path.join(data_dir, dir_names[0]))

        for dir_name in dir_names[1:]:
            recall_tmp = load_file(os.path.join(data_dir, dir_name))
            azure_recall = np.vstack((azure_recall, recall_tmp))    
 

        TP = azure_recall * 25
        FP = (1 - azure_precision) * 30


        precision = TP / (TP + FP)
        precision = pd.DataFrame(precision).mean(axis=0).to_list()

        recall = pd.DataFrame(azure_recall).mean(axis=0).to_list()

        # print("average precision:" , precision )
        # print("average recall:" , recall)

        return precision, recall


if __name__ == '__main__':

    ####################################################
    # generate JSON data for Azure Anomlay Detector test
    ####################################################

    generator = JsonDataGenerator(True)
    generator.generate_data_for_precision()
    generator.generate_data_for_reall()

    generator = JsonDataGenerator(False)
    generator.generate_data_for_precision()
    generator.generate_data_for_reall()
    
    
    ####################################################
    # numeircal test
    ####################################################
    azure_ad = AzureAD(True)
    azure_ad.test_precision()
    azure_ad.test_recall()
    azure_ad.parse_result_update()
    
    # ####################################################
    # # categorical test
    # ####################################################
    
    # azure_ad = AzureAD(False)
    # azure_ad.test_precision()
    # azure_ad.test_recall()
    # azure_ad.parse_result_update()



    


