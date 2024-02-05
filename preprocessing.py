import os
import pandas as pd
import pickle
import random
from tqdm import tqdm
import numpy as np
import shutil
from multiprocessing import Pool
import time
import warnings
warnings.filterwarnings('ignore')
from utils import *


class Preprocess:
    """
    Automated Data Validation
    """

    def __init__(self):

        self.RAW_DATA_DIR = '../data/raw data'
        self.DATA_DIR = '../data/csv_data'
        self.NUM_DATA_DIR = '../data/numerical pool new'
        self.CAT_DATA_DIR = '../data/categorical data'
        self.INVALID_DIR = '../data/invalid folder'

        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)

        if not os.path.exists(self.NUM_DATA_DIR):
            os.makedirs(self.NUM_DATA_DIR)
    
        if not os.path.exists(self.CAT_DATA_DIR):
            os.mkdir(self.CAT_DATA_DIR)

        if not os.path.exists('../data/invalid folder'):
            os.makedirs('../data/invalid folder')

        self.raw_dir_names = [name for name in os.listdir(self.RAW_DATA_DIR) \
                                if os.path.isdir(os.path.join(self.RAW_DATA_DIR, name))]

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isdir(os.path.join(self.DATA_DIR, name))]

        self.cat_dir_names = [name for name in os.listdir(self.CAT_DATA_DIR) \
                                if os.path.isfile(os.path.join(self.CAT_DATA_DIR, name))]

        self.num_dir_names = [name for name in os.listdir(self.NUM_DATA_DIR) \
                                if os.path.isfile(os.path.join(self.NUM_DATA_DIR, name))]

        # load file_dict.pk if exists
        if os.path.exists('../result/file_dict.pk'):
            self.file_dict = load_file('../result/file_dict.pk')
        else:
            print("Lack file_dict.pk")
            self.file_dict = {}

        # load stats_dict.pk if exists
        if os.path.exists('../result/stats_dict.pk'):
            self.stats_dict = load_file('../result/stats_dict.pk')
        else:
            print("Lack stats_dict.pk")
            self.stats_dict = {}


    def load_file_dict(self, dir_name):
        """
        build file dictionary
        output: dir_names: [file0, file1, file2] 
        """

        raw_dir = os.path.join(self.RAW_DATA_DIR, dir_name)
        file_names = [name for name in os.listdir(raw_dir) \
                      if os.path.isfile(os.path.join(raw_dir, name))]

        #######################################
        # Need > 60 days
        #######################################
        file_dict = {}
        if len(file_names) < 60:    

            # move to invalid folder
            dst = os.path.join('invalid folder', 'folders less than 60 days', dir_name)
            if not os.path.exists(dst):
                move_dir(dir_name, 'folders less than 60 days')
            else:
                print("Dis disr already exists: ", dir_name)
            
            return file_dict

        #######################################
        # Select continuous 60 files that have more than 50 records
        #######################################
        file_dict = {}
        num_valid_file = 0
        valid_file_names = []

        for file in file_names:
            file_path = os.path.join(self.RAW_DATA_DIR, dir_name, file)
            
            try:
                df = pd.read_csv(file_path, sep='\t', header=None)
                row_num = df.shape[0]
                
                if row_num >= 50:
                    num_valid_file += 1
                    valid_file_names.append(file)

                    if num_valid_file >= 60:
                        file_dict[dir_name] = valid_file_names
                        break

                else:
                    num_valid_file = 0
                    valid_file_names = []

            except:                                      # one file with different number of columns
                num_valid_file = 0
                valid_file_names = []
                print("Loading {} failed".format(file_path))

        #############################################
        # One folder with different number of columns
        #############################################
        
        if len(valid_file_names) == 60:
            num_cols = set()
            for file in file_dict[dir_name]:
                file_path = os.path.join(self.RAW_DATA_DIR, dir_name, file)
                df = pd.read_csv(file_path, sep='\t', header=None)
                num_cols.add(df.shape[1])

            if len(num_cols) > 1:
                print(dir_name)
                print('inconsistant number of columns')
                file_dict = {}

        return file_dict


    def copy_to_data_dir(self, dir_name):
        """
        copy 60 days' data from 'raw data 'to 'data'
        """

        dir_path = os.path.join(self.DATA_DIR, dir_name)

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        for file in self.file_dict[dir_name]:
            src = os.path.join(self.RAW_DATA_DIR, dir_name, file)
            dst = os.path.join(self.DATA_DIR, dir_name, file)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
            else:
                print("Already exist")



    def save(self, dir_name):
        """
        save 60 days' data to categorical data or numerical data
        """
        
        data_dir = os.path.join(self.DATA_DIR, dir_name)
        file_names = [name for name in os.listdir(data_dir) \
                      if os.path.isfile(os.path.join(data_dir, name))]

        df = self.load_one_txt(os.path.join(data_dir, file_names[0]))

        cat_cols = []
        num_cols = []

        def separate_num_cat(col):
    
            # remove NaN columns
            nan_count = col.isna().sum()
            nan_rate = nan_count / len(col)
            if nan_rate > 0.1: 
                # print('-------Removed NaN columns-------')
                # print(dir_name)
                # print(col.name)
                # print(col)
                return

            # numerical columns, exclude boolean columns
            if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
                num_cols.append(col.name)

            # categorical columns
            else:
                char_seq = col.map((lambda x:len(str(x)) > 200))
                char_rate = char_seq.sum() / len(col)
                if char_rate < 0.1:
                    cat_cols.append(col.name)
                # else:
                #     print('-------Removed > 200 Characters-------')
                #     print(dir_name)
                #     print(col.name)
                #     print(col)

        # separate_num_cat and select cols
        df.apply(separate_num_cat)

        # iterate 60-day data, and save it to disk
        cat_df_list = []
        num_df_list = []
        
        for file in file_names:
            file_path = os.path.join(self.DATA_DIR, dir_name, file)
            df = self.load_one_txt(file_path)

            if df is None:
                print("Loading {} failed".format(file_path))

            else:
                # category
                if len(cat_cols) > 0:
                    cat_df = df[cat_cols]
                    cat_df_list.append(cat_df)
                # numeric
                if len(num_cols) > 0:
                    num_df = df[num_cols]
                    num_df_list.append(num_df)

        if len(cat_df_list) > 0:
            save_path = os.path.join(self.CAT_DATA_DIR, dir_name)
            save_file(cat_df_list, save_path)

        if len(num_df_list) > 0:
            save_path = os.path.join(self.NUM_DATA_DIR, dir_name)
            save_file(num_df_list, save_path)


    """load raw txt data from specific file path"""
    def load_one_txt(self, file_path):

        try:
            df = pd.read_csv(file_path, 
                             sep='\t', 
                             header=None, 
                             prefix='C', 
                             skiprows=[0],
                             low_memory=False)

            df = df.dropna(axis=1, how='all')

        except:
            print("Loading {} file failed".format(file_path))
            df = None

        return df


    def load_one_col(self, file_path, col='C0'):
        """load one col from specific file path"""
        
        col = col.replace('C', '')
        col = int(col)

        try:
            df = pd.read_csv(file_path, 
                             sep='\t', 
                             header=None, 
                            #  prefix='C', 
                             usecols=[col],
                             skiprows=[0],
                             error_bad_lines=False)

            df = df.dropna(axis=1, how='all')

            if df.shape[0] == 0:
                df = None

        except:
            print(col)
            print("Loading {} file failed".format(file_path))
            df = None

        return df


    def estimate_dir_stats(self, dir_name):
        """
        for numerical data
        generate mean and std for a signle dir
        dir_name: txt files directory
        num: # of txt files
        return: average mean, sigma 
        """
        data_path = os.path.join(self.NUM_DATA_DIR, dir_name)
        df_list = load_file(data_path)

        stats_dict = {}
        for i, df in enumerate(df_list):
            if i==0:                            # first mean, sigma
                mu = pd.DataFrame(df.mean()).T
                sigma = pd.DataFrame(df.std()).T

            else:                                       # second ++ mean, sigma
                mu_temp = pd.DataFrame(df.mean()).T
                sigma_temp = pd.DataFrame(df.std()).T
                mu = pd.concat([mu, mu_temp])
                sigma = pd.concat([sigma, sigma_temp])

        mu = mu.mean()
        sigma = sigma.mean()

        if len(mu) > 0:
            stats_dict[dir_name] = (mu, sigma)
        else:
            stats_dict = None

        return stats_dict


if __name__ == "__main__":
    
    prep = Preprocess()
    # pool = Pool(30)

    #################################
    # load_file_dict
    #################################

    # start = time.time()

    # res_list = []
    # for dir_name in prep.raw_dir_names:
    #     res = pool.apply_async(prep.load_file_dict, 
    #                            args=(dir_name, ))     # A result from single process
    #     res_list.append(res)

    # pool.close()
    # pool.join()

    # end = time.time()
    # print ("Time: {} s".format(int(end - start)))

    # file_dict = {}
    # for res in res_list:
    #     if bool(res):
    #         file_dict.update(res.get())

    # with open('file_dict.pk', 'wb') as f:
    #     pickle.dump(file_dict, f)

    # print(len(file_dict))

    ###########################################
    # copy data from raw data to data direcotry
    ###########################################
    # for dir_name in list(prep.file_dict.keys()):
    #     pool.apply_async(prep.copy_to_data_dir, args=(dir_name, ))     # A result from single process
    # pool.close()
    # pool.join()


    ################################################
    # separate numerical and categorical data to disk
    #################################################
    # prep.save(prep.dir_names[0])
    # for dir_name in prep.dir_names:
    #     res = pool.apply_async(prep.save, args=(dir_name, ))
    # pool.close()
    # pool.join()


    #################################
    # estimate statistics of each dir
    #################################
    # res_list = []
    # dir_names = prep.num_dir_names
    # print(len(dir_names))

    # for dir_name in dir_names:
    #     res = pool.apply_async(prep.estimate_dir_stats, 
    #                            args=(dir_name, ))
    #     res_list.append(res)
    # pool.close()
    # pool.join()

    # stats_dict = {}
    # for res in res_list:
    #     if res.get() is not None:
    #         stats_dict.update(res.get())
    # print("stats_dict length: ", len(stats_dict))
    # save_file(stats_dict, '../result/stats_dict_09.pk')


    #################################
    # filter 1000 columns
    #################################

    size_list = []
    for dir_name in prep.cat_dir_names:
        dir_path = os.path.join('../data/categorical data/' + dir_name)
        size_list.append(os.path.getsize(dir_path) / float(1024*1024))

    # for i, val in enumerate(np.array(size_list) > 20):
    #     if val==True:
    #         dir_name = prep.cat_dir_names[i]
    #         dir_path = os.path.join('../data/categorical data/' + dir_name)
    #         os.remove(dir_path)