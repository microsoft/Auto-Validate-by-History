import numpy as np
from numpy.core.fromnumeric import sort
import os
import matplotlib.pyplot as plt
from numpy.ma.core import count
import pandas as pd
from azure_drift_detector import save_as_csv
from utils import *
from dists import *
from preprocessing import *
import string
import random
from tqdm import tqdm


class GenSample:
    def __init__(self, is_numeric): 

        self.is_numeric = is_numeric

        if self.is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.save_base_dir = '../data/synthesis_0907/numeric/'
            self.sim_col_list = load_file('../result/sim_col_list_num.pk')
        else:
            self.DATA_DIR = '../data/categorical data'
            self.save_base_dir = '../data/synthesis_0907/category/'
            self.sim_col_list = load_file('../result/sim_col_list_cat.pk')
        
        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]

        self.pert_day = 29
        

    # rule 1-2
    def change_schema(self, sample, sim_sample, p):
        """
        change schema
        p: 0.01 0.1 1
        """

        num_value = len(sample)
        sample_copy = sample.copy()
        
        if int(num_value * p) <= len(sim_sample):
            idx_org = np.random.choice(range(num_value), 
                                int(num_value * p), 
                                replace=False)
            idx_sim = np.random.choice(range(len(sim_sample)), 
                                int(num_value * p), 
                                replace=False)                    
            sample_copy[idx_org] = sim_sample[idx_sim]
        else:
            idx_org = np.random.choice(range(num_value), 
                                int(num_value * p), 
                                replace=False)
            idx_sim = np.random.choice(range(len(sim_sample)), 
                                int(num_value * p), 
                                replace=True)                    
            sample_copy[idx_org] = sim_sample[idx_sim]
            
        return sample_copy.to_numpy()
        

    # rule 3
    def change_unit(self, sample, p, scale):
        """
        change sample unit
        scale: x10 x100 x1000
        p: 0.1, 1.0
        """

        num_value = len(sample)
        sample_copy = sample.copy()
        sample_idx = np.random.choice(range(num_value), 
                                      int(num_value * p), 
                                      replace=False)

        sample_copy[sample_idx] = sample[sample_idx] * scale
        
        return sample_copy.to_numpy()


    # rule 4    Casing change (str-only, x6 setups)
    def swapcase(self, sample, p, direction):
        """
        swapcase with ratio p
        p: 0.01, 0.1, 1
        """
        
        assert (p <= 1 and p >= 0)
        assert (direction in ['lower2upper', 'upper2lower'])

        num_value = len(sample)
        sample_copy = sample.copy()
        idx = np.random.choice(range(num_value), int(num_value * p), replace=False)

        if direction == 'lower2upper':
            sample_copy[idx] = sample_copy[idx].str.upper()
        else:
            sample_copy[idx] = sample_copy[idx].str.lower()
        
        return sample_copy.to_numpy()


    # rule 5
    def increase_null(self, sample, p):
        """
        increase null by p
        p: 0.01, 0.1, 1
        """
        
        assert (p <= 1 and p >= 0)

        num_value = len(sample)

        if self.is_numeric:
            null_arr = np.full(int(num_value * p), 0)
            null_series = pd.Series(null_arr)
            sample_copy = pd.concat((sample, null_series))
        else:
            null_arr = np.full(int(num_value * p), 'NULL')
            null_series = pd.Series(null_arr)
            sample_copy = pd.concat((sample, null_series))
        
        return sample_copy.to_numpy()

    # rule 6
    def upper_sample(self, sample, p):
        """
        upper sample with times p
        p: 2, 10
        """
        
        assert( p >= 1 )

        sample_copy = sample.copy()
        for i in range(int(p-1)):
            sample_copy = pd.concat((sample_copy, sample))
        
        return sample_copy.to_numpy()
        
        
    def down_sample(self, sample, p):
        """
        down sample with ratio p
        p: 0.1, 0.5
        """
        
        assert (p <= 1 and p >= 0)
        num_value = len(sample)
        return sample.sample(int(num_value * p)).to_numpy()

    # rule 7
    def gen_biased_sample(self, sample, p, is_first=True):
        """
        generate biased samples
        p: 0.1, 0.5
        """
        sample_copy = sample.copy()
        sample_copy = sample_copy.dropna()

        num_value = len(sample_copy)
        col_name = sample_copy.name
        
        df = pd.DataFrame(sample_copy)
        sample_sort = df.assign(freq=df.apply(lambda x: df[col_name].value_counts()\
                                .to_dict()[x[col_name]], axis=1))\
                                .sort_values(by=['freq', col_name], ascending=[False, True])\
                                .loc[:,[col_name]]
        
        if is_first:
            sample_sort = sample_sort[0: int(num_value * p)]
        else:
            sample_sort = sample_sort[-int(num_value * p):]
             
        return sample_sort.to_numpy().flatten()
 
    # rule 8
    def pert_num_value(self, value, p):
        
        if not pd.isna(value):
            value = list(str(value))
            num_char = len(value)
            num_pert = int(num_char * p)
            if num_pert >= 1:
                idx = np.random.choice(range(num_char), num_pert, replace=False)
                for i in idx:
                    if value[i] in string.digits:
                        if i==0:
                            value[i] = random.sample('123456789', 1)[0]
                        else:
                            value[i] = random.sample(string.digits, 1)[0]
            value = ''.join(value)
            
            if 'e' in value:
                value = eval(value)
            else:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value) 

        return value


    def pert_cat_value(self, value, p):

        if not pd.isna(value):
            value = list(value)
            num_char = len(value)
            num_pert = int(num_char * p)
            if num_pert >= 1:
                idx = np.random.choice(range(num_char), num_pert, replace=False)
                for i in idx:
                    if value[i].isdigit():
                        value[i] = random.sample(string.digits, 1)[0]
                    else:
                        value[i] = random.sample(string.ascii_letters, 1)[0]
            value = ''.join(value)
        return value


    def pert_sample(self, sample, p):
        """
        perturbe sample
        p: 0.01  0.1  1
        """   

        if self.is_numeric:
            sample_pert = sample.apply(self.pert_num_value, args=(p,))
        else:
            sample_pert = sample.apply(self.pert_cat_value, args=(p,))
                
        return sample_pert.to_numpy()



    def insert_char(self, value, p):
        # numeric
        if self.is_numeric:
            if not pd.isna(value):
                value = list(str(value))
                num_digit = len(value)
                threshold = p * 100
                num_insertion = 0
                
                if 'e' not in value:
                    # for negative number
                    if value[0] == '-':
                        for i in range(1, num_digit+1):
                            possibility = random.randint(1, 100)
                            if possibility < threshold:
                                if i==1:
                                    random_char = random.sample('123456789', 1)[0] # the 1st number cannot be zero
                                    value.insert(i + num_insertion, random_char)
                                    num_insertion += 1
                                else:
                                    random_char = random.sample(string.digits, 1)[0]
                                    value.insert(i + num_insertion, random_char)
                                    num_insertion += 1

                    # for positive number
                    else:
                        if '.' in value:
                            pass

                        else:
                            pass
                        for i in range(num_digit+1):
                            possibility = random.randint(1, 100)
                            if possibility < threshold:
                                if i==0:
                                    random_char = random.sample('123456789', 1)[0]  # the 1st number cannot be zero
                                    value.insert(i + num_insertion, random_char)
                                    num_insertion += 1
                                else:
                                    random_char = random.sample(string.digits, 1)[0]
                                    value.insert(i + num_insertion, random_char)
                                    num_insertion += 1
                # '5.03956071e1-057'
                elif 'e' in value and '.' in value:
                    e_idx = value.index('e')
                    point_idx = value.index('.')
                    insert_idx = range(point_idx+2, e_idx-1)
                    if len(insert_idx) >= 1:
                        for i in insert_idx:
                            possibility = random.randint(1, 100)
                            if possibility < threshold:
                                random_char = random.sample(string.digits, 1)[0]
                                value.insert(i + num_insertion, random_char)
                                num_insertion += 1
                elif 'e' in value and '.' not in value:
                    e_idx = value.index('e')
                    insert_idx = range(e_idx-1)
                    if len(insert_idx) >= 1:
                        for i in insert_idx:
                            possibility = random.randint(1, 100)
                            if possibility < threshold:
                                random_char = random.sample(string.digits, 1)[0]
                                value.insert(i + num_insertion, random_char)
                                num_insertion += 1

                value = ''.join(value)
                
                # convert to float or int
                if 'e' in value:
                    value = eval(value)
                else:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
        # category
        else:
            punctuation = list(string.punctuation)
            punctuation.remove(',')
            punctuation = ''.join(punctuation)

            all_char = string.ascii_letters + string.digits + punctuation

            value = list(value)
            num_char = len(value)

            threshold = p * 100
            num_insertion = 0
            for i in range(num_char+1):
                possibility = random.randint(1, 100)
                if possibility < threshold:
                    random_char = random.sample(all_char, 1)[0]
                    value.insert(i + num_insertion, random_char)
                    num_insertion += 1
            
            value = ''.join(value)

        return value


    def del_char(self, value, p):
        """delet a char"""
        if not pd.isna(value):
            value = list(value)
            num_char = len(value)

            threshold = p * 100
            num_deletion = 0
            for i in range(num_char):
                possibility = random.randint(1, 100)
                if possibility < threshold:
                    value.pop(i - num_deletion)
                    num_deletion += 1

            value = ''.join(value)
        return value


    def del_digit(self, value, p):
        """delet a digit"""
        if not pd.isna(value):
            value = list(str(value))
            num_digit = len(value)
            value_cp = value.copy()
            threshold = p * 100
            num_deletion = 0

            if 'e' in value and '.' in value:
                e_idx = value.index('e')
                point_idx = value.index('.')
                del_idx = range(point_idx+2, e_idx-1)
                if len(del_idx) >= 1:
                    for i in del_idx:
                        if value[i- num_deletion] != '.' and value[i- num_deletion] != '-' and i !=0: # possible char in numerical value
                            possibility = random.randint(1, 100) # deletion
                            if possibility < threshold:
                                if i != 0:
                                    value.pop(i - num_deletion)
                                    num_deletion += 1         # element index will be changed after deletion

            elif 'e' in value and '.' not in value:
                e_idx = value.index('e')
                del_idx = range(e_idx-2)
                if len(del_idx) >= 1:
                    for i in del_idx:
                        if value[i- num_deletion] != '.' and value[i- num_deletion] != '-' and i !=0: # possible char in numerical value
                            possibility = random.randint(1, 100) # deletion
                            if possibility < threshold:
                                if i != 0:
                                    value.pop(i - num_deletion)
                                    num_deletion += 1         # element index will be changed after deletion
            else:
                for i in range(num_digit):
                    if value[i- num_deletion] != '.' or value[i- num_deletion] != '-': # possible char in numerical value
                        possibility = random.randint(1, 100) # deletion
                        if possibility < threshold:
                            if i != 0:
                                value.pop(i - num_deletion)
                                num_deletion += 1         # element index will be changed after deletion

            value = ''.join(value)
            # '5.503956071e-5'    # parse 
            if value == '':
                value = np.nan
            elif value in ['-.', '.', '-']:
                value = 0
            else:
                # convert to float or int
                if 'e' in value:
                    value = eval(value)
                else:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)

        return value


    def char_insertion(self, sample, p):
        """rule 9"""
        sample_copy = sample.copy()
        sample_pert = sample_copy.apply(self.insert_char, args=(p,))

        return sample_pert.to_numpy()


    def char_deletion(self, sample, p):
        """rule 10"""
        sample_copy = sample.copy()
        
        if self.is_numeric:
            sample_pert = sample_copy.apply(self.del_digit, args=(p,))
        else:
            sample_pert = sample_copy.apply(self.del_char, args=(p,))

        return sample_pert.to_numpy()


    def padding(self, sample, p, is_leading):
        """rule 10"""

        pert_sample = sample.copy()

        count = len(sample)
        pert_count = int(count * p)
        pert_idx = np.random.choice(range(count), pert_count, replace=False)
        if is_leading:
            for i in pert_idx:
                pert_sample[i] = ' ' + sample[i]
        else:
            for i in pert_idx:
                pert_sample[i] = sample[i] + ' '

        return pert_sample.to_numpy()


    def gen_save_sample(self, dir_name):
        # numeric 
        if self.is_numeric:

            dir_path = os.path.join(self.DATA_DIR, dir_name)
            df_list = load_file(dir_path)
            cols = df_list[0].columns.to_list()
            num_col = len(cols)
            assert(num_col >= 2)

            all_pert_samples = []

            for i, col in enumerate(cols):
                pert_samples = []
                sample = df_list[self.pert_day][col]   # Default: the 29th day used for recall sample synthesis

                # rule 1-2  Partial schema-change  (x3 setups)
                if i < (num_col-1):    # get the closest column of the same type
                    close_col = cols[i+1]
                else:
                    close_col = cols[i-1]
                close_sample = df_list[self.pert_day][close_col]
                pert_samples.append(self.change_schema(sample, close_sample, 0.1))
                pert_samples.append(self.change_schema(sample, close_sample, 0.5))
                pert_samples.append(self.change_schema(sample, close_sample, 1.0))

                # rule 3    Unit change, numeric-only   (x6 setups)
                pert_samples.append(self.change_unit(sample, 0.1, 10))
                pert_samples.append(self.change_unit(sample, 0.1, 100))
                pert_samples.append(self.change_unit(sample, 0.1, 1000))
                pert_samples.append(self.change_unit(sample, 1.0, 10))
                pert_samples.append(self.change_unit(sample, 1.0, 100))
                pert_samples.append(self.change_unit(sample, 1.0, 1000))

                # rule 5    Increased NULL/0 (x3 setups)
                pert_samples.append(self.increase_null(sample, 0.1))
                pert_samples.append(self.increase_null(sample, 0.5))
                pert_samples.append(self.increase_null(sample, 1.0))

                # rule 6    Down/Up-sample to change row count (x4 setups)
                pert_samples.append(self.upper_sample(sample, 2))
                pert_samples.append(self.upper_sample(sample, 10))
                pert_samples.append(self.down_sample(sample, 0.5))
                pert_samples.append(self.down_sample(sample, 0.1))

                # rule 7    Biased sample (x4 setups)    Sort values, then pick the first/last p% as tests (p=10%, 50%)
                pert_samples.append(self.gen_biased_sample(sample, 0.1, is_first=True))
                pert_samples.append(self.gen_biased_sample(sample, 0.5, is_first=True))
                pert_samples.append(self.gen_biased_sample(sample, 0.1, is_first=False))
                pert_samples.append(self.gen_biased_sample(sample, 0.5, is_first=False))

                # rule 8    Char/Digit perturbation (x3 setups)
                pert_samples.append(self.pert_sample(sample, 0.1))
                pert_samples.append(self.pert_sample(sample, 0.5))
                pert_samples.append(self.pert_sample(sample, 1.0))

                # rule 9    Char/Digit insertion (x2 setups)
                pert_samples.append(self.char_insertion(sample, 0.1))
                pert_samples.append(self.char_insertion(sample, 0.5))

                # rule 10   Char/Digit deletion (x2 setups)
                pert_samples.append(self.char_deletion(sample, 0.1))
                pert_samples.append(self.char_deletion(sample, 0.5))

                all_pert_samples.append(pert_samples)

            save_path = os.path.join(self.save_base_dir, dir_name)
            save_file(np.array(all_pert_samples), save_path)


        # category
        else:
            dir_path = os.path.join(self.DATA_DIR, dir_name)
            df_list = load_file(dir_path)
            cols = df_list[0].columns.to_list()
            num_col = len(cols)
            assert(num_col >= 2)
        
            all_pert_samples = []
            for i, col in enumerate(cols):
                pert_samples = []
                sample = df_list[self.pert_day][col]   #  the 29th day used for recall sample synthesis
                sample = sample.astype(str)            #  for True/False

                # rule 1-2  Partial schema-change  (x3 setups)
                if i < (num_col-1):                    # get the closest column of the same type
                    close_col = cols[i+1]
                else:
                    close_col = cols[i-1]
                close_sample = df_list[self.pert_day][close_col]
                pert_samples.append(self.change_schema(sample, close_sample, 0.1))
                pert_samples.append(self.change_schema(sample, close_sample, 0.5))
                pert_samples.append(self.change_schema(sample, close_sample, 1.0))

                # rule 4 Casing change, str-only (x6 setups)
                pert_samples.append(self.swapcase(sample, 0.1, 'lower2upper'))
                pert_samples.append(self.swapcase(sample, 0.5, 'lower2upper'))
                pert_samples.append(self.swapcase(sample, 1.0, 'lower2upper'))
                pert_samples.append(self.swapcase(sample, 0.1, 'upper2lower'))
                pert_samples.append(self.swapcase(sample, 0.5, 'upper2lower'))
                pert_samples.append(self.swapcase(sample, 1.0, 'upper2lower'))

                # rule 5    Increased NULL/0 (x3 setups)
                pert_samples.append(self.increase_null(sample, 0.1))
                pert_samples.append(self.increase_null(sample, 0.5))
                pert_samples.append(self.increase_null(sample, 1.0))

                # rule 6    Down/Up-sample to change row count (x4 setups)
                pert_samples.append(self.upper_sample(sample, 2))
                pert_samples.append(self.upper_sample(sample, 10))
                pert_samples.append(self.down_sample(sample, 0.5))
                pert_samples.append(self.down_sample(sample, 0.1))

                # rule 7    Biased sample (x4 setups)    Sort values, then pick the first/last p% as tests (p=10%, 50%)
                pert_samples.append(self.gen_biased_sample(sample, 0.1, is_first=True))
                pert_samples.append(self.gen_biased_sample(sample, 0.5, is_first=True))
                pert_samples.append(self.gen_biased_sample(sample, 0.1, is_first=False))
                pert_samples.append(self.gen_biased_sample(sample, 0.5, is_first=False))

                # rule 8    Char perturbation (x3 setups)
                pert_samples.append(self.pert_sample(sample, 0.1))
                pert_samples.append(self.pert_sample(sample, 0.5))
                pert_samples.append(self.pert_sample(sample, 1.0))

                # rule 9    Char insertion  (x2 setups)
                pert_samples.append(self.char_insertion(sample, 0.1))
                pert_samples.append(self.char_insertion(sample, 0.5))

                # rule 10   Char/Digit deletion (x2 setups)
                pert_samples.append(self.char_deletion(sample, 0.1))
                pert_samples.append(self.char_deletion(sample, 0.5))

                # rule 10  Padding (x6 setups)
                pert_samples.append(self.padding(sample, 0.1, is_leading=True))
                pert_samples.append(self.padding(sample, 0.5, is_leading=True))
                pert_samples.append(self.padding(sample, 1.0, is_leading=True))
                pert_samples.append(self.padding(sample, 0.1, is_leading=False))
                pert_samples.append(self.padding(sample, 0.5, is_leading=False))
                pert_samples.append(self.padding(sample, 1.0, is_leading=False))

                all_pert_samples.append(pert_samples)

            save_path = os.path.join(self.save_base_dir, dir_name)
            save_file(np.array(all_pert_samples), save_path)


class GenSamForAzureDriftDetection():

    def __init__(self, is_numeric):
        self.is_numeric = is_numeric

        # baseline data： 2021-5-1  ~  2021-5-30
        self.date_time = [x.strftime('%Y-%m-%d') + ' ' + \
                            x.strftime('%H:%M:%S') for x  \
                            in list(pd.date_range(start='2021-05-01', \
                            end='2021-06-29'))]

            
        if is_numeric:
            self.DATA_DIR = '../data/numerical data'
            self.PERT_DATA_DIR = "../data/synthesis_0819/numeric"
            self.sim_col_list = load_file('../result/sim_col_list_num.pk')
            self.save_base_dir = '../data/drift_detector0831/numeric'

        else:
            self.DATA_DIR = '../data/categorical data'
            self.PERT_DATA_DIR = "../data/synthesis_0819/category"
            self.sim_col_list = load_file('../result/sim_col_list_cat.pk')
            self.save_base_dir = '../data/drift_detector0831/category'


        self.hist_wind_size = 30
        self.pred_wind_size = 30
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

        self.dir_names = [name for name in os.listdir(self.DATA_DIR) \
                          if os.path.isfile(os.path.join(self.DATA_DIR, name))]


    def reSample(self, sample, count):
        sample = pd.Series(sample.flatten())
        if len(sample) > count:
            # down sampling
            sample = sample.sample(count, replace=False).to_numpy()

        elif len(sample) < count:
            # up sampling
            diff_num = count - len(sample)
            diff_sample = sample.sample(diff_num, replace=True)
            sample = pd.concat((sample, diff_sample)).to_numpy()
            
        return sample


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


    def reGenSample(self, dir_name):
        file_path = os.path.join(self.DATA_DIR, dir_name)
        df_list = load_file(file_path)
        cols = df_list[0].columns.to_list()

        pert_file_path = os.path.join(self.PERT_DATA_DIR, dir_name)
        pert_samples = load_file(pert_file_path)

        for j, col in enumerate(cols):

            dir_col_name = dir_name + '_' + col

            # precision
            for i, day_value in enumerate(df_list):
                if i==0:
                    hist_value = df_list[i]
                    count = day_value.shape[0]
                    hist_data_time = [self.date_time[i]] * count
                else:

                    hist_value = pd.concat((hist_value, day_value))
                    count = day_value.shape[0]
                    hist_data_time.extend([self.date_time[i]] * count)

            hist_value['datetime'] = hist_data_time
            
            save_path = os.path.join(self.save_base_dir, dir_col_name, 'data_for_precision.csv')
            save_as_csv(hist_value, save_path)


             # prepare data for recall test on synthetic data
            if self.is_numeric:
                df_list_copy = df_list[:50].copy()
            else:
                df_list_copy = df_list[:60].copy()

            for i, day_value in enumerate(df_list_copy):
                if i==0:
                    hist_value = df_list[i]
                    count = day_value.shape[0]
                    hist_data_time = [self.date_time[i]] * count
                    
                elif i>0 and i<30:
                    hist_value = pd.concat((hist_value, day_value))
                    count = day_value.shape[0]
                    hist_data_time.extend([self.date_time[i]] * count)

                else:
                    day_value = df_list_copy[30]  #######################################
                    count = day_value.shape[0]
                    pert_sample = pert_samples[j][i-30]
                    re_pert_samples = self.reSample(pert_sample, count)
                    day_value[col] = re_pert_samples
                    hist_value = pd.concat((hist_value, day_value))
                    
                    hist_data_time.extend([self.date_time[i]] * count)


            hist_value['datetime'] = hist_data_time
            save_path = os.path.join(self.save_base_dir, dir_col_name, 'data_for_recall_syn.csv')
            save_as_csv(hist_value, save_path)


            # prepare data for recall test on real data    50% real 
            sim_col_list = self.get_similar_cols(dir_name, col, is_train=False) # 25 similar columns

            for i, day_value in enumerate(df_list[:55]):
                if i==0:
                    hist_value = df_list[i]

                    count = day_value.shape[0]
                    hist_data_time = [self.date_time[i]] * count
                    
                elif i>0 and i<30:
                    hist_value = pd.concat((hist_value, day_value))

                    count = day_value.shape[0]
                    hist_data_time.extend([self.date_time[i]] * count)

                else:
                    day_value = df_list[30]   #######################
                    count = day_value.shape[0]
                    hist_data_time.extend([self.date_time[i]] * count)
                    # load similar data
                    item = sim_col_list[i-30]
                    sim_dir = list(item.keys())[0]
                    sim_col = item[sim_dir]
                    sim_dir_path = os.path.join(self.DATA_DIR, sim_dir)
                    sim_df_list = load_file(sim_dir_path)

                    sim_sample = sim_df_list[self.hist_wind_size][sim_col].to_numpy()
                    re_sim_samples = self.reSample(sim_sample, count)
                    day_value[col] = re_sim_samples
                    hist_value = pd.concat((hist_value, day_value))

            hist_value['datetime'] = hist_data_time
            save_path = os.path.join(self.save_base_dir, dir_col_name, 'data_for_recall_real.csv')
            save_as_csv(hist_value, save_path)




if __name__ == '__main__':

    ################################
    # Synthetic Data for all methods
    ################################

    # numeric
    gen_sample = GenSample(is_numeric=True)
    gen_sample.gen_save_sample(gen_sample.dir_names[0])

    # pool = Pool(10)
    # for dir_name in gen_sample.dir_names:
    #     pool.apply_async(gen_sample.gen_save_sample, args=(dir_name, ))
    # pool.close()
    # pool.join()

    
    # category
    # gen_sample = GenSample(is_numeric=False)
    # gen_sample.gen_save_sample(gen_sample.dir_names[0])

    # pool = Pool(15)
    # for dir_name in gen_sample.dir_names:
    #     pool.apply_async(gen_sample.gen_save_sample, args=(dir_name, ))
    # pool.close()
    # pool.join()



    ###########################################
    # Synthetic Data for Azure Drift Detection
    ###########################################

    # gen = GenSamForAzureDriftDetection(is_numeric=True)
    # # gen.reGenSample(gen.dir_names[0])

    # pool = Pool(25)
    # for dir_name in gen.dir_names:
    #     pool.apply_async(gen.reGenSample, args=(dir_name,  ))
    # pool.close()
    # pool.join()


    # gen = GenSamForAzureDriftDetection(is_numeric=False)
    # # gen.reGenSample(gen.dir_names[0])

    # pool = Pool(30)
    # for dir_name in gen.dir_names:
    #     pool.apply_async(gen.reGenSample, args=(dir_name,  ))
    # pool.close()
    # pool.join()
    







