import numpy as np
from numpy.lib.arraysetops import unique
from numpy.lib.function_base import append
from utils import *
from multiprocessing import Pool

class Deequ:
    
    def __init__(self):

        self.NUM_DATA_DIR = '../data/numerical data'
        self.NUM_DATA_POOL = '../data/numeric_pool'
        self.CAT_DATA_DIR = '../data/categorical data'
        self.CAT_DATA_POOL = '../data/category_pool'

        self.rule_base_dir = '../result/deequ/rules/'

        self.num_dir_names = [name for name in os.listdir(self.NUM_DATA_DIR) \
                              if os.path.isfile(os.path.join(self.NUM_DATA_DIR, name))]

        self.cat_dir_names = [name for name in os.listdir(self.CAT_DATA_DIR) \
                              if os.path.isfile(os.path.join(self.CAT_DATA_DIR, name))]

        self.hist_wind_size = 29      # 0-28 days for training(generating rules), 29th day for recall data synthesis
        self.pred_wind_size = 30      # 30-59 day for testing
        self.total_wind_size = self.hist_wind_size + self.pred_wind_size

        self.num_sim_col_list = load_file('../result/sim_col_list_num_09.pk')
        self.cat_sim_col_list = load_file('../result/sim_col_list_cat_09.pk')

        self.pert_day = 29

        self.threshold = np.arange(0, 1.05, 0.05)

    def is_complete(self, sample):
        """
        the sample is complete(no NaN) or not
        sample: one-array sample
        return: True/False
        """

        if sample.isna().sum() == 0:
            return True
        else:
            return False


    def is_integer(self, sample):
        is_int_list = []

        for val in sample:
            if (val % 1 == 0):
                is_int_list.append(True)
            else:
                is_int_list.append(False)

        if all(is_int_list)==True:
            return True
        else:
            return False


    def is_unique_num(self, sample):
        """
        the sample is unique or not
        sample: one-array sample
        return: True/False
        """
        # check float type; 
        is_int = self.is_integer(sample)

        if is_int:
            if len(np.unique(sample)) == len(sample):
                return True                             # is unique, integer
            else:
                return False                            # is not unique, integer
        else:
            return False                                # is float


    def is_unique_cat(self, sample):
        """
        the sample is unique or not
        sample: one-array sample
        return: True/False
        """

        if len(np.unique(sample)) == len(sample):
            return True                             # is unique
        else:
            return False                            # is not unique


    def is_non_negative(self, sample):
        """the sample value >= 0"""

        if (sample < 0).sum() == 0:
            return True
        else:
            return False


    def is_unique_value_less_ratio_num(self, sample, ratio):
        """ 
        the # of unique Value / total # of rows < 0.1 
        which means most values are the same
        return: if not unique(unique val ratio < 0.1)
                    return True
                else the column is unique
                    return False
        """

        # check float type; 
        is_int = self.is_integer(sample)

        if is_int:
            unique_sample = np.unique(sample)

            if len(unique_sample) / len(sample) <= ratio:
                # most vals are the same
                return True
            else:
                return False

        else:
            return False


    def is_unique_value_less_ratio_cat(self, sample, ratio):
        """ 
        the # of unique Value / total # of rows < 0.1 
        which means most values are the same
        return: if not unique(unique val ratio < 0.1)
                    return True
                else the column is unique
                    return False
        """
        # ratio = ratio           # vary  preci-recall             all is integer
        unique_sample = np.unique(sample)

        if len(unique_sample) / len(sample) <= ratio:
            # most vals are the same
            return True
        else:
            return False


    def generate_rules(self):
        """generate rules based on 29-day training data set"""

        # numeric

        for ratio in self.threshold:
            all_rule_dict = {}
            for dir_name in self.num_dir_names:

                dir_rule_dict = {}
                dir_path = os.path.join(self.NUM_DATA_DIR, dir_name)
                df_list = load_file(dir_path)
                cols = df_list[0].columns.to_list()

                df_list.pop(29)

                for col in cols:
                    #  is_complete, is_unique, is_non_negative
                    numeric_rule = [False, False, False, False]
                    is_complete = []
                    is_unique = []
                    is_unique_value_less_ratio = []
                    is_non_negative = []

                    unique_val_set = set()
                    for i in range(self.hist_wind_size):
                        is_complete.append(self.is_complete(df_list[i][col]))
                        is_unique.append(self.is_unique_num(df_list[i][col]))  # True: unique integer

                        unique_val_state = self.is_unique_value_less_ratio_num(df_list[i][col], ratio)  # most elements are the same
                        if unique_val_state:
                            unique_val_set = unique_val_set.union(set(np.unique(df_list[i][col])))
                        is_unique_value_less_ratio.append(unique_val_state)

                        is_non_negative.append(self.is_non_negative(df_list[i][col]))

                    
                    if all(is_complete)==True:
                        numeric_rule[0] = True
                    
                    if all(is_unique)==True:
                        numeric_rule[1] = True
    
                    if all(is_unique_value_less_ratio)==True:
                        numeric_rule[2] = unique_val_set

                    if all(is_non_negative)==True:
                        numeric_rule[3] = True
        
                    dir_rule_dict.update({col: numeric_rule})

                all_rule_dict.update({dir_name: dir_rule_dict})
            
            file_path = os.path.join(self.rule_base_dir, 'deequ_num_rule_dict_{:.2f}.pk'.format(ratio))
            save_file(all_rule_dict, file_path)


            # category
            all_rule_dict = {}
            for dir_name in self.cat_dir_names:
        
                dir_rule_dict = {}
                dir_path = os.path.join(self.CAT_DATA_DIR, dir_name)
                df_list = load_file(dir_path)
                cols = df_list[0].columns.to_list()

                df_list.pop(29)

                for col in cols:
                    #  is_complete, is_unique, uniqueValueRatio
                    category_rule = [False, False, False]

                    is_complete = []
                    is_unique = []
                    is_unique_value_less_ratio = []
                    unique_val_set = set()
                    for i in range(self.hist_wind_size):
                        col_vals = df_list[i][col].astype(str)   # boolean type
                        is_complete.append(self.is_complete(col_vals))
                        is_unique.append(self.is_unique_cat(col_vals))
                        unique_val_state = self.is_unique_value_less_ratio_cat(col_vals, ratio)
                        is_unique_value_less_ratio.append(unique_val_state)
                        if unique_val_state:
                            unique_val_set = unique_val_set.union(set(np.unique(col_vals))) 
                        
                    
                    if all(is_complete)==True:
                        category_rule[0] = True
                    
                    if all(is_unique)==True:
                        category_rule[1] = True
                    
                    if all(is_unique_value_less_ratio)==True:
                        category_rule[2] = unique_val_set
                        
                    dir_rule_dict.update({col: category_rule})

                all_rule_dict.update({dir_name: dir_rule_dict})
            
            file_path = os.path.join(self.rule_base_dir, 'deequ_cat_rule_dict_{:.2f}.pk'.format(ratio))
            save_file(all_rule_dict, file_path)


    def pred_result(self, rule_dict, dir_name, col, sample, dtype='numeric'):

        if dtype == 'numeric':

            rule = rule_dict[dir_name][col]
            y_pred = 0

            # is_complete
            if rule[0]:
                if not self.is_complete(sample):
                    y_pred = 1

            # is_unique integer
            if rule[1]:
                if not self.is_unique_num(sample):
                    y_pred = 1

            # is_unique_value_less_ratio
            if rule[2] != False:
                unique_val = np.unique(sample)
                if np.sum([True for val in unique_val if val in rule[2]]) == 0:
                    # print(unique_val)
                    # print('sum: ', np.sum([True for val in unique_val if val in rule[2]]))
                    y_pred = 1
 
            # is_non_negative
            if rule[3]:
                if not self.is_non_negative(sample):
                    y_pred = 1

        else:
            rule = rule_dict[dir_name][col]
            y_pred = 0

            # is_complete
            if rule[0]:
                if not self.is_complete(sample):
                    y_pred = 1

            # is_unique
            if rule[1]:
                if not self.is_unique_cat(sample):
                    y_pred = 1

            # is_unique_value_less_ratio
            if rule[2] != False:
                unique_val = np.unique(sample)
                if np.sum([True for val in unique_val if val in rule[2]]) == 0:
                    # print(unique_val)
                    # print('sum: ', np.sum([True for val in unique_val if val in rule[2]]))
                    y_pred = 1

        return y_pred



    def test_deequ_num(self, ratio):
        """
        test precision
        """
        # numeric precision

        FP_list = []
        TP_real_list = []
        TP_syn_list = []

        file_path = os.path.join(self.rule_base_dir, 'deequ_num_rule_dict_{:.2f}.pk'.format(ratio))
        rule_dict = load_file(file_path)

        for dir_name in self.num_dir_names:
    
            dir_path = os.path.join(self.NUM_DATA_DIR, dir_name)
            df_list = load_file(dir_path)
            cols = df_list[0].columns.to_list()
            df_list.pop(self.pert_day)   # delete 29th day

            # real
            sim_col_dicts = [i for i in self.num_sim_col_list if i['target_dir'] == dir_name]

            # synthetic
            test_generated_sample_p = load_file('../data/synthesis_0907/numeric_test/' + dir_name)

            for j, col in enumerate(cols):

                y_pred = []
                for i in range(self.pred_wind_size):
                    sample = df_list[i+self.hist_wind_size][col]
                    y_pred_tmp = self.pred_result(rule_dict, dir_name, col, sample, dtype='numeric')
                    y_pred.append(y_pred_tmp)

                y_pred = np.array(y_pred)
                fp = y_pred.sum(axis=0)
                FP_list.append(fp)

                ################ real ###############
                sim_col_dict = sim_col_dicts[j]
                target_col = sim_col_dict['target_col']
                assert(target_col == col)

                similar_col_list = sim_col_dict['similar_col'][1::2]   # 50% x 50 days

                y_pred = []
                for i, item in enumerate(similar_col_list):          # 50 the most similar data
                    sim_dir = list(item.keys())[0]      
                    sim_col = item[sim_dir]
                    sim_dir_path = os.path.join(self.NUM_DATA_POOL, sim_dir)
                    sim_df_list = load_file(sim_dir_path)

                    sample = sim_df_list[self.pred_wind_size][sim_col]
                    y_pred_tmp= self.pred_result(rule_dict, dir_name, target_col, sample, dtype='numeric')
                    y_pred.append(y_pred_tmp)

                tp = np.array(y_pred).sum(axis=0)  # (50, 20, 16)  ==>   (20, 16)
                TP_real_list.append(tp)

                ################ syn ################
                y_pred = []
                num_syn_sample = test_generated_sample_p[j].shape[0]
                for i in range(num_syn_sample):
                    sample = pd.Series(test_generated_sample_p[j][i]) 
                    y_pred_tmp = self.pred_result(rule_dict, dir_name, col, sample, dtype='numeric')
                    y_pred.append(y_pred_tmp)
            
                y_pred = np.array(y_pred)

                tp = y_pred.sum(axis=0)
                TP_syn_list.append(tp)


        save_file((FP_list, TP_real_list, TP_syn_list), '../result/deequ/ratio{:.2f}/numeric.pk'.format(ratio))




    def test_deequ_cat(self, ratio):

        file_path = os.path.join(self.rule_base_dir, 'deequ_cat_rule_dict_{:.2f}.pk'.format(ratio))
        rule_dict = load_file(file_path)

        FP_list = []
        TP_real_list = []
        TP_syn_list = []

        for dir_name in self.cat_dir_names:

            dir_path = os.path.join(self.CAT_DATA_DIR, dir_name)
            df_list = load_file(dir_path)
            cols = df_list[0].columns.to_list()

            # real
            sim_col_dicts = [i for i in self.cat_sim_col_list if i['target_dir'] == dir_name]

            # syn
            test_generated_sample_p = load_file('../data/synthesis_0907/category_test/' + dir_name)

            for j, col in enumerate(cols):
                y_pred = []
                for i in range(self.pred_wind_size):
                    sample = df_list[i+self.hist_wind_size][col].astype(str)
                    y_pred_tmp = self.pred_result(rule_dict, dir_name, col, sample, dtype='category')
                    y_pred.append(y_pred_tmp)

                y_pred = np.array(y_pred)
                fp = y_pred.sum(axis=0)
                FP_list.append(fp)


                ############## real ############
                sim_col_dict = sim_col_dicts[j]
                target_col = sim_col_dict['target_col']
                assert(target_col == col)
                similar_col_list = sim_col_dict['similar_col'][1::2]   # 50% x 50 days

                y_pred = []
                for i, item in enumerate(similar_col_list):            # 25 the most similar columns

                    sim_dir = list(item.keys())[0]      
                    sim_col = item[sim_dir]
                    sim_dir_path = os.path.join(self.CAT_DATA_POOL, sim_dir)
                    sim_df_list = load_file(sim_dir_path)

                    sample = sim_df_list[self.pred_wind_size][sim_col].astype(str)
                    y_pred_tmp = self.pred_result(rule_dict, dir_name, target_col, sample, dtype='category')
                    y_pred.append(y_pred_tmp)

                tp = np.array(y_pred).sum(axis=0)  # (50, 20, 16)  ==>   (20, 16)
                TP_real_list.append(tp)

                ############## syn  ############
                y_pred = []
                num_syn_sample = test_generated_sample_p[j].shape[0]
                for i in range(num_syn_sample):
                    sample = pd.Series(test_generated_sample_p[j][i]).astype(str)
                    y_pred_tmp = self.pred_result(rule_dict, dir_name, col, sample, dtype='category')
                    y_pred.append(y_pred_tmp)

                y_pred = np.array(y_pred)
                tp = y_pred.sum(axis=0)
                TP_syn_list.append(tp)

        save_file((FP_list, TP_real_list, TP_syn_list), '../result/deequ/ratio{:.2f}/category.pk'.format(ratio))





    def parse_result_update(self, is_numeric):

        precision_real_list = []
        precision_syn_list = []
        recall_real_list = []
        recall_syn_list = []

        if is_numeric:
            num_syn = 27.0
        else:
            num_syn = 33.0

        for ratio in self.threshold:
            if is_numeric:
                FP_list, TP_real_list,TP_syn_list  = load_file('../result/deequ/ratio{:.2f}/numeric.pk'.format(ratio))
            else:
                FP_list, TP_real_list, TP_syn_list = load_file('../result/deequ/ratio{:.2f}/category.pk'.format(ratio))


            FP_arr = np.array(FP_list)
            TP_real_arr = np.array(TP_real_list)
            TP_syn_arr = np.array(TP_syn_list)

            precision_real = pd.Series(TP_real_arr / (TP_real_arr + FP_arr)).mean()
            precision_syn = pd.Series(TP_syn_arr / (TP_syn_arr + FP_arr)).mean()
            recall_real = pd.Series(TP_real_arr / 25.0).mean()
            recall_syn = pd.Series(TP_syn_arr / num_syn).mean()

            precision_real_list.append(precision_real)
            precision_syn_list.append(precision_syn)
            recall_real_list.append(recall_real)
            recall_syn_list.append(recall_syn)

        return precision_real_list, precision_syn_list, recall_real_list, recall_syn_list



if __name__ == '__main__':

    deequ = Deequ()
    deequ.generate_rules()

    # for ratio in deequ.threshold:
    #     deequ.test_deequ_num(ratio)
    #     deequ.test_deequ_cat(ratio)

    pool = Pool(48)
    for ratio in deequ.threshold:
        pool.apply_async(deequ.test_deequ_num, args=(ratio, ))
        pool.apply_async(deequ.test_deequ_cat, args=(ratio, ))
    pool.close()
    pool.join()

    # precision_real, precision_syn, recall_real, recall_syn = deequ.parse_result_update(is_numeric=True)
    # precision_real, precision_syn, recall_real, recall_syn = deequ.parse_result_update(is_numeric=False)