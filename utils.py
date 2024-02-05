from sys import prefix
from numpy.lib.arraysetops import unique
import pandas as pd
import os
import pickle
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
# from autodv import *
from scipy import interpolate
from dists import *

warnings.filterwarnings('ignore')
p_list = [0.05, 0.1, 0.15, 0.2, 0.25, 
                       0.3, 0.35, 0.4, 0.45, 0.5]

dist_class = ['EMD', 'JS_div', 'KL_div', 'KS_dist', 'Cohen_dist', 
              'Min', 'Max', 'Mean', 'Median', 'Count', 
              'Sum', 'Range', 'Skew', '2-moment', '3-moment', 'Majority']


def load_file(file_path):
    "load data"

    assert(os.path.exists(file_path) == 1)

    with open(file_path, 'rb') as f:
        res = pickle.load(f)

    return res


def save_file(res, res_path):
    dir_path = os.path.dirname(res_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    with open(res_path, 'wb') as f:
        pickle.dump(res, f)


def move_dir(folder, reason):
    """
    src: source folder name
    dst: reason
    """
    
    src = os.path.join('raw data', folder)
    dst = os.path.join('invalid folder', reason, folder)
    shutil.move(src, dst)


def display_distribution(sample_p, sample_q, i, 
                         dir_name_p, dir_name_q, 
                         reason, res, 
                         save_path='../result/distribution/', 
                         dtype='numeric',
                         best_metric='None'):

    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    if dtype=='numeric':

        sns.distplot(sample_p, ax=ax, hist=False,
                     kde_kws={'lw': 2,'label':'Curr Day'})

        sns.distplot(sample_q, ax=ax, hist=False, rug=True,
                     kde_kws={"lw": 2,'label': 'Prev Day'})

        plt.title(dir_name_p + '\n'+ 
                  dir_name_q + '\n' + 
                  'Fail Day: No.' + str(i) + ' day' + '\n' + 
                  'Resaon: ' + ', '.join(reason) + '\n' + 
                  str(res) + '\n' + 
                  'Selected Metric: ' + '\n' +
                  str(best_metric))

    else:

        combined_frequencies_p, combined_frequencies_q, _, _, unique_combined = \
             convert_freq_for_two_samples(sample_p, sample_q)
        
        ax.bar(range(len(combined_frequencies_p)), 
                          combined_frequencies_p, label='Curr Day', alpha=0.2)

        plt.title(dir_name_p + '\n'+ 
                  dir_name_q + '\n' + 
                 'Fail Day: No.' + str(i) + ' day' + '\n' + 
                 'Resaon: ' + ', '.join(reason) + '\n' +
                 str(res) + '\n' +
                 'Selected Metric: ' + '\n' +
                 str(best_metric))

        plt.tight_layout()

        ax.bar(range(len(combined_frequencies_q)), 
                          combined_frequencies_q, label='Prev Day', alpha=0.2)
        plt.tight_layout()

        ax.set_xticklabels(unique_combined)
        plt.xticks(rotation=60)



    if sample_p.name==None:
        sample_p.name = 'C?'

    save_path = save_path + dir_name_p + sample_p.name + \
                dir_name_q + sample_q.name + ' ' + str(i) + '.png'

    plt.legend()
    plt.tight_layout()

    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
            
    plt.savefig(save_path)
    plt.close()
        


def plot_result(res, title, scale_range, 
                dist_class, save_path='../result/', 
                dtype='numeric', p_list=None):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if dtype == 'numeric':
        res = np.array(res)
        num_dist = res.shape[2]
        
        for j, scale in enumerate(scale_range):

            plt.figure(figsize=(16, 9))

            for i in range(num_dist):

                plt.subplot(num_dist, 1, i+1)
                plt.hist(res[j][:, i])
                plt.ylabel(dist_class[i], rotation=30)
    
                if i== 0:
                    title_name = title + '_scale_' + ('%2.6f'%scale)

                    if num_dist > 1:
                        plt.title('Average {}:{}'.format(title, [float('%.2f'%val) for val in res[j].mean(axis=0)]))
                    else:
                        plt.title('Average {}:{}'.format(title, float('%.2f'%res[j].mean())))

    elif dtype=='dynamic_metric_recall':

        res = np.array(res)
        num_dist = res.shape[2]

        # iterate p_list
        for i in range(res.shape[0]):   # (# of p_list, # of cols, # of dists)
            metric_recall = res[i]      # (# of cols, # of dists)
            metric_recall = metric_recall.swapaxes(0, 1)  # (# of dists, # of cols)

            plt.figure(figsize=(16, 9))
            # iterate dists
            for j in range(metric_recall.shape[0]):
                plt.subplot(num_dist, 1, j+1)
                plt.hist(metric_recall[j])
                plt.ylabel(dist_class[j], rotation=30)
                plt.xlim((0, 1))

                if j== 0:
                    plt.title('Average {}:{}'.format(title, 
                        [float('%.2f'%val) for val in metric_recall.mean(axis=1)]))
  
            plt.tight_layout()     
            plt.savefig(save_path + title + '_p_'+ str(p_list[i]) + '.png')
            plt.show()



def plot_PR_curve(precision_path, recall_path, save_path, title, dist_class, method='z_score'):

    plt.figure()
    
    if method=='z_score' or method=='max' or method=='tfdv':

        precision = load_file(precision_path)
        recall = load_file(recall_path)
        
        precision_arr = np.array(precision)  # (20, 1478, 16)
        recall_arr = np.array(recall)        # (20, 1454, 16)

        assert precision_arr.shape[0] == recall_arr.shape[0]
        assert precision_arr.shape[2] == recall_arr.shape[2]

        ave_precision = precision_arr.mean(axis=1)   # (20, 16)
        ave_recall = recall_arr.mean(axis=1)         # (20, 16)

        for i in range(ave_precision.shape[1]):
            # curve = zip(ave_recall[:, i], ave_precision[:, i])
            # curve = np.array(sorted(curve, key=lambda x:x[0]))
            # plt.plot(curve[:, 0], curve[:, 1], label=dist_class[i])

            plt.scatter(ave_recall[:, i], ave_precision[:, i], label=dist_class[i])
            plt.legend()


    plt.xlabel('recall')
    plt.ylabel('precision')

    plt.tight_layout()
    save_path = os.path.join(save_path, title)
    plt.savefig(save_path + '.png')
    plt.show()


def compare_two_methods(precision_path_z_score, recall_path_z_score, 
                        precision_path_tfdv, recall_path_tfdv, 
                        save_path, title, dist_class_z_score, dist_class_tfdv, dtype='numeric'):

    plt.figure()

    precision_arr_z_score = np.array(load_file(precision_path_z_score))
    recall_arr_z_score = np.array(load_file(recall_path_z_score))

    precision_arr_tfdv = np.array(load_file(precision_path_tfdv))
    recall_arr_tfdv = np.array(load_file(recall_path_tfdv))

    assert precision_arr_z_score.shape[0] == recall_arr_z_score.shape[0]
    assert precision_arr_z_score.shape[2] == recall_arr_z_score.shape[2]

    assert precision_arr_tfdv.shape[0] == recall_arr_tfdv.shape[0]
    assert precision_arr_tfdv.shape[2] == recall_arr_tfdv.shape[2]

    ave_precision_z_score = precision_arr_z_score.mean(axis=1)   # (20, 16)
    ave_recall_z_score = recall_arr_z_score.mean(axis=1)         # (20, 16)

    ave_precision_tfdv = precision_arr_tfdv.mean(axis=1)   # (20, 16)
    ave_recall_tfdv = recall_arr_tfdv.mean(axis=1)         # (20, 16)

    if dtype == 'numeric':
        df_z_score = pd.DataFrame({'L-1-Precision': ave_precision_z_score[:, 0], 
                                   'L-1-Recall': ave_recall_z_score[:, 0], 
                                   'Count-Precision': ave_precision_z_score[:, 9], 
                                   'Count-Recall': ave_recall_z_score[:, 9]})
        df_z_score.to_csv('z_score_numeric_precision_recall.csv', index=None)

        df_tfdv = pd.DataFrame({'JS-div-Precision': ave_precision_tfdv[:, 0], 
                                'JS-div-Recall': ave_recall_tfdv[:, 0]})
        df_tfdv.to_csv('tfdv_numeric_precision_recall.csv', index=None)
        
        for i in [0, 9]:
            curve = zip(ave_recall_z_score[:, i], ave_precision_z_score[:, i])
            curve = np.array(sorted(curve, key=lambda x:x[0]))
            func = interpolate.interp1d(curve[:, 0], curve[:, 1], kind='cubic')
            x = np.linspace(curve[:, 0].min(), curve[:, 0].max(), 100)
            y = func(x)
            plt.plot(x, y, label='Z-Score-' + dist_class_z_score[i])

            # plt.scatter(ave_recall_z_score[:, i], 
            #             ave_precision_z_score[:, i], 
            #             label='Z-Score-' + dist_class_z_score[i])
            plt.legend()


        curve = zip(ave_recall_tfdv[:, 0], ave_precision_tfdv[:, 0])
        curve = np.array(sorted(curve, key=lambda x:x[0]))
        curve = curve[3:, :]
        func = interpolate.interp1d(curve[:, 0], curve[:, 1], kind='cubic')
        x = np.linspace(curve[:, 0].min(), curve[:, 0].max(), 100)
        y = func(x)
        plt.plot(x, y, label='TFDV-' + dist_class_tfdv[0])

        # plt.scatter(ave_recall_tfdv[:, 0], 
        #             ave_precision_tfdv[:, 0], 
        #             label='TFDV-' + dist_class_tfdv[0], marker='+')
        plt.legend()

    else:
        df_z_score = pd.DataFrame({'L-1-Precision': ave_precision_z_score[:, 0], 
                                   'L-1-Recall': ave_recall_z_score[:, 0], 
                                   'Cosine-Precision': ave_precision_z_score[:, 2], 
                                   'Cosine-Recall': ave_recall_z_score[:, 2]})
        df_z_score.to_csv('z_score_category_precision_recall.csv', index=None)


        df_tfdv = pd.DataFrame({'L-inf-Precision': ave_precision_tfdv[:, 0], 
                                'L-inf-Recall': ave_recall_tfdv[:, 0]})
        df_tfdv.to_csv('tfdv_category_precision_recall.csv', index=None)

        for i in [0, 2]:

            plt.plot(ave_recall_z_score[:, i], 
                        ave_precision_z_score[:, i], 
                        label='Z-Score-' + dist_class_z_score[i])
            plt.legend()

        curve = zip(ave_recall_tfdv[:, 0], ave_precision_tfdv[:, 0])
        curve = np.array(sorted(curve, key=lambda x:x[0]))
        curve = curve[9:, :]
        func = interpolate.interp1d(curve[:, 0], curve[:, 1], kind='slinear')
        x = np.linspace(curve[:, 0].min(), curve[:, 0].max(), 100)
        y = func(x)
        plt.plot(x, y, label='TFDV-' + dist_class_tfdv[0])
        # plt.plot(ave_recall_tfdv[:, 0], 
        #             ave_precision_tfdv[:, 0], 
        #             label='TFDV-' + dist_class_tfdv[0])
        plt.legend()

    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title(title)

    plt.tight_layout()
    save_path = os.path.join(save_path, title + '_comparison')
    plt.savefig(save_path + '.png')
    plt.show()              



def plot_dynamic_result_off_line():

    dir_path = '../result/dynmic_zscore/temp'
    dir_names = [name for name in os.listdir(dir_path) \
                          if os.path.isfile(os.path.join(dir_path, name))]

    precision_arr = []
    metric_recall_arr = []
    for i, dir_name in enumerate(dir_names):
        if i==0:
            file_path = os.path.join(dir_path, dir_name)
            precision, metric_recall = load_file(file_path)   # (10, 16)
        else:
            file_path = os.path.join(dir_path, dir_name)
            precision_temp, metric_recall_temp = load_file(file_path)   # (10, 16)
            precision = np.vstack((precision, precision_temp))
            metric_recall = np.vstack((metric_recall, metric_recall_temp))


    
    # metric_recall = metric_recall.swapaxes(0, 1)   #(# p list, # of cols, # of dists)
    num_dist = metric_recall.shape[2]

    # for i in range(metric_recall.shape[0]):
    #     print(i)
    #     plt.figure(figsize=(16, 9))
    #     recall = metric_recall[i].swapaxes(0, 1)
    #     for j in range(recall.shape[0]):
    #         plt.subplot(num_dist, 1, j+1)
    #         plt.hist(recall[j])
    #         plt.ylabel(dist_class[j], rotation=30)
    #         # plt.xlim((0, 1))

    #         if j== 0:
    #             plt.title('Average {}:{}'.format('Metrix Recall', 
    #                 [float('%.2f'%val) for val in recall.mean(axis=1)]))

    #         # plt.tight_layout()     
    #         plt.savefig('../result/dynamic_z_score/dynamic_num_metric_recall/dynamic_num_metric_recall_p_'+ str(p_list[i]) + '.png')
    #         plt.show()

    precision = precision.swapaxes(0, 1)
    # num_dist = precision.shape[0]
    plt.figure(figsize=(16, 9))

    for i in range(precision.shape[0]):
        plt.subplot(num_dist, 1, i+1)
        plt.hist(precision[i])
        plt.ylabel(dist_class[i], rotation=30)
        # plt.xlim((0, 1))

        if i== 0:
            plt.title('Average {}:{}'.format('Numeric Precision', 
                [float('%.2f'%val) for val in precision.mean(axis=1)]))

    # plt.tight_layout()     
    plt.savefig('../result/dynamic_z_score/dynamic_num_precision/dynamic_num_precision_p_.png')
    plt.show()


def save_as_csv(data, res_path):
    dir_path = os.path.dirname(res_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    data.to_csv(res_path, index=False, sep='\t')
    

        # res = np.array(res)
        # num_dist = res.shape[2]

        # # iterate p_list
        # for i in range(res.shape[0]):   # (# of p_list, # of cols, # of dists)
        #     metric_recall = res[i]      # (# of cols, # of dists)
        #     metric_recall = metric_recall.swapaxes(0, 1)  # (# of dists, # of cols)

        #     plt.figure(figsize=(16, 9))
        #     # iterate dists
        #     for j in range(metric_recall.shape[0]):
        #         plt.subplot(num_dist, 1, j+1)
        #         plt.hist(metric_recall[j])
        #         plt.ylabel(dist_class[j], rotation=30)
        #         plt.xlim((0, 1))

        #         if j== 0:
        #             plt.title('Average {}:{}'.format(title, 
        #                 [float('%.2f'%val) for val in metric_recall.mean(axis=1)]))
  
        #     plt.tight_layout()     
        #     plt.savefig(save_path + title + '_p_'+ str(p_list[i]) + '.png')
        #     plt.show()

if __name__ == '__main__':
    # autodv = AutoDV()

    # plot_PR_curve('../result/z_score/numeric_precision.pk', 
    #               '../result/z_score/numeric_recall.pk', 
    #               './', 'PR-Curve-Numeric-Z_score', 
    #               autodv.num_dist_class, 
    #               method='z_score')

    # plot_PR_curve('../result/z_score/category_precision.pk', 
    #               '../result/z_score/category_recall.pk', 
    #               './', 'PR-Curve-Category-Z_score', 
    #               autodv.cat_dist_class, 
    #               method='z_score')


    # plot_PR_curve('../result/tfdv/numeric_precision2.pk', 
    #               '../result/tfdv/numeric_recall2.pk', 
    #               './', 'PR-Curve-Numeric-TFDV-2', 
    #               ['JS_div'], 
    #               method='tfdv')

    # plot_PR_curve('../result/tfdv/category_precision2.pk', 
    #               '../result/tfdv/category_recall2.pk', 
    #               './', 'PR-Curve-Category-TFDV-2', 
    #               ['l-inf'], 
    #               method='tfdv')


    # compare_two_methods('../result/z_score/numeric_precision.pk', '../result/z_score/numeric_recall.pk', 
    #                         '../result/tfdv/numeric_precision2.pk', '../result/tfdv/numeric_recall2.pk', 
    #                         './', 'Numeric Z-Score V.S. TFDV', autodv.num_dist_class, ['JS_div'], dtype='numeric')

    # compare_two_methods('../result/z_score/category_precision.pk', '../result/z_score/category_recall.pk', 
    #                         '../result/tfdv/category_precision2.pk', '../result/tfdv/category_recall2.pk', 
    #                         './', 'Category Z-Score V.S. TFDV', autodv.cat_dist_class, ['L-inf'], dtype='category')

    # plot_dynamic_result_off_line()

    precision = load_file('../result/dynamic_z_score/dynamic_cat_precision.pk')
    precision = precision.swapaxes(0, 1)
    num_dist = precision.shape[0]
    plt.figure(figsize=(16, 9))

    for i in range(precision.shape[0]):
        plt.subplot(num_dist, 1, i+1)
        plt.hist(precision[i])
        plt.ylabel(dist_class[i], rotation=30)
        # plt.xlim((0, 1))

        if i== 0:
            plt.title('Average {}:{}'.format('Category Precision', 
                [float('%.2f'%val) for val in precision.mean(axis=1)]))

    # plt.tight_layout()   
      
    plt.savefig('../result/dynamic_z_score/dynamic_cat_precision/dynamic_cat_precision.png')
    plt.show()
