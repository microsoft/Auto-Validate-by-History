import pandas as pd

def _init_dict():
    """
    for accumulating recording
    """
    global glo_eff_dict
    glo_eff_dict = {'single_time': 0, 
                    'two_time': 0, 
                    'clause_time': 0,
                    'total_time': 0,

                    'clause_exclude_time': 0,
                    
                    'dist_num': 0,
                    'clause_num': 0
                    }


def _init_df():
    """
    dataframe for final output
    """

    global glo_eff_df
    glo_eff_df = {
        'single_df': pd.DataFrame(columns=['col', 'num_iter', '1000', '2000', '3000', '4000', '5000']),
        'two_df': pd.DataFrame(columns=['col', 'num_iter', '1000', '2000', '3000', '4000', '5000']),
        'clause_df': pd.DataFrame(columns=['col', 'num_iter', '1000', '2000', '3000', '4000', '5000']),
        'other_df': pd.DataFrame(columns=['col', 'num_iter', '1000', '2000', '3000', '4000', '5000']),
        'total_df': pd.DataFrame(columns=['col', 'num_iter', '1000', '2000', '3000', '4000', '5000'])
    }


def _init():
    _init_dict()
    _init_df()


def update_df(key, value):
    eff_df = pd.DataFrame(value, columns=['col', 'num_iter', '1000', '2000', '3000', '4000', '5000'])
    glo_eff_df[key] = glo_eff_df[key].append(eff_df, ignore_index=True)


def get_df(key):
    return glo_eff_df[key]


def set_value(key, value):
    """ define a global variable """

    glo_eff_dict[key] = value


def get_value(key):
    """ get  a global varibale """

    return glo_eff_dict[key]


if __name__ == '__main__':
    print('test')