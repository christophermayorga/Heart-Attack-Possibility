import pandas as pd
import numpy as np
import os
import requests

# turn off warning boxes
import warnings
warnings.filterwarnings('ignore')


def get_df():
    '''
    returns Heart Attack Possibility data into a csv
    '''
    if os.path.isfile('heart.csv'):
        df = pd.read_csv('heart.csv')#, index_col=0)
        return df
    
    
def get_info(df):
    '''
    This function takes in a dataframe and prints out information about the dataframe.
    '''

    print(df.info())
    print()
    print('------------------------')
    print()
    print('This dataframe has', df.shape[0], 'rows and', df.shape[1], 'columns.')
    print()
    print('------------------------')
    print()
    print('Null count in dataframe:')
    print('------------------------')
    print(df.isnull().sum())
    print()
    print('------------------------')
    print(' Dataframe sample:')
    print()
    return df.sample(3)