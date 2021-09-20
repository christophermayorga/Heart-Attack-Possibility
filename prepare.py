import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from pydataset import data
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix,mean_squared_error, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# import logistic_regression_util
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from matplotlib.colors import ListedColormap

def clean_heart_data(df):
    '''
    This function takes in the raw dataset and returns a cleaned dataframe.
    '''
    # re-name the columns
    df = df.rename(columns={'sex': 'is_male',
                            'cp': 'chest_pain',
                            'trestbps': 'resting_blood_pressure',
                            'chol': 'cholesterol',
                            'fbs': 'fasting_blood_sugar',
                            'thalach': 'max_heart_rate'})
    
    # dropping the duplciates rows
    df = df.drop_duplicates()
    
    return df

def split_data(df):
    '''
    This function is designed to split out data for modeling into train, validate, and test 
    dataframes.
    
    It will also perform quality assurance checks on each dataframe to make sure the target 
    variable was correctcly stratified into each dataframe.
    '''
    
    ## splitting the data stratifying for out target variable
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123,
                                        stratify = df.target)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123,
                                   stratify= train_validate.target)
    
    print('Making Sure Our Shapes Look Good')
    print(f'Train: {train.shape}, Validate: {validate.shape}, Test: {test.shape}')
    
    print('Making Sure We Have Positive Cases In Each Split\n')
    
    print('Train Target Value Counts:')
    print(train.target.value_counts())
    print('----------------------------\n')
    
    print('Validate Target Value Counts:')
    print(validate.target.value_counts())
    print('----------------------------\n')
    
    print('Test Target Value Counts:')
    print(test.target.value_counts())
    print('----------------------------\n')
    
    return train, validate, test

def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).
     
    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled