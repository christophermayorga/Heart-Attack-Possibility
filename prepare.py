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