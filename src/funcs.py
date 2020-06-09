import pandas as pd
import numpy as np
import scipy.stats as stats

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


import sklearn
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score

from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LinearRegression

from sklearn.metrics import confusion_matrix

#read-in data frame from data/consolidate_v2.ipynb
df_Xy = pd.read_csv('data/dfXy.csv')

def plot_trends(X,y):
    
    """
    PARAMETERS:
    
    X - pandas Dataframe equivalent to feature matrix
    y- pandas Series equivalent to target
    """
    
    y_label = input('What is your Target? Include Units ')



    fig, axs = plt.subplots(len(X.columns), figsize = (15,60))

    for col, ax in zip(X.columns,axs):
    
        tgt = y
        series = X[col]

        ax.plot(series.index,series, label = col)
        ax.tick_params(axis='y', labelcolor='red')

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis


        ax2.set_ylabel(y_label, fontsize = 12)  # we already handled the x-label with ax1
        ax2.plot(tgt.index,tgt, color = 'blue', alpha = 0.5, label = 'gold')

        ax2.tick_params(axis='y', labelcolor='blue')


        ax.set_title(f'{col} vs Price of Gold, Historical')
        
        if 'rate' in col:
            ax.set_ylabel(f'{col} (%)', fontsize = 12)

        else:
            ax.set_ylabel(f'{col} (Index Score)', fontsize = 12)
        
        ax2.legend()
    
    plt.tight_layout()
    
    return None

def cross_val_and_score(model,X_train, X_test, y_train, y_test):
    
    
    """
    PARAMETERS:
    model - unfit scikit learn model object
    X_train - Training feature dataset in pandas DataFrame
    X_test - Test feature dataset in pandas series
    y_train - Training target dataset in pandas DataFrame
    y_test - Test target dataset in pandas series
    
    RETURNS:
    numpy array of cross_validation and final train/test score
    
    """
    
    tscv = TimeSeriesSplit()
    
    score_label = ''
    
    if 'class' in str(model).lower():
        score_label = 'Model Accuracy Score'
    else:
        score_label = 'Model R^2 Score'
        
    
    score = []
    count = 1
    for tr_index, val_index in tscv.split(X_train):
        X_tr, X_val = X_train.loc[tr_index], X_train.loc[val_index]
        y_tr, y_val = y_train.loc[tr_index], y_train.loc[val_index]
        
        m_dif = model
    
        m_dif.fit(X_tr,y_tr)
        
        score_ = m_dif.score(X_val,y_val)
    
        score.append(score_)
        print(f'Cross_Val {count} {score_label}:{round(score_,3)}')
        count += 1

    m_dif = model
    m_dif.fit(X_train,y_train)
    
    score_ = m_dif.score(X_test,y_test)
    score.append(score_)
    print(f'Final {score_label}:{round(score_,3)}')
    return score, m_dif

def plot_model(model,X_test,y_test):

    """PARAMETERS:
    model - scikit learn model object, fit with training data
    X_test - Test feature dataset in pandas DataFrame
    y_test - Test target dataset in pandas Series
    
    RETURNS:
    Plots predictions on top of values in time series plot in matplotlib
    """
    

    title = input('What is this model called?: ')
    y_label = input('What is your target variable?: ')
    
    fig,ax = plt.subplots(figsize = (15,12))
    
    ax.plot(df_Xy.index[X_test.index],model.predict(X_test), label = 'Predicted')
    ax.plot(df_Xy.index[y_test.index], y_test, label = 'Actual', alpha = 0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    
    return None

def cross_val_and_score_ma(y_ma_train, y_train, y_ma_test, y_test):
    
    
    """
    PARAMETERS:
    
    y_ma - Moving average to compare against target in pandas series format
    y_test - Test target dataset in pandas series format
    
    RETURNS:
    numpy array of cross_validation and final train/test score
    
    """
    
    tscv = TimeSeriesSplit()
    
    ma_type = input('Is this moving average a classifier(enter "c") or regressor(enter "r")?')
    
    score_label = ''
    
    score = []
    count = 1
    
    if ma_type == 'c':
        score_label = 'Model Accuracy Score'
        
        for tr_index, val_index in tscv.split(y_train):
            y_ma_tr, y_ma_val = y_ma_train.loc[tr_index], y_ma_train.loc[val_index]
            y_tr, y_val = y_train.loc[tr_index], y_train.loc[val_index]
        
            score_ = accuracy_score(y_val, y_ma_val)
            score.append(score_)
            print(f'Cross_Val {count} {score_label}:{round(score_,3)}')
            count += 1


        
        score_ = accuracy_score(y_test,y_ma_test)
        score.append(score_)
        print(f'Final {score_label}:{round(score_,3)}')
        return score
    else:
        score_label = 'Model R^2 Score'
        
        for tr_index, val_index in tscv.split(y_train):
            y_ma_tr, y_ma_val = y_ma_train.loc[tr_index], y_ma_train.loc[val_index]
            y_tr, y_val = y_train.loc[tr_index], y_train.loc[val_index]
        
            score_ = r2_score(y_val,y_ma_val)
            score.append(score_)
            print(f'Cross_Val {count} {score_label}:{round(score_,3)}')
            count += 1


        
        score_ = r2_score(y_test,y_ma_test)
        score.append(score_)
        print(f'Final {score_label}:{round(score_,3)}')
        return score

def plot_ma_model(y_ma,y_test):

    """PARAMETERS:
    model - scikit learn model object, fit with training data
    X_test - Test feature dataset in pandas DataFrame
    y_test - Test target dataset in pandas Series
    
    RETURNS:
    Plots predictions on top of values in time series plot in matplotlib
    """
    

    title = input('What is this model called?: ')
    y_label = input('What is your target variable?: ')
    
    fig,ax = plt.subplots(figsize = (15,12))
    
    ax.plot(df_Xy.index[y_ma.index],y_ma, label = 'Predicted')
    ax.plot(df_Xy.index[y_test.index], y_test, label = 'Actual', alpha = 0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    
    return None

def print_confusion_matrix(model,X_test,y_test):

    """PARAMETERS:
    model - scikit learn classifier model object, fit with training data
    X_test - Test feature dataset in pandas DataFrame
    y_test - Test target dataset in pandas Series
    
    RETURNS:
    pandas Dataframe confusion matrix for the model
    """
    
    tn, fp, fn, tp = confusion_matrix(y_test,model.predict(X_test)).ravel()
    
    d_true = {'Actual Positive': [tp, fn], 'Actual Negative': [fp,tn]}
    
    df = pd.DataFrame.from_dict(d_true, orient = 'columns')
    
    df.rename(index = {0:'Predicted Positive', 1: 'Predicted Negative'}, inplace = True)
    return df





