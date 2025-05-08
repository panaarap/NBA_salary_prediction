# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:23:17 2024

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline 
import seaborn as sns
from scipy.stats import norm, skew, probplot
from scipy import stats
import cpi
from sklearn.model_selection import KFold, cross_val_score, train_test_split

def read_data(filename):
    data=pd.read_csv(filename,delimiter=';') 
    data['college'] = data['college'].astype(str).str.replace("[^A-Za-z0-9_]+","_",regex=True)
    data['college'] = data['college'].astype(str).str.replace(" ","_",regex=True)
    data.season=data.season.str.split('-', expand=True)[0] # Keep 20XX from 20XX-YY
    return data

def clean_data(data):
    # make copy of dataframe
    df=data
    
    # fill nan college values with 'Missing' 
    df.loc[:, ('college')]=df.loc[:, ('college')].fillna('Missing')
    
    # fill nan draft_number values with int 100 
    df.loc[:, ('draft_number')]=df.loc[:, ('draft_number')].fillna(100)
    
    
    
    # Clean $ sign and commas from salary 
    df.loc[:, 'salary']=df.loc[:, 'salary'].str.replace('$', '')
    df.loc[:, 'salary']=df.loc[:, 'salary'].str.replace(',', '')
    df.salary=pd.to_numeric(df.salary)
    # try:
    #     df.season=pd.to_numeric(df.season)
    #     # df.loc[df["draft_number"] == '0', "draft_number"] = 100
    # except:
    #     print('season not numeric or no draft_number=0')
        
    
    
    pd.to_numeric(df.loc[:, 'salary'])
    
    # combine draft number and round information / drop draft year
    df.loc[df["draft_number"] == "Undrafted", "draft_number"] = 100
    # df.loc[:, 'draft_number']=pd.to_numeric(df.draft_number)
    df.draft_number=pd.to_numeric(df.draft_number)
    try:
        df.loc[df["draft_number"] == 0, "draft_number"] = 100
    except:
        print('no draft number = 0')
    df=df.drop(labels=['draft_round','draft_year'], axis =1 )
    
    # remove nan  rows
    df=df.dropna()
    df.info()
    return df

def get_inflated(df,save=False):
    cpi.update()
    # df['salary_inflated'] = df.apply(lambda x: cpi.inflate(x['salary'], x['season']), axis=1)

    salary_arr=df.loc[:,'salary']
    season_arr=df.loc[:,'season']

    salary_inflated=[]
    for salary,season in zip (salary_arr,season_arr):
        salary_inflated.append(cpi.inflate(salary, season))
        
    df_inflated=df    
    df_inflated.loc[:,'salary_inflated']=salary_inflated
    if save:
        df_inflated.to_csv('salary_inflated.csv', index=False)
    return df_inflated

def add_fit_to_histplot(a, fit=stats.norm, ax=None):

    if ax is None:
        ax = plt.gca()

    # compute bandwidth
    bw = len(a)**(-1/5) * a.std(ddof=1)
    # initialize PDF support
    x = np.linspace(a.min()-bw*3, a.max()+bw*3, 200)
    # compute PDF parameters
    params = fit.fit(a)
    # compute PDF values
    y = fit.pdf(x, *params)
    # plot the fitted continuous distribution
    ax.plot(x, y, color='#282828')
    return ax

# # sample data
# x = np.random.default_rng(0).normal(1, 4, size=500) * 0.1

# # plot histogram with gaussian fit
# sns.histplot(x, stat='density')
# add_fit_to_histplot(x, fit=stats.norm);

def get_distribution_plot(variable):
    # Check Distribution of salary 
    sns.histplot(variable, stat='density', color='#1f77b4', alpha=0.4, edgecolor='none')
    add_fit_to_histplot(variable, fit=stats.norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(variable)
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title(str(variable.name)+' distribution')

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(variable, plot=plt)
    plt.show()
    return mu, sigma


#Validation function
def rmsle_cv(model,X_train,y_train,n_folds = 5):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    # rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    rmse= (-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_absolute_percentage_error", cv = kf))
# =============================================================================
#     # model.fit( X_train.values, y_train) !!!!REMEMBER THIS
# =============================================================================
    # r2=(cross_val_score(model, X_train.values, y_train, scoring="r2", cv = kf))
    # return(rmse,r2)
    return(rmse)
    # return(r2)

# def test_model(model):
#     score = rmsle_cv(model,X_train,y_train)
#     return model,score.mean()

def get_combined_salary_cap(df):
    # read salary cap csv
    salary_cap_df=pd.read_csv('salary_cap_df.xls',delimiter=',') 
    #clean salary cap scv
    salary_cap_df.loc[:, 'salary_cap']=salary_cap_df.loc[:, 'salary_cap'].str.replace('$', '')
    salary_cap_df.loc[:, 'salary_cap']=salary_cap_df.loc[:, 'salary_cap'].str.replace(',', '')
    salary_cap_df.salary_cap=pd.to_numeric(salary_cap_df.salary_cap)
    #combine df
    combined_df=pd.concat([
        df,
        salary_cap_df
    ], axis=1, ignore_index=False)
    combined_df = combined_df[[col for col in combined_df.columns if col != 'salary'] + ['salary']]
    return combined_df

def add_salary_cap(data):
    cap_data=pd.read_csv('salary_allocation_2000_2023.xls',delimiter=',') 
    list1=[]
    list2=[]
    for team,season in zip(data['TEAM_ABBREVIATION'].values,data['season'].values):
        # print(team)
        # print(season)
        # 'SEA', 'NJN', 'VAN', 'CHH', 'NOH', 'NOK'
        if team == 'NJN' : 
            team = 'BKN'
        if team=='NOH':
            team='NOP'
        if team=='SEA':
            team='OKC'
        if team=='VAN':
            team='MEM'
        if team=='CHH':
            team='CHA'
        if team=='NOK':
            team='NOP'
        # print(data.at[team,season])
        row=cap_data[(cap_data['Year']==season) & (cap_data['Team']==team)]
        
        try:
        
            list1.append(row.iloc[0]['Salary_allocation'],)
            list2.append(row.iloc[0]['Adjusted_salary_allocation'])
        except:
            
            print(f'Error while loading salary cap data on team name:{team}')
    
    final_df=data
    final_df['Salary_allocation']=list1
    final_df['Adjusted_salary_allocation']=list2
           
    return final_df

def target_salary_cap_percentage(data,cap='Adjusted_salary_allocation'):
    data['salary']=data['salary']/data[cap]
    data=data.drop(columns=[cap])
    return data

from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

def run_gridsearch(name,baseline_model,train,test,features,target):
    
    # PARAMETERS FOR EACH MODEL
    if name=='Lgbm':
        parameters = {
            'num_leaves': [31, 127],
            'reg_alpha': [0.1, 0.5],
            'min_data_in_leaf': [30, 50, 100, 300, 400],
            'lambda_l1': [0, 1, 1.5],
            'lambda_l2': [0, 1],
            'verbosity': [-1]
            }
   
    elif name == 'Xgboost':
        parameters = {
        
            'n_estimators': [950],
            'max_depth': [3, 4,],
            'learning_rate': [0.025],
            'subsample': [0.7, 0.75, 0.8],
            'colsample_bytree': [0.95, 0.975, 1.0],
            'min_child_weight': [8, 10, 12],
            'gamma': [0.5, 0.6, 0.7],
            'alpha': [8.0, 9.5, 10.0],
            'lambda': [6.0, 6.5, 7.0],
            
              }
        # known best parms
        # parameters={'alpha': [10.0], 'colsample_bytree': [0.975], 'gamma': [0.7], 'lambda': [7.0], 'learning_rate': [0.025], 'max_depth': [3], 'min_child_weight': [8], 'n_estimators': [950], 'subsample': [0.75]}
    elif name == 'Catboost':
        parameters = {
            # 'depth': [3,5,7],
            # 'learning_rate': [0.005,0.01],
            # 'num_leaves': [31, 127],
            
            'min_data_in_leaf': [30, 50, 100, ],
            
            
            # 'n_estimators': [50, 100, 200],
            # 'subsample': [0.6, 0.8, 1.0],
            # 'learning_rate': [ 0.05, 0.1],
            
            # 'l2_leaf_reg': [1,2,3,4,5,6],
            # 'subsample':np.arange(0.75,0.85,0.1),
            # 'depth': [4,6,8],
            # 'learning_rate': [ 0.04 ,0.05 , 0.06],
            "bootstrap_type": ["Bayesian"]#, "Bernoulli", "MVS"],
            ,"bagging_temperature" : np.arange(0,1.5,0.25)
                       }
        # known best parms
        # parameters={'bagging_temperature': [1.25], 'bootstrap_type': ['Bayesian'], 'min_data_in_leaf': [30]}
           
      
    tscv = TimeSeriesSplit(n_splits=5)
    
    gridsearch = GridSearchCV(baseline_model, parameters,cv=tscv)
    gridsearch.fit(train[features],train[target])
    # Best model is already train on the whole train dataset
    best_model = gridsearch.best_estimator_ 
    best_params = gridsearch.best_params_

    baseline_model.fit(train[features],train[target]) # train baseline model on train dataset
    
    # predictions
    y_pred_best = best_model.predict(test[features])
    y_pred_base = baseline_model.predict(test[features])
    
    # scores rmse + r2
    rmse_best=np.sqrt(mean_squared_error(test[target], y_pred_best))
    rmse_base=np.sqrt(mean_squared_error(test[target], y_pred_base))
    
    rmse=[rmse_best,rmse_base]
    
    r2_best=r2_score(test[target], y_pred_best)
    r2_base=r2_score(test[target], y_pred_base)
    
    r2=[r2_best,r2_base]
    
    rmse_df=pd.DataFrame(rmse, index=["Rmse_tuned", "Rmse_base"], columns=['RMSE'])
    ax=rmse_df.plot(kind='barh',grid=True,xlabel='value',title=f"Model: {name} Rmse  score base vs tuned model (lower is better)",
                 legend=0,figsize=(15, 12),fontsize=15)
    ax.title.set_size(25)
    for i, v in enumerate(rmse_df['RMSE']):
        ax.text(v + 0.002, i,'{:.3f}'.format(v), ha='left', va='center',fontsize=11)
   
    
    r2_df=pd.DataFrame(r2,index=["R2_tuned", "R2_base"], columns=['R2'])
    ax=r2_df.plot(kind='barh',grid=True,xlabel='value',title=f'Model: {name} R2 score base vs tuned model (higher is better)',
                 legend=0,figsize=(15, 12),fontsize=15)
    ax.title.set_size(25)
    for i, v in enumerate(r2_df['R2']):
        ax.text(v + 0.002, i,'{:.3f}'.format(v), ha='left', va='center',fontsize=11)
   
    return best_model,best_params,rmse_df,r2_df
       

def run_multiple_gridsearch(baseline_model,train,test,features,target):
    
    # Initialize 
    all_best_models = []
    all_best_params = []
    all_rmse = []
    all_r2 = []
    
    
    for name,model in baseline_model.items():
        print(f'Evaluating model {name} ...')
        best_model,best_params,rmse,r2=run_gridsearch(name,model,train,test,features,target)
        
        
       # Append results 
        all_best_models.append(best_model)
        all_best_params.append(best_params)
        
        
        rmse.columns = [f'{name}_RMSE']
        r2.columns = [f'{name}_R2']
        
        all_rmse.append(rmse)
        all_r2.append(r2)
    
    # Combine RMSE and R2
    rmse_combined = pd.concat(all_rmse, axis=1)
    r2_combined = pd.concat(all_r2, axis=1)
    
    #  DataFrames best models and best parameters
    best_models_df = pd.DataFrame({
        'Model': list(baseline_model.keys()),
        'Best Model': all_best_models
    })
    
    best_params_df = pd.DataFrame({
        'Model': list(baseline_model.keys()),
        'Best Parameters': all_best_params
    })
    
    return best_models_df, best_params_df, rmse_combined, r2_combined
    
   
def train_score_ensemble(ensemble,rmse_df,r2_df,name,train, test,features,target):
    ensemble.fit(train[features], train[target])
    ensemble_results_df={'r2':[ensemble.score(test[features], test[target])], 'rmse': [np.sqrt(mean_squared_error(test[target], ensemble.predict(test[features])))] }
    ensemble_results_df=pd.DataFrame(ensemble_results_df, index=[name])

    print(f'{name} results: r2-> {ensemble_results_df.r2.values} rmse -> {ensemble_results_df.rmse.values}')

    rmse_merge=pd.concat([rmse_df,ensemble_results_df.rmse])
    ax_rmse=rmse_merge.plot(kind='barh',grid=True,xlabel='value',title=f"Rmse Score: Tuned models vs {name} (lower is better)",
                 legend=0,figsize=(15, 5),fontsize=15)
     # Set xlim with some padding for RMSE
    max_rmse = rmse_merge.max() * 1.05  # 5% padding
    ax_rmse.set_xlim(0, max_rmse)
    # annotations for RMSE
    for i, v in enumerate(rmse_merge):
        ax_rmse.text(v + 0.001, i, f"{v:.4f}", color='black', fontweight='bold', fontsize=10)
    plt.show()
    
    r2_merge=pd.concat([r2_df,ensemble_results_df.r2])
    ax_r2=r2_merge.plot(kind='barh',grid=True,xlabel='value',title=f"R2 Score: Tuned models vs {name} (Higher is better)",
                 legend=0,figsize=(15, 5),fontsize=15)
    ax_r2.set_xlim(0, 0.75)
    # annotations for R2
    for i, v in enumerate(r2_merge):
        ax_r2.text(v + 0.001, i, f"{v:.4f}", color='black', fontweight='bold', fontsize=12)
    plt.show()
    return ensemble,rmse_merge,r2_merge

