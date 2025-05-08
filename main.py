# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:35:10 2024


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

import util 
from sklearn.preprocessing import PowerTransformer

# =============================================================================
# Data prep
# =============================================================================

data= pd.read_csv('api_hoops_common__2000_23_stats_salary_df.xls')

data.info()


df=data.drop(labels=['Rank','Base Salary','PLAYER_ID','PLAYER_NAME_x','PLAYER_NAME_y','SEASON','PERSON_ID'
                     ,'DISPLAY_FIRST_LAST' ], axis =1 )
df.rename(columns={'Adjusted Salary': 'salary'}, inplace=True)
df.rename(columns={'Season': 'season'}, inplace=True)
df.rename(columns={'SCHOOL': 'college'}, inplace=True)
df.rename(columns={'COUNTRY': 'country'}, inplace=True)
df.rename(columns={'DRAFT_YEAR': 'draft_year'}, inplace=True)
df.rename(columns={'DRAFT_ROUND': 'draft_round'}, inplace=True)
df.rename(columns={'DRAFT_NUMBER': 'draft_number'}, inplace=True)


df=util.clean_data(df)

# =============================================================================
# import taxes
# df=taxes.merge_taxes(df)
# =============================================================================
# =============================================================================
#salarycap//salary allocation//sallary percentage 
# df=util.add_salary_cap(df)
# df=df.drop(columns=['Salary_allocation'])
# df=util.target_salary_cap_percentage(df)
# =============================================================================
df.info()

# Counting categorical and numerical values

categorical=0
numerical=0
for type in df.dtypes:
    # print (type)
    if type == 'object':
        categorical+=1
    else :
        numerical+=1
print(f'Number of categorical values: {categorical} \nNumber of numerical values: {numerical} ')

#Stats  
df.describe()    
df.salary.describe().apply(lambda x: format(x, 'f'))

#Plot games-Salary
fig, ax = plt.subplots()
ax.scatter(x=df["GP"], y=df["salary"], s=1)
plt.ylabel("Salary", fontsize=13)
plt.xlabel("GP (Games Played)", fontsize=13)
plt.show()

df = df[df['GP'] >= 20]
fig, ax = plt.subplots()
ax.scatter(x=df["GP"], y=df["salary"],s=1 )
ax.set_xlim(xmin=0)
plt.ylabel("Salary", fontsize=13)
plt.xlabel("GP (Games Played)", fontsize=13)
plt.show()


# Correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
corrmat = numeric_df.corr()

f, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(corrmat, vmax=0.8, square=True);

#Correlation between salary and others
salary_correlations = corrmat['salary']
print(salary_correlations.sort_values(ascending=False))

# check highly correllated pairs 
from feature_selection import check_high_corr_pairs

df,high_corr_pairs = check_high_corr_pairs(corrmat=corrmat,df=df,ratio=0.9,remove=True)


# =============================================================================
# Data scaling
# =============================================================================

# #Distribution plot
util.get_distribution_plot(df['salary'])   

# Log-transformation of the salary variable
#Use the numpy fuction log1p which  applies log(1+x) to all elements of the column
df['salary'] = np.log1p(df['salary'])

# #Check the new distribution 
util.get_distribution_plot(df['salary'])   

#Find numeric features
numeric_feats = (df.dtypes.drop(['salary','season']))[df.dtypes != "object"].index

# # Check skew of features

# Check the skew of all numerical features
skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness = skewness.drop('salary')
print(skewness)

# for feat in numeric_feats:
#     sns.displot(df[feat]).set(title=feat + ' skew: ' + f'{skew(df[feat]):.4f}') 
#     # plt.legend(feat + 'skew'+str(skew(df[feat])))

# Transformation of  skewed features
skewness = skewness[abs(skewness) > 0.75]
skewness=skewness.dropna()
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

# yeo-johnson transformation
from scipy.special import boxcox1p
skewed_features = skewness.index
pt = PowerTransformer(method='yeo-johnson')
pt.fit(df[skewed_features])
df[skewed_features]=pt.transform(df[skewed_features])

# Scaling
from sklearn.preprocessing import StandardScaler,RobustScaler,Normalizer 
scaler = RobustScaler()

df[numeric_feats]=scaler.fit_transform(df[numeric_feats])


# Label encoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['TEAM_ABBREVIATION']= labelencoder.fit_transform(df['TEAM_ABBREVIATION'].values) 
# df['Position']= labelencoder.fit_transform(df['Position'].values) 
df['season']= labelencoder.fit_transform(df['season'].values) 
df['college']= labelencoder.fit_transform(df['college'].values) 
df['country']= labelencoder.fit_transform(df['country'].values) 
try:
    df['Tax_Level']= labelencoder.fit_transform(df['Tax_Level'].values) 

except:
    print('tax_level is not used')
# =============================================================================
# Split train -test 
# =============================================================================

# Models
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

target=df['salary']
feature=df.drop(columns=['salary','Player'])

train=df[df['season']<16]
test=df[df['season']>=16]


# =============================================================================
# Validate and compare models
# =============================================================================

# import sliding_window
from sliding_window import run_validation_and_analysis
# Determine feature columns 
exclude_columns = ['Player', 'salary']
features = [col for col in df.columns if col not in exclude_columns]
target = 'salary'

# Models to Validate

lasso =  Lasso(alpha =0.001, random_state=1)
enet =  ElasticNet(alpha=0.001, l1_ratio=.9, random_state=1)
krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
xgboost = xgb.XGBRegressor()
rf = RandomForestRegressor( n_estimators=100,random_state=1)
lgbm = lgb.LGBMRegressor(force_col_wise=True)
catb = CatBoostRegressor(logging_level='Silent')
svr = SVR()

models = {
    'Lasso': lasso,
    'Enet':enet,
    'Krr':krr,
    'Xgboost':xgboost,
    'RF':rf,
    'Lgboost':lgbm,
    'Catboost':catb,
    'Svr':svr,
    '(Base)SimpleLinear': LinearRegression(),
}

# Run Validation and Analysis
train_window_size = 3
start_train_seasons = 3
test_size = 1

all_results_df, aggregated_metrics,rolling_splits,expanding_splits = run_validation_and_analysis(
    data=train,
    features=features,
    target=target,
    models=models,
    train_window_size=train_window_size,
    start_train_seasons=start_train_seasons,
    test_size=test_size
    )

print(f'Aggregated_metrics\n {aggregated_metrics}')

from util import run_gridsearch,run_multiple_gridsearch


baseline_model={
    'Lgbm':lgbm,
    'Xgboost':xgboost,
    'Catboost':catb,
    # 'Lgboost2':lgbm,
    }


best_models_df, best_params_df, rmse_df, r2_df = run_multiple_gridsearch(
    baseline_model, train, test, features, target)



"""
Saved best parameters from gridsearch

best_params_df=pd.read_csv('best_params.csv')
import ast
best_params_df['Best Parameters'] = best_params_df['Best Parameters'].apply(ast.literal_eval)
rmse_df=pd.read_csv('rmse_df.csv',index_col=False)
rmse_df.index=['Rmse_tuned','Rmse_base']
r2_df=pd.read_csv('r2_df.csv')
r2_df.index=['R2_tuned','R2_base']

"""


# =============================================================================
# # PLOT BEST MODEL PREDICTED VS ACTUAL VALUES
# =============================================================================


best_parameters=best_params_df.loc[best_params_df['Model'] == 'Catboost', 'Best Parameters'].iloc[0]

# if used: Saved best parameters from gridsearh

best_model=CatBoostRegressor(**best_parameters,logging_level='Silent')
best_model.fit(train[features], train[target])

y_pred = best_model.predict(test[features])  # Predicted values

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(test[target], y_pred, color='blue', label='Predicted vs Actual')
# plt.scatter(np.expm1(y_test), y_pred, color='blue', label='Predicted vs Actual')

# Plot the y=x line (ideal case)
plt.plot([min(test[target]), max(test[target])], [min(test[target]), max(test[target])], color='red', linestyle='--', label='Ideal Line (y = x)')
# plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))], [min(np.expm1(y_test)), max(np.expm1(y_test))], color='red', linestyle='--', label='Ideal Line (y = x)')

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.legend()

# Display the plot
plt.show()


# =============================================================================
# # Feature importance 
# =============================================================================

# Built -in Catboost methods
feature_importance = best_model.get_feature_importance()
for name, importance in zip(features, feature_importance):
    print(f"Feature: {name}, Importance: {importance:.2f}")
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(20, 15))
sns.barplot(x=feature_importance[sorted_idx[::-1]], y=np.array(features)[sorted_idx[::-1]], palette='Spectral')

plt.title('Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Features',  fontsize=12)   
plt.tight_layout()
plt.show()

# Using SHAP 
import shap
shap.initjs()

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(test[features])
# features = X_train.columns.tolist()

shap.summary_plot(shap_values, test[features], feature_names=features, plot_type="bar")
shap.summary_plot(shap_values, test[features], feature_names=features)

# Check Feature importance using LIME
import lime
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer( test[features].values, feature_names=features,verbose=True, mode='regression')


# Write html object to a file and read on browser
import webbrowser

for sample_idx in [10,15]:
    exp = explainer.explain_instance( test[features].values[sample_idx], best_model.predict, num_features=10)
    exp.save_to_file(f'lime_temp/lime_{sample_idx}.html')
    url = f'C:/Users/User/sports_analytics/lime_temp/lime_{sample_idx}.html'
    webbrowser.open(url, new=2)

# =============================================================================
# Residuals
# =============================================================================

unscaled_pred=np.expm1(y_pred)/1000000
unscaled_actual=np.expm1(test[target].values)/1000000

d={'prediction':y_pred,'actual':test[target].values, 'unscaled pred (millions)':unscaled_pred , 'unscaled actual (millions)':unscaled_actual,
   'player':test.Player.values,
   'season':test.season.values,
   'residual':unscaled_pred-unscaled_actual
   }
residual_df=pd.DataFrame(d)



# =============================================================================
# Ensembles
# =============================================================================
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from util import train_score_ensemble


cope_rmse,copy_r2=rmse_df, r2_df
#Tuned models
best_catboost_params=best_params_df.loc[best_params_df['Model'] == 'Catboost', 'Best Parameters'].iloc[0]
best_catboost=CatBoostRegressor(**best_catboost_params,logging_level='Silent')

best_lgbm_params=best_params_df.loc[best_params_df['Model'] == 'Lgbm', 'Best Parameters'].iloc[0]
best_lgbm = lgb.LGBMRegressor(**best_lgbm_params,force_col_wise=True)

best_xgboost_params=best_params_df.loc[best_params_df['Model'] == 'Xgboost', 'Best Parameters'].iloc[0]
best_xgboost=xgb.XGBRegressor(**best_xgboost_params)
# best_xgboost=xgb.XGBRegressor()

# Voting Regressor
vr = VotingRegressor(
    [('Catboost', best_catboost),
    ('Xgboost', best_xgboost), 
    ('Lgbm', best_lgbm),
    ],weights=[3,2,2])

vr,rmse_df,r2_df=train_score_ensemble(
                                ensemble=vr,
                                rmse_df=rmse_df.loc['Rmse_tuned'],
                                r2_df=r2_df.loc['R2_tuned'],
                                name='Voting_regressor',
                                test=test,
                                train=train,
                                features=features,
                                target=target)
# Stacking Regressor 

# cat+xgb+lgbm-> svr -> out
estimators = [
    ('Catboost', best_catboost),
    ('Xgboost', best_xgboost), 
    ('Lgbm', best_lgbm),
]

sr = StackingRegressor(
    estimators=estimators,
    final_estimator= SVR()
)

sr,rmse_df,r2_df=train_score_ensemble(
                                ensemble=sr,
                                rmse_df=rmse_df,
                                r2_df=r2_df,
                                name='Stacking_regressor',
                                test=test,
                                train=train,
                                features=features,
                                target=target)

# sr_complex : (cat+xgb+lgbm ) -> vr(svr,krr,lasso) -> out

vr2 = VotingRegressor(
    [('SVR', svr),
    ('Krr', krr), 
    ('Lasso', lasso),
    ],weights=[3,1,2])

sr_complex=StackingRegressor(
    estimators=estimators,
    final_estimator= vr2
)

sr_complex,rmse_merge,r2_merge=train_score_ensemble(
            ensemble=sr_complex,
            rmse_df=rmse_df,
            r2_df=r2_df,
            name='Complex_Stacking_regressor',
            test=test,
            train=train,
            features=features,
            target=target)


