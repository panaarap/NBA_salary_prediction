# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 19:15:05 2024

@author: User
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm
import matplotlib.pyplot as plt
    
# Create Rolling and Expanding Splits
def rolling_window_splits_by_season(data, train_window_size, test_size=1):
    seasons = data['season'].unique()
    splits = []
    for i in range(train_window_size, len(seasons) - test_size + 1):
        train_seasons = seasons[i - train_window_size:i]
        test_seasons = seasons[i:i + test_size]
        train_indices = data[data['season'].isin(train_seasons)].index
        test_indices = data[data['season'].isin(test_seasons)].index
        splits.append((train_indices, test_indices))
    return splits

def expanding_validation_splits_by_season(data, start_train_seasons, test_size=1):
    seasons = data['season'].unique()
    splits = []
    for i in range(start_train_seasons, len(seasons) - test_size + 1):
        train_seasons = seasons[:i]
        test_seasons = seasons[i:i + test_size]
        train_indices = data[data['season'].isin(train_seasons)].index
        test_indices = data[data['season'].isin(test_seasons)].index
        splits.append((train_indices, test_indices))
    return splits

# Train and Evaluate Function
def train_and_evaluate(data, splits, features, target, model_name, model):
    results = []
    for idx, (train_indices, test_indices) in tqdm(enumerate(splits), total=len(splits)):
        # Split data
        train_data = data.loc[train_indices]
        test_data = data.loc[test_indices]

        # Separate features and target
        X_train, y_train = train_data[features], train_data[target]
        X_test, y_test = test_data[features], test_data[target]

        # Train the model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Store results
        results.append({
            'Split': idx + 1,
            'Train_Seasons': train_data['season'].unique().tolist(),
            'Test_Seasons': test_data['season'].unique().tolist(),
            'Model': model_name,
            'MAE': mae,
            'R2': r2,
            'RMSE': rmse,
            'MAPE': mape,
            'Train_Size': len(X_train),
            'Test_Size': len(y_test)
        })
    return pd.DataFrame(results)

# Aggregated Metrics for Evaluation
def evaluate_aggregated_metrics(results_df):
    aggregated = results_df.groupby(['Model', 'Validation_Type']).agg({
        'MAE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).reset_index()
    return aggregated



# Wrapper Function
def run_validation_and_analysis(data, features, target, models, train_window_size, start_train_seasons, test_size=1):
    # Generate splits
    rolling_splits = rolling_window_splits_by_season(data, train_window_size=train_window_size, test_size=test_size)
    expanding_splits = expanding_validation_splits_by_season(data, start_train_seasons=start_train_seasons, test_size=test_size)

    # Evaluate models and store results
    all_results = []
    for model_name, model in models.items():
        print(f'Validating: {model_name}')
        # Evaluate Rolling
        print('Rolling window..')
        rolling_results = train_and_evaluate(data, rolling_splits, features, target, model_name, model)
        rolling_results['Validation_Type'] = 'Rolling'

        # Evaluate Expanding
        print('Expanding window..')
        expanding_results = train_and_evaluate(data, expanding_splits, features, target, model_name, model)
        expanding_results['Validation_Type'] = 'Expanding'

        # Combine results
        all_results.append(rolling_results)
        all_results.append(expanding_results)

    all_results_df = pd.concat(all_results, ignore_index=True)

    # Aggregated metrics
    aggregated_metrics = evaluate_aggregated_metrics(all_results_df)

    # Visualization: Line Plot for RMSE
    plt.figure(figsize=(12, 6))
    for model_name in models.keys():
        model_results = all_results_df[all_results_df['Model'] == model_name]
        rolling_rmse = model_results[model_results['Validation_Type'] == 'Rolling']['RMSE']
        expanding_rmse = model_results[model_results['Validation_Type'] == 'Expanding']['RMSE']
        plt.plot(range(1, len(rolling_rmse) + 1), rolling_rmse, label=f'{model_name} Rolling')
        plt.plot(range(1, len(expanding_rmse) + 1), expanding_rmse, label=f'{model_name} Expanding')

    plt.xlabel('Split')
    plt.ylabel('RMSE')
    plt.title('Model RMSE Comparison (Rolling vs Expanding)')
    plt.legend()
    plt.show()

    # Visualization: Bar Plot for RMSE Comparison
    bar_data = all_results_df.groupby(['Model', 'Validation_Type'])['RMSE'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    bar_plot = bar_data.pivot(index='Model', columns='Validation_Type', values='RMSE')
    bar_plot.plot(kind='bar', figsize=(10, 6), colormap='viridis', alpha=0.8)
    plt.title('Mean RMSE by Model and Validation Type')
    plt.ylabel('Mean RMSE')
    plt.xticks(rotation=0)
    plt.legend(title='Validation Type')
    plt.tight_layout()
    plt.show()

    # # Save Results
    # all_results_df.to_csv('validation_results.csv', index=False)
    # aggregated_metrics.to_csv('aggregated_metrics.csv', index=False)

    return all_results_df, aggregated_metrics,rolling_splits,expanding_splits




