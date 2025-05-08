# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 20:09:23 2025

@author: User
"""
import pandas as pd 
import numpy as np

def check_high_corr_pairs(corrmat,df,ratio=0.9,remove=False ):
    # Get upper triangle of correlation matrix (no duplicates)
    upper_triangle = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))
    
    
    # pairs with correlation > 0.9
    high_corr_pairs = upper_triangle.stack().reset_index()
    high_corr_pairs.columns = ['Feature-1', 'Feature-2', 'Correlation']
    high_corr_pairs = high_corr_pairs[high_corr_pairs['Correlation'] > ratio].sort_values('Correlation', ascending=False)
    
    print(f'Top collinear pairs (correlation > {ratio}): \n {high_corr_pairs}')
    if(remove):
    # Remove featues based on the high correlation pairs
        df=df.drop(labels=['FGM','FG3A','FTA', 'FGA', 'REB_PCT','REB','POSS' ], axis =1 ) 
    return df,high_corr_pairs
# # ex
# df,high_corr_pairs=check_high_corr_pairs(corrmat,df,ratio=0.9,remove=False )



