# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:03:19 2024

@author: User
"""
import pandas as pd 
def merge_taxes(data):

    taxes_data= pd.read_csv('taxlevels.csv',delimiter=',') 
    # taxes_data=taxes_data.drop(labels=['State/Province'  ,'State Tax Rate'], axis =1 )
    # taxes_data.rename(columns={'Team': 'TEAM_ABBREVIATION'}, inplace=True)
    # taxes_data.rename(columns={'Tax Level': 'Tax_Level'}, inplace=True)
    merged_data = data.merge(taxes_data, on='TEAM_ABBREVIATION', how='left')
    return merged_data


# new_teams = pd.DataFrame({
#     'TEAM_ABBREVIATION': ['SEA', 'NJN', 'CLE', 'VAN', 'CHH', 'NOH', 'CHA', 'NOK', 'NOP'],
#     'Tax_Level': ['Low', 'Moderate', 'No Tax', 'No Tax', 'Low', 'Low', 'Moderate', 'Low', 'Low']  # Example tax levels
# })

# # Append the missing teams to the tax_data DataFrame
# updated_tax_data = pd.concat([taxes_data, new_teams], ignore_index=True)

# updated_tax_data.to_csv('taxlevels.csv', index=False)