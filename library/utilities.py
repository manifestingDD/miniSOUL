import pandas as pd
import numpy as np 
import time
from datetime import date

import warnings
warnings.filterwarnings("ignore") 


global_names = {
    'prefix': {
        'spend': 'M_P_',
        'inc': 'TIncT_P_',
        'minc': 'nMTIncT_P_',
        'cpt' : 'CPT_P_',
        'mcpt': 'MCPT_P_',
        'delta': 'Pct_Delta'
        },
    'months_abbv' : ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
    'months_full' : ['October', 'November', 'December', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September']
}



def crafting_time(raw_time_file, planning_years):
    """
    Processing raw time file from Snowflake. Adding week counts for each month
    args:
        raw_time_file (str): Path to the raw time file (CSV)
        planning_years (list): List of years to include in the time reference
    """
    time_ref  = pd.read_csv(raw_time_file) 
    time_ref = time_ref[[
        'FIS_WK_END_DT', 'FIS_YR_NB', 'FIS_MO_NB', 'FIS_QTR_NB', 'FIS_WK_NB', 
    ]].drop_duplicates() 
    time_ref = time_ref[time_ref['FIS_YR_NB'].isin(planning_years)]
    time_ref['FIS_WK_END_DT'] = pd.to_datetime(time_ref['FIS_WK_END_DT']).dt.date
    time_ref = time_ref.sort_values(by=['FIS_WK_END_DT']) 

    counting_months = time_ref.groupby(['FIS_YR_NB', 'FIS_MO_NB']).size().reset_index(name='weeks_count') 
    counting_months['lag_weeks'] = counting_months.groupby(['FIS_YR_NB'])['weeks_count'].cumsum().shift(fill_value=0) 
    counting_months.loc[counting_months.FIS_MO_NB == 1, 'lag_weeks'] = 0

    time_ref = time_ref.merge(counting_months, how = 'left', on = ['FIS_YR_NB', 'FIS_MO_NB']).drop_duplicates() 

    counting_months2 = time_ref[['FIS_YR_NB', 'FIS_MO_NB']].drop_duplicates()
    counting_months2 = counting_months2.reset_index(drop=True)
    counting_months2['lag_months'] = counting_months2.index   

    time_ref = time_ref.merge(counting_months2, how = 'left', on = ['FIS_YR_NB', 'FIS_MO_NB']).drop_duplicates() 

    return time_ref.reset_index(drop =True)



def split_months_indices(months, start_month, end_month):
    """
    Splits a list of months into two lists: one containing the months between the start and end month (inclusive),
    and another containing the rest of the months.
    """
    # Find the indices of the start and end months
    start_index = months.index(start_month)
    end_index = months.index(end_month)
    
    # Handle the case where the start month comes after the end month in the list
    if start_index <= end_index:
        # Get the indices of the months between the start and end months (inclusive)
        indices_between = list(range(start_index, end_index + 1))
        # Get the indices of the rest of the months
        indices_rest = list(range(start_index)) + list(range(end_index + 1, len(months)))
    else:
        # Get the indices of the months between the start and end months (inclusive), considering the wrap-around
        indices_between = list(range(start_index, len(months))) + list(range(end_index + 1))
        # Get the indices of the rest of the months
        indices_rest = list(range(end_index + 1, start_index))
    
    # Convert 0-based indices to 1-based indices
    indices_between = [index + 1 for index in indices_between]
    indices_rest = [index + 1 for index in indices_rest]
    
    return indices_between, indices_rest



def revise_spend(spend_plan, newSpend_array, planning_months):
    """
    Revising the spend plan with new spend values for a particular period (months)
    args:
        spend_plan (pd.DataFrame): Original spend plan DataFrame
        newSpend_array (list or np.array): New spend values to be applied
        planning_months (list): List of fiscal month numbers for which the spend is revised
    """
    spend_plan_revised = spend_plan.copy()
    spend_plan_revised.iloc[:, 1:] = spend_plan.iloc[:, 1:]
    for i, month in enumerate(planning_months):
        spend_plan_revised.loc[spend_plan_revised['FIS_MO_NB'] == month, spend_plan.columns[1:]] = newSpend_array[i*len(spend_plan.columns[1:]):(i+1)*len(spend_plan.columns[1:])]
    return spend_plan_revised


def metric_compare(df, varname): 
    v1 = df.loc[df[df.columns[0]] == varname, :].iloc[0, 1]
    v2 = df.loc[df[df.columns[0]] == varname, :].iloc[0, 2]
    difference = v2 - v1
    pct = difference / v1 * 100
    
    if difference < 0:
        direction = "Decrease"
        string_diff = f'<span style="color:maroon">{difference:.1f}</span>'
        pct_diff = f'<span style="color:maroon">{pct:.1f}%</span>'
    else:
        direction = "Increase"
        string_diff = f'<span style="color:blue">{difference:.1f}</span>'
        pct_diff = f'<span style="color:blue">{pct:.1f}%</span>'

    
    return direction, string_diff, pct_diff, difference