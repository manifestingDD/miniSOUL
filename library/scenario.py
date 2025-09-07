import pandas as pd
import numpy as np 
import time
from datetime import date

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore") 
from library.utilities import global_names



def multiplier_grid(df_plan, df_base, threshold):
    medias =  df_plan.columns[1:]
    grid = {
        'media' : list(medias), 
        'mty_12' : [12] * len(df_plan.columns[1:]),
        'mty_fixed' : [],
        'mty_dynamic' : []
    }

    base_check = df_base.copy()
    for x in medias:
        s_yr = base_check[x].values.sum() 
        if s_yr == 0:
            base_check[x] = 0
        else:
            base_check[x] = base_check[x].values / s_yr
    base_check = base_check > threshold / 100
    for x in medias:
        grid['mty_fixed'].append(base_check[x].sum())


    plan_check = df_plan.copy()
    for x in medias:
        s_yr = plan_check[x].values.sum() 
        if s_yr == 0:
            plan_check[x] = 0
        else:
            plan_check[x] = plan_check[x].values / s_yr
    plan_check = plan_check > threshold / 100
    for x in medias:
        grid['mty_dynamic'].append(plan_check[x].sum())


    return pd.DataFrame(grid)




def compute_reward_X(X, spend_data, planning_year, df_time, df_params, df_curve, df_adjust_grid):
    """
    Compute the rewards over a 104-week period for a given media type and calculate
    marginal increments and costs, with an updated multiplier calculation based on a threshold.

    Args:
        X (str): The media type (e.g., "X1").
        spend_data (pd.DataFrame): DataFrame containing spending data with columns:
            - 'FIS_MO_NB': Fiscal month number.
            - Media spending columns (e.g., 'X1', 'X2', 'X3').
        planning_year (int): The fiscal year of the current spending plan.
        threshold (float): A value between 0 and 100. Months with spending percentage
            greater than threshold/100 will be counted towards the multiplier.

    Returns:
        tuple: A tuple containing:
            - reward_df (pd.DataFrame): DataFrame of shape (104, 13) with the reward curves.
            - minc_X (float): Total marginal rewards for media X over 12 months.
            - mc_X (float): Total marginal costs for media X over 12 months.
    """
    r_12 = 12
    r_fixed = df_adjust_grid.loc[df_adjust_grid.media == X, 'mty_fixed'].values[0]
    r_dynamic = df_adjust_grid.loc[df_adjust_grid.media == X, 'mty_dynamic'].values[0]

    # ================================================================================================================
    # Step 1: Extract the spending array and compute the multiplier
    # ================================================================================================================
    spend_array = spend_data[X].values  # Spending for media X over 12 months
    # Compute total spending S_total
    S_total = spend_array.sum()
    if S_total > 0:
        benchmark_spend = 3 * S_total / np.count_nonzero(spend_array) # Compute the benchmark, or upper bound, for monthly spending
    else:
        benchmark_spend = 100000000000000000000000000000   # Some very large number as a placeholder

    # # Handle the case when total spending is zero to avoid division by zero
    # if S_total == 0:
    #     spend_percentages = np.zeros_like(spend_array)
    # else:
    #     # Compute the percentage of total spend per month
    #     spend_percentages = spend_array / S_total

    # # Compute the multiplier based on the threshold
    # multiplier = np.sum(spend_percentages >= (threshold / 100))

    # # If multiplier is zero, set it to 1 to avoid division by zero later
    # if multiplier == 0:
    #     multiplier = 1

    # ================================================================================================================
    # Step 2: Compute the timing stuff
    # ================================================================================================================
    # Prepare a list to hold the 104-length arrays for each month
    monthly_arrays = []

    # Get fiscal years and months from spend_data
    fiscal_months = spend_data.FIS_MO_NB.values.flatten()

    # Filter df_time for the relevant fiscal year
    df_time_filtered = df_time[df_time['FIS_YR_NB'] == planning_year]

    # Group by fiscal month and count the number of weeks in each month
    weeks_in_month = df_time_filtered.groupby('FIS_MO_NB')['FIS_WK_NB'].nunique()
    weeks_in_month = weeks_in_month.reindex(range(1, 13), fill_value=0)

    # Compute cumulative weeks to determine the starting week for each month
    cumulative_weeks = weeks_in_month.cumsum()
    start_weeks = [0] + cumulative_weeks.values.tolist()[:-1]

    timeline = df_time[df_time.FIS_YR_NB.between(planning_year, planning_year + 2)].shape[0]
    timeline = np.zeros(timeline)

    # Initialize list of marginal cost and increament
    minc_X = []
    mc_X = []

    # ================================================================================================================
    # Loop over each month to compute the rewards
    # ================================================================================================================
    for idx, S_mo in enumerate(spend_array):
        fiscal_month = fiscal_months[idx]

        if S_mo == 0:
            # If the spending is zero, append an array of zeros
            monthly_array = timeline.copy()  # Create a an array of all 0s
            monthly_arrays.append(monthly_array)
            minc_X.append(0)
            mc_X.append(0)
            continue

        # Step 2: Calculate annualized spending and find the closest match in df_params
        # S_yr = min(S_mo, benchmark_spend) * r_fixed   # Orignal GP logic
        S_yr = S_mo * r_fixed # New GP logic
        M_P_X_col = global_names['prefix']['spend'] + X
        TIncT_P_X_col = global_names['prefix']['inc'] + X
        CPT_P_X_col = global_names['prefix']['cpt'] + X
        nMTIncT_P_X_col = global_names['prefix']['minc'] + X  # Marginal rewards column
        MCPT_P_X_col = global_names['prefix']['mcpt'] + X
        Pct_Delta_col = 'Pct_Delta'   

        # Find the index of the closest spending value in df_params
        idx_closest = (np.abs(df_params[M_P_X_col] - S_yr)).idxmin()

        if idx_closest == 0:
            reward_mo = 0
        else:
            cpt_yr = df_params[CPT_P_X_col].iloc[idx_closest]
            if idx_closest >= df_params.iloc[1:].shape[0]:
                # Logic 1: use row CPT and actual monthly spend
                # ................................................................
                # DD logic:
                # reward_mo = df_params[inc_prefix + X].iloc[idx_closest] / multiplier
                # GP Logic:
                S_mo_cap = df_params[M_P_X_col].iloc[idx_closest] / r_fixed
                reward_mo = S_mo_cap / cpt_yr
            else:
            # Logic 2: read row (annual) increment and divide by the multiplie
            # ............................................................
                reward_mo = S_mo / cpt_yr 
            

        # Calculate minc_mo and mc_mo for marginal cost per reward
        mc_mo = 0.1/100 * S_mo * r_fixed
        if idx_closest >= df_params.iloc[1:].shape[0]:
            mcpt_yr = df_params[MCPT_P_X_col].values[-1]
            numerator = mc_mo / mcpt_yr 
            denominator = S_yr / df_params[M_P_X_col].values[-1]
            minc_mo = numerator / denominator
        else:
            minc_mo = mc_mo / df_params[MCPT_P_X_col].iloc[idx_closest]

        mc_X.append(mc_mo)
        minc_X.append(minc_mo)


        # Step 3: Generate the reward curve for the month
        timing_curve = df_curve[X].values  # Timing curve of length 52
        monthly_reward_curve = reward_mo * timing_curve
        # monthly_arrays.append(monthly_reward_curve)

        # Step 4: Create a 3yr-length array with appropriate leading zeros
        start_week = start_weeks[fiscal_month - 1]
        end_week = start_week + len(monthly_reward_curve) 

        monthly_array = timeline.copy()  
        monthly_array[start_week:end_week] = monthly_reward_curve 
        monthly_arrays.append(monthly_array)
    
    
    # # #Aggregate the monthly arrays into a final reward curve
    # # monthly_arrays = [arr[:len(timeline)] for arr in monthly_arrays]  # Ensure each array is at most 104 elements
    # # # Pad any shorter arrays to exactly 104 elements
    # # monthly_arrays = [np.pad(arr, (0, len(timeline) - len(arr)), 'constant') if len(arr) < len(timeline) else arr for arr in monthly_arrays]
    reward_X = np.sum(monthly_arrays, axis=0)


    # Construct the output DataFrame
    columns = [f'Month_{i+1}' for i in range(12)] + ['aggregated']
    data = np.column_stack(monthly_arrays + [reward_X])
    reward_df = pd.DataFrame(data, columns=columns)

    return reward_df, minc_X, mc_X


def marginal_computations(X, spend_data, planning_year, planning_months, unit_revenue, 
                          df_time, df_params, df_curve, df_adjust_grid):
    months = list(range(1, 13)) 
    planning_index = [month - 1 for month in planning_months if month in months]

    res = compute_reward_X(X, spend_data, planning_year, df_time, df_params, df_curve, df_adjust_grid) 
    mc_list = np.array(res[2])
    minc_list = np.array(res[1])

    plan_sum_mc = mc_list[planning_index].sum()
    plan_sum_minc = minc_list[planning_index].sum()

    mcpa = plan_sum_mc / plan_sum_minc
    mroas = unit_revenue * plan_sum_minc / plan_sum_mc
    return  mcpa, mroas, [plan_sum_mc, plan_sum_minc]


def compute_plan_reward(spend_data, planning_year, lead_years, lag_years,  
                        df_time, df_params, df_curve, df_adjust_grid):
    """
    Compute the reward curves for all medias over an extended time frame, broken down by month and aggregated.

    Args:
        spend_data (pd.DataFrame): DataFrame containing spending data with columns:
            - 'FIS_MO_NB': Fiscal month number.
            - Media spending columns (e.g., 'X1', 'X2', 'X3').
        lead_years (int): Number of years to add as leading zeros.
        lag_years (int): Number of years to add as trailing zeros.2
        planning_year (int): The fiscal year of the current spending plan.
        threshold (float): Threshold value for reward calculations.

    Returns:
        list: A list of 13 numpy arrays:
            - First 12 arrays represent the monthly rewards summed across all media
            - Last array represents the aggregated rewards summed across all media
            Each array includes leading and trailing zeros.
    """
    # Ensure global access to necessary dataframes
    # global df_curve, df_params, df_time


    # Step 1: Get the list of media columns from spend_data
    medias = spend_data.columns.tolist()[1:]

    # Initialize lists to store the monthly and aggregated reward arrays
    monthly_rewards = [[] for _ in range(12)]  # One list for each month
    aggregated_rewards = []  # List for aggregated rewards

    # Loop over each media to compute its reward curves
    for media in medias:
        # Extract spending data for the media
        media_spend_data = spend_data[['FIS_MO_NB', media]]

        # Call compute_reward_X for the current media
        reward_df = compute_reward_X(media, media_spend_data, planning_year, 
                                     df_time, df_params, df_curve, df_adjust_grid
                                     )[0]

        # Extract monthly columns and aggregated column
        for month in range(12):
            month_col = f'Month_{month+1}'
            monthly_rewards[month].append(reward_df[month_col].values)
        
        # Extract and store aggregated rewards
        aggregated_rewards.append(reward_df['aggregated'].values)

    # Step 2: Sum the reward arrays across all media for each month and aggregated
    summed_monthly_rewards = [np.sum(month_arrays, axis=0) for month_arrays in monthly_rewards]
    summed_aggregated_rewards = np.sum(aggregated_rewards, axis=0)

    # Step 3: Add leading and trailing zeros to each array
    leading_zeros = np.zeros(lead_years * 52)
    trailing_zeros = np.zeros(lag_years * 52)

    # Create final list of extended arrays (12 monthly + 1 aggregated)
    extended_reward_curves = []
    
    # Process monthly arrays
    for monthly_reward in summed_monthly_rewards:
        extended_monthly = np.concatenate([leading_zeros, monthly_reward, trailing_zeros])
        extended_reward_curves.append(extended_monthly)
    
    # Process aggregated array
    extended_aggregated = np.concatenate([leading_zeros, summed_aggregated_rewards, trailing_zeros])
    extended_reward_curves.append(extended_aggregated)

    return extended_reward_curves



def plan_forecast_craft(spend_data, planning_year, lead_years, lag_years, 
                        df_time, df_params, df_curve, df_adjust_grid):
    """
    Generate monthly and quarterly forecast tables based on spending data and reward calculations.

    Args:
        spend_data (pd.DataFrame): DataFrame containing spending data with columns:
            - 'FIS_MO_NB': Fiscal month number.
            - Media spending columns (e.g., 'NTV', 'ING', 'STR').
        planning_year (int): The fiscal year of the current spending plan.
        lead_years (int): Number of years to add as leading zeros.
        lag_years (int): Number of years to add as trailing zeros.
        cutoff (float): Threshold value for reward calculations.

    Returns:
        tuple: A tuple containing:
            - craft_mo (pd.DataFrame): DataFrame with monthly forecast data.
            - craft_qtr (pd.DataFrame): DataFrame with quarterly forecast data.
    """
    df_time_scenario = df_time[df_time['FIS_YR_NB'].between(planning_year - lead_years, planning_year + lag_years)]
    weekly_table = df_time_scenario[['FIS_WK_END_DT', 'FIS_YR_NB', 'FIS_QTR_NB', 'FIS_MO_NB']]
    results = compute_plan_reward(spend_data, planning_year, lead_years, lag_years, df_time, df_params, df_curve, df_adjust_grid)

    names = []
    for i in range(len(results)-1):
        serl = list(results[i])
        if len(serl) < weekly_table.shape[0]:
            serl = serl + [0] * (weekly_table.shape[0] - len(serl))
        if len(serl) > weekly_table.shape[0]:
            serl = serl[:weekly_table.shape[0]]
        col_name = str(planning_year) + ' ' + global_names['months_abbv'][i]
        names.append(col_name)
        weekly_table[col_name] = serl

    # Monthly Table
    # ================================================================================================================
    month_label = {}
    for i, item in enumerate(global_names['months_abbv']):
        month_label[i + 1] = item

    monthly_table = weekly_table.groupby(['FIS_YR_NB', 'FIS_MO_NB'])[names].sum().reset_index()
    rewards = monthly_table.iloc[:, 2:].values.T
    monthly_table.FIS_MO_NB.replace(month_label, inplace=True) 
    monthly_table['timeline'] = monthly_table.FIS_YR_NB.astype(str) + " " + monthly_table.FIS_MO_NB.astype(str) 
    
    shard1 = pd.DataFrame({'Spending Month': names, "Spend": spend_data.iloc[:, 1:].sum(axis = 1).values})
    shard2 = pd.DataFrame(rewards)
    shard2.columns = monthly_table['timeline'].values
    craft_mo = pd.concat([shard1, shard2], axis=1) 


    # Monthly Table
    # ================================================================================================================
    quarter_table = weekly_table.groupby(['FIS_YR_NB', 'FIS_QTR_NB'])[names].sum().reset_index()
    rewards = quarter_table.iloc[:, 2:].values.T 
    rewards = rewards.reshape(4, 3, -1) # Turning monthly tracking into quarterly tracking
    rewards = rewards.sum(axis = 1)
    quarter_table.FIS_QTR_NB = quarter_table.FIS_QTR_NB.astype(str)
    quarter_table['timeline'] = quarter_table.FIS_YR_NB.astype(str) + " Q" + quarter_table.FIS_QTR_NB.astype(str)

    names = [str(planning_year) + " Q" + str(x) for x in range(1, 5)]
    shard1 = pd.DataFrame({'Spending Quarter': names, 
                        "Spend": spend_data.iloc[:, 1:].values.sum(axis = 1).reshape(4, 3).sum(axis = 1)})
    shard2 = pd.DataFrame(rewards)
    shard2.columns = quarter_table['timeline'].values
    craft_qtr = pd.concat([shard1, shard2], axis=1) 

    return craft_mo, craft_qtr



def forecast_table_summarizer(table, target):
    """
    Taking a forecast table (from plan_forecast_craft) and summarizing it by adding a total row at the end.
    args:
        table (pd.DataFrame): The forecast table to summarize, output of `plan_forecast_craft` function.
    """
    shard1 = [f'Total {target}', ""]
    shard2 = list(table.iloc[:, 2:].sum(axis = 0).values)

    table.loc[-1] = np.array(shard1 + shard2)
    table = table.reset_index(drop = True)

    shard1 = table.iloc[:-1]
    shard2 = table.iloc[[-1]]

    table2 = pd.concat([shard2, shard1], axis = 0).reset_index(drop = True)
    return table2






def comparison_plot(media_labels, source_tables, metrics, names, colors, currency_symbol, title):
    """
    Create a comparison plot for two sets of media data.
    
    Parameters:
    - media_labels: List of media names.
    - source_tables: List of DataFrames containing media data.
    - metrics: List of metrics to compare.
    - names: Names for the two sets of data.
    - colors: Colors for the bars and lines.
    - currency_symbol: Symbol for currency formatting.
    - title: Title of the plot.
    
    Returns:
    - fig: Plotly figure object.
    """
    # Drop media with zero spend in both tables
    # ******************************************************************************
    table0 = source_tables[0]
    table1 = source_tables[1] 
    medias_drop0 = set(table0[table0.Spend == 0].Media.values)
    medias_drop1 = set(table1[table1.Spend == 0].Media.values)
    medias_drop = medias_drop0.intersection(medias_drop1)
    table0 = table0[~table0.Media.isin(medias_drop)]
    table1 = table1[~table1.Media.isin(medias_drop)]   
    media_labels = [label for label in media_labels if label not in medias_drop]

    # Gather metrics 
    # ******************************************************************************
    metric1_s1 = table0[metrics[0]].values / 1000
    metric1_s2 = table1[metrics[0]].values / 1000
    max_metric1 =  max(max(metric1_s1), max(metric1_s2))

    metric2_s1 = table0[metrics[1]].values 
    metric2_s2 = table1[metrics[1]].values
    max_metric2 =  max(max(metric2_s1), max(metric2_s2))

    metric1_compare = [(s2 - s1) / s1 * 100 for s1, s2 in zip(metric1_s1, metric1_s2)]
    metric2_compare = [(s2 - s1) / s1 * 100 for s1, s2 in zip(metric2_s1, metric2_s2)]
    default_textpos = 0.5 * max_metric1


    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Bar Plots
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    fig.add_trace(
        go.Bar(
            x= media_labels,
            y = metric1_s1, 
            name = f"{metrics[0]}, {names[0]}",
            marker_color = colors[0],
        )
    )
    fig.add_trace(
        go.Bar(
            x= media_labels,
            y = metric1_s2, 
            name = f"{metrics[0]}, {names[1]}",
            marker_color = colors[1]
        )
    )
    # Line Plots
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    fig.add_trace(
        go.Scatter(
            x = media_labels,
            y = metric2_s1,
            name=f"{metrics[1]}, {names[0]}",
            line=dict(
                color= "#9A9EA1", 
                width=3, dash='dash',
                shape='spline',  # This creates a smooth curve
                smoothing=1.3    # Adjust smoothing intensity (0.5-1.5 range works well)
                ),
            marker=dict(
                size=10,         # Larger marker size
                color=colors[0],
                line=dict(
                    width=1,
                    color='white'
                )
            )

        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x = media_labels,
            y = metric2_s2,
            name=f"{metrics[1]}, {names[1]}",
            line=dict(
                color= 'black', 
                width=3, dash='dash',
                shape='spline',  # This creates a smooth curve
                smoothing=1.3    # Adjust smoothing intensity (0.5-1.5 range works well)
                ),
            marker=dict(
                size=10,         # Larger marker size
                color=colors[1],
                line=dict(
                    width=1,
                    color='white'
                )
            )

        ),
        secondary_y=True,
    )

    # Annotations for metric 1 change
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for i, channel in enumerate(media_labels):
        change = metric1_compare[i]
        
        # Determine color based on change
        color = "green" if change >= 0 else "red"
        
        # Format text with plus/minus sign and percentage
        if change >= 0:
            text = f"+{change:.1f}%"  # Add plus sign for positive changes
        else:
            text = f"{change:.1f}%"   # Negative sign is automatically included
        
        # Improved positioning logic
        current_value = max(metric1_s1[i], metric1_s2[i])
        
        # If the value is very small (less than 5% of the maximum), use a fixed position
        if current_value < 0.03 * max_metric1:
            ypos = 0.12 * max_metric1  # Position at 15% of max height for very small values
        # If it's the maximum value, add a bit more space
        elif current_value >= 0.95 * max_metric1:
            ypos = 1.05 * max_metric1  # Position at 110% of max for the largest values
        # For medium values, position proportionally
        else:
            ypos = current_value + (0.125 * max_metric1)  # Position above the bar with consistent spacing
        
        # Add the annotation without arrows
        fig.add_annotation(
            x=channel,
            y=ypos, 
            text=text,
            showarrow=False,  # No arrow
            font=dict(
                color=color, 
                size=14,      # Slightly larger font for better visibility
                weight='bold' # Make it bold for emphasis
            ),
            align='center',
            bgcolor='rgba(255,255,255,0.7)',  # Semi-transparent white background
            bordercolor=color,
            borderwidth=1,
            borderpad=3
        )

    # Other cosmetics
    fig.update_layout(
        # Wider plot for spacing
        width=1300,
        height=700,
        # Extra large left margin
        margin=dict(t=80, r=50, b=100, l=150),
        # Title styling
        title=dict(
            text= title,
            font=dict(
                size=28,
                color= colors[1],
                weight='bold'
            ),
            x=0.5,
            xanchor="center"

        ),
        # Other layout
        barmode='group',
        legend=dict(
            x=1.15,              # Position the legend on the far left
            y=0.9,              # Position the legend at the very top
            xanchor="right",      # Anchor the legend's x-position to its left edge
            yanchor="top",       # Anchor the legend's y-position to its top edge
            bgcolor='rgba(255, 255, 255, 0.75)', # Optional: give legend a background
            bordercolor="Black", # Optional: give legend a border
            borderwidth=1        # Optional: set border width
        ),
        paper_bgcolor= "white",
        plot_bgcolor="white",
    )

    # Set x-axis properties
    fig.update_xaxes(
        title_text="",
        showgrid=False,
        showline=True,
        linewidth=2,
        linecolor='lightgray'
    )

    # Set y-axes properties
    fig.update_yaxes(
        title_text= f"{metrics[0]} (k)",
        title_font=dict(size=16),
        range=[0, 1.2 * max(max(metric1_s2), max(metric1_s1))],
        showgrid=True,
        gridcolor='lightgray',
        secondary_y=False,
        tickformat=','
    )

    fig.update_yaxes(
        title_text= metrics[1],
        title_font=dict(size=16),
        range=[0, 1.2 * max(max(metric2_s2), max(metric2_s1))],
        showgrid=False,
        secondary_y=True,
        tickprefix= currency_symbol,
        ticksuffix=''
    )
    
    return fig



def plan_wrapper(plan, 
                 planning_year,
                 planning_period, 
                 target_period, 
                 target_weeks,
                 price, 
                 media_list, media_labels, media_mapping,
                 df_base, df_params, df_time, df_curve, df_adjust_grid,
                 target
                 ):
    """
    Wraps the plan data and computes the summary statistics for the given planning year and period. 
    Args:
        plan (pd.DataFrame): The spending plan DataFrame with columns:
            - FIS_MO_NB: Fiscal month number
            - Media columns: Spending for each media type
        planning_year (int): The fiscal year of the current spending plan.
        planning_period (list): List of fiscal month numbers for the planning period, e.g. [1, 2, ..., 12].
        target_period (list): list of strings representing target weeks, e.g. ['2025 Jul', '2025 'Aug', ..., '2026 Dec'].
        target_weeks (list): list of index of target weeks in df_time_filtered, e.g. [90, 91, 92, ..., 104].
                             df_time_filtered = df_time[df_time['FIS_YR_NB'].between(planning_year-1, planning_year+1)]
        price (float): Unit price for attendance.
        media_list (list): List of media names in the plan.
        media_labels (list): List of media labels for display.
        media_mapping (dict): Mapping of media names to their labels.
        df_base (pd.DataFrame): Base data DataFrame.
        df_params (pd.DataFrame): Parameters DataFrame.
        df_time (pd.DataFrame): Time DataFrame.
        df_curve (pd.DataFrame): Curve DataFrame.
        df_adjust_grid (pd.DataFrame): Dataframe storing month-to-year adjust parameter under different rules
    """
    # ========================================================================================================================
    # Aggregate Summary
    # ========================================================================================================================
    info = {}
    # Spend
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    info['plan period spend'] = np.round(plan[plan.FIS_MO_NB.isin(planning_period)].iloc[:, 1:].values.sum(), 1)

    # Rewards
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    plan_reward_panel = plan_forecast_craft(plan, 
                        planning_year, 1, 2, 
                        df_time, df_params, df_curve, df_adjust_grid      
                        )[0]
    plan_driven_reward = plan_reward_panel.iloc[[x-1 for x in planning_period], :][target_period].values.sum()
    plan_driven_total_reward = plan_reward_panel.iloc[[x-1 for x in planning_period], 2:].values.sum()
    info[f'plan-driven {target} in target period'] = int(np.round(plan_driven_reward, 0))
    info[f'plan-driven total {target} (target period + future months)'] = int(np.round(plan_driven_total_reward, 0))

    base_reward_panel = plan_forecast_craft(df_base, 
                        planning_year - 1, 0, 3,
                        df_time, df_params, df_curve, df_adjust_grid       
                        )[0]
    future_reward_panel = plan_forecast_craft(plan,
                        planning_year + 1, 1, 2,
                        df_time, df_params, df_curve, df_adjust_grid
                        )[0]
    stacked_panel = pd.concat([base_reward_panel, plan_reward_panel, future_reward_panel], axis=0)
    cumu_reward = stacked_panel[target_period].values.sum()
    info[f'total {target} in target period'] = int(np.round(cumu_reward,0))

    info['cpa'] = np.round(info['plan period spend'] / plan_driven_total_reward, 1)
    info['roas'] =  np.round(price * plan_driven_total_reward / info['plan period spend'], 2)

    # MCPA, MROAS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    plan_agg_mc = []
    plan_agg_minc = []
    for X in plan.columns[1:]:
        res = marginal_computations(X, plan, planning_year, planning_period, price, df_time, df_params, df_curve, df_adjust_grid)
        # print(X)
        # print(res[0])
        # print("")
        plan_agg_mc.append(res[-1][0])
        plan_agg_minc.append(res[-1][1]) 

    info['mcpa'] = np.round(sum(plan_agg_mc) / sum(plan_agg_minc), 1)
    info['mroas'] =  np.round(price * sum(plan_agg_minc) / sum(plan_agg_mc), 1)

    # ========================================================================================================================
    # Media Level Summary
    # ========================================================================================================================
    media_dash = {
        "Media": [],
        'Spend': [],
        f'Total {target}':[],
        f'Target period {target}': [],
        'CPA': [], 
        'MCPA':[],
        'ROAS': [],
        'MROAS': [],
    }
    for x in media_list:
        media_dash['Media'].append(x)
        spendX = plan[plan.FIS_MO_NB.isin(planning_period)][x].sum()
        media_dash['Spend'].append(np.round(spendX, 1))
        # Media driven attendance, CPT, ROA
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        rewardX = compute_reward_X(x, plan, planning_year, df_time, df_params, df_curve, df_adjust_grid)[0]
        rewardX = rewardX[[f"Month_{x}" for x in planning_period]].sum(axis=1)
        rewardX_CF = rewardX.values.sum() # Current + future
        media_dash[f'Total {target}'].append(int(np.round(rewardX_CF, 0)))
        rewardX = rewardX[[x - 52 for x in target_weeks]].sum()
        media_dash[f'Target period {target}'].append(int(np.round(rewardX, 0)))
        media_dash['CPA'].append(np.round(spendX / rewardX_CF, 1))
        media_dash['ROAS'].append(np.round(price * rewardX_CF / spendX, 1))
        # MROAS, MCPA
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        res = marginal_computations(x, plan, planning_year, planning_period, price, df_time, df_params, df_curve, df_adjust_grid)
        media_dash['MCPA'].append(np.round(res[0], 1))
        media_dash['MROAS'].append(np.round(res[1], 1))

    df_media_dash = pd.DataFrame(media_dash['Media'], columns=['Media'])
    for key, value in media_dash.items():
        if key != 'Media':
            df_media_dash[key] = value 
    df_media_dash['Media'].replace(media_mapping, inplace=True)


    # ========================================================================================================================
    # Plan forecast
    # ========================================================================================================================
    shard_base = plan_forecast_craft(
        spend_data = df_base, 
        planning_year = planning_year - 1, 
        lead_years = 0,
        lag_years = 3,
        df_time = df_time, 
        df_params = df_params, 
        df_curve = df_curve, 
        df_adjust_grid= df_adjust_grid  
    )   

    shard_plan = plan_forecast_craft(
        spend_data = plan, 
        planning_year = planning_year, 
        lead_years = 1,
        lag_years = 2,
        df_time = df_time, 
        df_params = df_params, 
        df_curve = df_curve,
        df_adjust_grid= df_adjust_grid 
    )

    shard_future = plan_forecast_craft(
        spend_data = plan, 
        planning_year = planning_year + 1, 
        lead_years = 2,
        lag_years = 1,
        df_time = df_time, 
        df_params = df_params, 
        df_curve = df_curve, 
        df_adjust_grid= df_adjust_grid
    )


    forecast_monthly = pd.concat([shard_base[0], shard_plan[0]], axis=0)
    forecast_monthly = forecast_table_summarizer(forecast_monthly, target)
    forecast_monthly.iloc[:, 2:] = forecast_monthly.iloc[:, 2:].astype(float).astype(int)

    forecast_quarterly = pd.concat([shard_base[1], shard_plan[1]], axis=0)
    forecast_quarterly = forecast_table_summarizer(forecast_quarterly, target)
    forecast_quarterly.iloc[:, 2:] = forecast_quarterly.iloc[:, 2:].astype(float).astype(int)


    # ========================================================================================================================
    # Results wrapper
    # ========================================================================================================================
    res = {
        'aggregate_info': info,
        'media_dash': df_media_dash,
        'forecasts': [forecast_monthly, forecast_quarterly]
    }
    
    return res



def reporter(spend_plans, planning_year, planning_period, target_period, target_weeks,
             price, currency_symbol, threshold,
             media_list, media_labels, media_mapping,
             df_base, df_params, df_time, df_curve,
             target
             ):
    """
    Generate a comprehensive report comparing two spending plans, including summary statistics, media-level comparisons, and visualizations.
    Args:
        spend_plans (list): List of two DataFrames, each representing a spending plan.
        planning_year (int): The fiscal year of the current spending plan.
        planning_period (list): List of fiscal month numbers for the planning period, e.g. [1, 2, ..., 12].
        target_period (list): list of strings representing target weeks, e.g. ['2025 Jul', '2025 'Aug', ..., '2026 Dec'].
        target_weeks (list): list of index of target weeks in df_time_filtered (defined previously), e.g. [90, 91, 92, ..., 104]
        price (float): Unit price for attendance.
        currency_symbol (str): Symbol for currency formatting, e.g. "$".
        threshold (float): number for adjusting the month-to-year. 
        media_list (list): List of media names in the plans.
        media_labels (list): List of media labels for display.
        media_mapping (dict): Mapping of media names to their labels.
        df_base (pd.DataFrame): Base data DataFrame.
        df_params (pd.DataFrame): Parameters DataFrame.
        df_time (pd.DataFrame): Time DataFrame.
        df_curve (pd.DataFrame): Curve DataFrame.
        df_adjust_grids (pd.DataFrame): pair of Dataframes storing month-to-year adjust parameter under different rules
    """
    adjust_grid0 = multiplier_grid(spend_plans[0], df_base, threshold)
    adjust_grid1 = multiplier_grid(spend_plans[1], df_base, threshold)


    res0 =  plan_wrapper(spend_plans[0], planning_year, planning_period, target_period, target_weeks, 
                         price, media_list, media_labels, media_mapping, 
                         df_base, df_params, df_time, df_curve, adjust_grid0, target
                         )
    res1 =  plan_wrapper(spend_plans[1], planning_year, planning_period, target_period, target_weeks, 
                         price, media_list, media_labels, media_mapping, 
                         df_base, df_params, df_time, df_curve, adjust_grid1, target
                         )
    agg0, media0, forecasts0 = res0['aggregate_info'], res0['media_dash'], res0['forecasts']
    agg1, media1, forecasts1 = res1['aggregate_info'], res1['media_dash'], res1['forecasts']

    # Summary table
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    shard0 = pd.DataFrame([agg0]).T.reset_index()
    shard0.columns = [' ', 'Original']
    shard1 = pd.DataFrame([agg1]).T.reset_index()
    shard1.columns = [' ', 'Optimized']

    summary = pd.merge(shard0, shard1, on=' ') 
    summary['Change(%)'] = np.round((summary['Optimized'] - summary['Original']) / summary['Original'] * 100, 1)

    # Media level comparison
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    metrics1 = ['Spend', 'MROAS']
    metrics2 = [f'Target period {target}', 'CPA']

    plot1 = comparison_plot(
        media_labels=media_labels,
        source_tables=[media0, media1],
        metrics= metrics1,
        names= ['Original', 'Optimized'],
        colors= ["#91D2E2", '#375174'],
        currency_symbol= currency_symbol,
        title= f"{metrics1[0]} & {metrics1[1]} Media Level Comparison"
    )

    plot2 = comparison_plot(
        media_labels=media_labels,
        source_tables=[media0, media1],
        metrics= metrics2,
        names= ['Original', 'Optimized'],
        colors= ["#DDB87D", "#9C5813"],
        currency_symbol= currency_symbol,
        title= f"{metrics2[0] } & {metrics2[1]} Media Level Comparison"
    )

    crafts = {
        'plans': spend_plans,
        'summary': summary,
        'plan_summary': [agg0, agg1],
        'media_dash' : [media0, media1],
        'plots': [plot1, plot2],
        'forecasts': {
            'plan1': forecasts0,
            'plan2': forecasts1
        }
    }

    return crafts