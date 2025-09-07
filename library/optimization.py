import pandas as pd
import numpy as np 
import time
from datetime import date

import warnings
warnings.filterwarnings("ignore") 
from library.utilities import global_names, revise_spend
from library.scenario import multiplier_grid, compute_reward_X, compute_plan_reward, plan_forecast_craft, forecast_table_summarizer, comparison_plot

months_abbv = global_names['months_abbv']


def compute_target_reward(spend_data, planning_year, planning_months, lead_years, lag_years, target_weeks,
                          df_time, df_params, df_curve, df_adjust_grid):
    """
    Computing reward for a specific target period based on the spend data and planning parameters.
    Similar arguments as compute_plan_reward, with 2 extra arguments:
        1) `target_weeks` = e.g. [104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
          index of weeks in df_time[dftime.FIS_YR_NB.between(planning_year - 1, planning_year + 1)]

        2) `planning_months` = e.g. [7, 8, 9], 
          the fiscal month number for planning months that user specified in the UI

    """
    res = compute_plan_reward(
        spend_data = spend_data,
        planning_year= planning_year, 
        lead_years = lead_years,  
        lag_years = lag_years,
        df_time= df_time,
        df_params= df_params,
        df_curve= df_curve,
        df_adjust_grid = df_adjust_grid
    )
    current_reward = 0
    for x in planning_months:
        current_reward += res[x-1][target_weeks].sum()
        
    return current_reward








def minimizer(
        reward_goal, media_list,
        base_spend_plan, spend_plan, future_spend_plan,
        df_base, df_time, df_time_filtered, df_params, df_curve, df_bounds, multiplier_threshold,
        planning_year, planning_months, planning_weeks, target_weeks
):
    df_adjust_grid = multiplier_grid(spend_plan, df_base, multiplier_threshold)

    # Step 0: Set spendings in the spending month to 0
    # ================================================================================================================
    spend_plan_wiped = spend_plan.copy()
    spend_plan_wiped.loc[spend_plan_wiped['FIS_MO_NB'].isin(planning_months), spend_plan_wiped.columns[1:]] = 0


    # Step 1: For each {media, month}, compute the potential gain percentgae (due to media timing)
    # ================================================================================================================
    result_data = []
    for month in planning_months:
        planning_start = df_time_filtered[(df_time_filtered['FIS_YR_NB'] == planning_year) & (df_time_filtered['FIS_MO_NB'] == month)].index.values[0]
        impact_start = target_weeks[0]
        if impact_start >  planning_start:
            skipping_weeks = impact_start - planning_start
            cumu_start = skipping_weeks+1
            cumu_end = skipping_weeks + len(target_weeks) - 1
            cumulative_indices = df_curve.iloc[cumu_start : cumu_end, :].sum()
            result_data.append(cumulative_indices.tolist())
        if impact_start <= planning_start:
            cumu_start = 0
            cumu_end = target_weeks[-1] - planning_start + 1
            cumulative_indices = df_curve.iloc[cumu_start : cumu_end, :].sum()
            result_data.append(cumulative_indices.tolist())
    # Now we have cumulative impact during the target period from each {media, planning month}
    cumu_timing_df = pd.DataFrame(result_data, index=planning_months)
    cumu_timing_df.columns = media_list 


    # Step 2: Flatten media timing index, media names, and media spend
    # ================================================================================================================
    array_cumu_timing = cumu_timing_df.values.flatten()
    array_media_entries = np.tile(media_list, len(planning_months))
    spend0 = []
    for month in planning_months:
        spend0.extend(spend_plan[spend_plan['FIS_MO_NB'] == month].iloc[:, 1:].values.flatten())
    spend0 = np.array(spend0)

    # Step 3: Form the lower bound array and make the array of multipliers at {month, media} level
    # ================================================================================================================
    array_lb = []
    for spend, media in zip(spend0, array_media_entries):
        lb_pct = df_bounds.loc[df_bounds['Media'] == media, 'LB'].values[0]
        ub_pct = df_bounds.loc[df_bounds['Media'] == media, 'UB'].values[0]
        
        array_lb.append(spend * lb_pct)
    array_lb = np.array(array_lb)

    spend_lb = spend_plan_wiped.copy()
    for i, month in enumerate(planning_months):
        spend_lb.loc[spend_lb['FIS_MO_NB'] == month, spend_plan.columns[1:]] = array_lb[i*len(spend_plan.columns[1:]):(i+1)*len(spend_plan.columns[1:])]


    # Step 4: Computing the month-to-year multipliers
    #         - Some medias may have 0-spend months, multipliers need to be less than 12 in this case
    # ================================================================================================================
    # multipliers = [] 
    # for x in media_list:
    #     x_sum = spend_lb[x].values.sum()
    #     multipliers.append(len([x for x in spend_lb[x].values if x/x_sum >= (multiplier_threshold / 100)]))

    multipliers = pd.DataFrame({
        'Media': media_list,
        'LB': df_adjust_grid['mty_fixed']
    })
    multipliers = np.tile(multipliers.LB.values, len(planning_months))


    # Step 5: Create the reward panel (i.e. 300pct table for each {media, planning month})
    #         - This table now considers the media timing impact, i.e. even for the same media
    #           and same spend, the MCPT may be different due to the timing
    #         - The multiplier is computed regarding to the planning year though
    # ================================================================================================================
    reward_panels = []
    for i, (media_entry, cumu_timing) in enumerate(zip(array_media_entries, array_cumu_timing)):
        multiplier = multipliers[i]

        # Define variable names
        spend_varname = global_names['prefix']['spend'] + media_entry
        inc_varname = global_names['prefix']['inc'] + media_entry
        cpt_varname = global_names['prefix']['cpt'] + media_entry
        mcpt_varname = global_names['prefix']['mcpt'] + media_entry

        # Create reward_panel for this entry
        reward_panel = df_params[['PCT_Change', spend_varname, inc_varname, cpt_varname, mcpt_varname]].copy()

        # Adjust spend and inc columns
        reward_panel[spend_varname] /= multiplier
        reward_panel[inc_varname] = (reward_panel[inc_varname] / multiplier) * cumu_timing

        # Create CPT column
        reward_panel[cpt_varname] = reward_panel[spend_varname] / reward_panel[inc_varname]

        # Create MCPT columns
        if cumu_timing == 0:
            reward_panel[mcpt_varname] = reward_panel[mcpt_varname] / 0.0001
        else:
            reward_panel[mcpt_varname] = reward_panel[mcpt_varname] / cumu_timing

        # Rename columns
        reward_panel.columns = ['pct', f'S_{i}', f'R_{i}', f'CPT_{i}', f'MCPT_{i}']
        
        reward_panels.append(reward_panel)
        
    # Combine all reward panels and rename to df_params_monthly
    df_params_monthly = pd.concat(reward_panels, axis=1)

    # Keep only one 'pct' column
    df_params_monthly = df_params_monthly.loc[:, ~df_params_monthly.columns.duplicated()]
    df_params_monthly = df_params_monthly.fillna(0)


    # Step 6 : form the upperbound array
    # ================================================================================================================
    array_ub = []
    array_300 = []
    for i, media in enumerate(array_media_entries):
        max_spend = df_params_monthly.loc[df_params_monthly['pct'] == 300, f'S_{i}'].values[0]
        array_300.append(max_spend)
    array_300 = np.array(array_300)
    for spend, media in zip(spend0, array_media_entries):
        ub_pct = df_bounds.loc[df_bounds['Media'] == media, 'UB'].values[0]
        array_ub.append(min(spend * ub_pct, array_300[len(array_ub)]))
    array_ub = np.array(array_ub)


    # Step 7: Initialize the iteration trackers
    # ================================================================================================================
    current_budget = np.sum(array_lb)
    remaining_budget = np.sum(array_ub) - np.sum(array_lb)
    remaining_budget0 = remaining_budget.copy()
    print(f"Benchmark Budget: {np.sum(spend0)}")
    print(f"Current Budget: {current_budget}") 
    print(f"Remaining Budget: {remaining_budget}")

    spend1 = array_lb.copy() # Start with the scenario where each media is at their lowest boundary

    # Create the allocation_rank dataframe
    # This is the main tracker of media saturation, i.e. which {media, month} has lowest MCPT
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    allocation_rank = pd.DataFrame({
        'entry': range(len(spend1)),
        'current_MCPT': np.zeros(len(spend1)),
        'updates' : np.zeros(len(spend1))
    })
    for i, (spend, media) in enumerate(zip(spend1, array_media_entries)):
        # Find the closest spend value in df_params_monthly
        closest_row = df_params_monthly[f'S_{i}'].sub(spend).abs().idxmin()
        # Get the corresponding reward and MCPT values
        mcpt = df_params_monthly.loc[closest_row, f'MCPT_{i}']
        # Update current_reward and allocation_rank
        allocation_rank.loc[i, 'current_MCPT'] = mcpt
    # Sort allocation_rank
    allocation_rank = allocation_rank.sort_values('current_MCPT').reset_index(drop=True)
    entries_at_upper_bound = list(allocation_rank.loc[allocation_rank.current_MCPT == 0, 'entry'].values)
    allocation_rank = allocation_rank[allocation_rank['current_MCPT'] > 0]
    allocation_rank.current_MCPT = allocation_rank.current_MCPT.round(0)

    # Step 8: Compute current reward status
    # ================================================================================================================
    num_months = [x for x in range(1, 13)]

    credit_base_year = compute_target_reward(
        spend_data = df_base,
        planning_year= planning_year - 1,
        lead_years= 0,
        lag_years= 0,
        target_weeks= target_weeks,
        planning_months= num_months,
        df_time = df_time,
        df_params = df_params,
        df_curve = df_curve,
        df_adjust_grid= df_adjust_grid
    )

    revised_plan = revise_spend(spend_plan, spend1, planning_months)
    current_reward = compute_target_reward(
        spend_data = revised_plan,
        planning_year= planning_year,
        lead_years= 1,
        lag_years= 1,
        target_weeks= target_weeks,
        planning_months= planning_months,
        df_time = df_time,
        df_params = df_params,
        df_curve = df_curve,
        df_adjust_grid = df_adjust_grid
    )
    starting_current_reward = current_reward.copy()

    other_months = [m for m in num_months if m not in  planning_months]
    credit_current_year = compute_target_reward(
        spend_data = revised_plan,
        planning_year = planning_year,
        planning_months = other_months,
        lead_years= 1, 
        lag_years= 1,
        target_weeks = target_weeks,
        df_time = df_time,
        df_params = df_params,
        df_curve = df_curve,
        df_adjust_grid = df_adjust_grid
    )

    credit_future_year = compute_target_reward(
        spend_data = spend_plan,
        planning_year= planning_year + 1,
        lead_years= 2,
        lag_years= 0,
        target_weeks= target_weeks,
        planning_months= num_months,
        df_time = df_time,
        df_params = df_params,
        df_curve = df_curve,
        df_adjust_grid = df_adjust_grid
    )


    total_credit = credit_base_year + credit_current_year + credit_future_year
    current_goal = reward_goal - total_credit


    # Function for one iterative step
    # ================================================================================================================
    def update_entry(entry, target_mcpt, spend1, allocation_rank, remaining_budget):
        i = allocation_rank.loc[entry, 'entry']
        # Skip if this entry has already hit upper bound
        if i in entries_at_upper_bound:
            print(f"Skipping Entry {i} ({array_media_entries[i]}) as it has already reached upper bound")
            return spend1, current_reward, remaining_budget  


        print("Allocating to entry", i, "Media", array_media_entries[i])
        current_spend = spend1[i]
        old_mcpt = allocation_rank.loc[entry, 'current_MCPT']
        
        # Find the row with target MCPT
        target_row = df_params_monthly[f'MCPT_{i}'].sub(target_mcpt).abs().idxmin()
        next_row = np.minimum(target_row + 1, df_params_monthly.shape[0] - 1)
        new_spend = df_params_monthly.loc[next_row, f'S_{i}']

        # Check if new_spend exceeds the upper bound
        if new_spend >= array_ub[i]:
            print(f"Cannot allocate to ${new_spend}, adjusting to upper bound ${array_ub[i]}") 
            new_spend = array_ub[i]
            closest_row = df_params_monthly[f'S_{i}'].sub(new_spend).abs().idxmin()
            new_mcpt = df_params_monthly.loc[closest_row, f'MCPT_{i}']
            entries_at_upper_bound.append(i)
            print(f"Entry {i} ({array_media_entries[i]}) has reached its upper bound and will not be allocated any more budget.")
        else:
            new_mcpt = df_params_monthly.loc[next_row, f'MCPT_{i}']

        spend_increase = new_spend - current_spend
        
        if spend_increase > remaining_budget:
            # Insufficient remaining budget
            new_spend = current_spend + remaining_budget
            closest_row = df_params_monthly[f'S_{i}'].sub(new_spend).abs().idxmin()
            new_reward = df_params_monthly.loc[closest_row, f'R_{i}']
            new_mcpt = df_params_monthly.loc[closest_row, f'MCPT_{i}']
            allocated_budget = remaining_budget
            remaining_budget = 0
        else:
            allocated_budget = spend_increase
            remaining_budget -= spend_increase

        spend1[i] = new_spend
        if spend1[i] == array_ub[i]:
            entries_at_upper_bound.append(i)
            print(f"Entry {i} ({array_media_entries[i]}) has reached its upper bound and will not be allocated any more budget.")
        

        allocation_rank.loc[entry, 'current_MCPT'] = new_mcpt
        allocation_rank.loc[entry, 'updates'] += 1

        # Find the media with the target MCPT
        target_entry = allocation_rank[allocation_rank['current_MCPT'] == target_mcpt].iloc[0].name
        target_media_index = allocation_rank.loc[target_entry, 'entry']
        target_media = array_media_entries[target_media_index]


        # Update on reward debt
        revised_spend = revise_spend(spend_plan, spend1, planning_months)
        current_reward = compute_target_reward(
            spend_data = revised_spend,
            planning_year= planning_year,
            lead_years= 1,
            lag_years= 1,
            target_weeks= target_weeks,
            planning_months= planning_months,
            df_time= df_time,
            df_params= df_params,
            df_curve= df_curve,
            df_adjust_grid = df_adjust_grid
        )
        reward_debit = current_goal - current_reward

        # Reporting
        print(f"Media: {array_media_entries[i]} (Entry {i})" )
        print(f"Target MCPT: {target_mcpt:.4f} (from {target_media})")
        print(f"Spend: {current_spend:.2f} >>> {new_spend:.2f}  (allocated {spend_increase:.2f})")
        print(f"MCPT: {old_mcpt:.2f} >>> {new_mcpt:.2f}")
        print(f"Remaining budget: {remaining_budget:.2f}")
        print(f"Current reward: {current_reward:.2f}")
        print("--------------------")
        
        return spend1, allocation_rank, remaining_budget, current_reward


    # Step 9: Main optimization loop
    # ================================================================================================================
    iteration = 1
    while remaining_budget > 0 and current_reward < current_goal and iteration < 10000:
        # print(f"\nIteration {iteration}:")
        allocation_rank['current_MCPT'] = allocation_rank['current_MCPT'].round(0)
        allocation_rank = allocation_rank.sort_values('current_MCPT')

        # Instead of removing entries at upper bound, create a filtered view for allocation decisions
        remaining_entries = allocation_rank[~allocation_rank.entry.isin(entries_at_upper_bound)]
        lowest_mcpt = remaining_entries['current_MCPT'].min() if not remaining_entries.empty else 0
        lowest_mcpt_entries = remaining_entries[remaining_entries['current_MCPT'] == lowest_mcpt]

        # Deciding running case:
        special_case = 0
        if len(remaining_entries) == 0:
            special_case = 1
        if remaining_entries['current_MCPT'].nunique() == 1 and len(remaining_entries) > 1:
            special_case = 2
        if len(remaining_entries) == 1:
            special_case = 3
        if len(lowest_mcpt_entries) > 1 and remaining_entries['current_MCPT'].nunique() > 1:
            special_case = 4

        # Special Case 1: All entries have reached their upper bounds
        # .............................................................................................................
        if special_case == 1:
            print("Running Special Case 1")
            print("All entries have reached their upper bounds. Ending allocation process.")
            break

        # Special Case 2: All entries have the same MCPT
        # .............................................................................................................
        if special_case == 2:
            print("Running Special Case 2")
            print("All remaining entries have the same MCPT. Allocating fixed amount to entry with most room.")

            # Calculate remaining capacity for each entry
            remaining_capacity = []
            for idx in remaining_entries.index:
                i = allocation_rank.loc[idx, 'entry']
                capacity = array_ub[i] - spend1[i]
                remaining_capacity.append({
                    'entry_idx': idx,
                    'entry': i,
                    'capacity': capacity
                })

            # Sort by remaining capacity (descending)
            remaining_capacity = sorted(remaining_capacity, key=lambda x: x['capacity'], reverse=True)

            # Select entry with most capacity
            selected = remaining_capacity[0]
            entry = selected['entry_idx']
            i = selected['entry'] 
            print(f"Selected {array_media_entries[i]} (Entry {i}) with remaining capacity of ${selected['capacity']:.2f}")

            # Cap allocation at 3000 or distance to upper bound
            distance_to_upper = array_ub[i] - spend1[i]
            max_allocation = min(3000, distance_to_upper, remaining_budget)
            print(f"Allocating {max_allocation:.2f} to {array_media_entries[i]} (Entry {i})")

            # Update spend and lookup new reward/MCPT
            current_spend = spend1[i]
            spend1[i] += max_allocation
            remaining_budget -= max_allocation

            # Find closest row in df_params_monthly for the new spend level
            closest_row = df_params_monthly[f'S_{i}'].sub(spend1[i]).abs().idxmin()
            new_mcpt = df_params_monthly.loc[closest_row, f'MCPT_{i}']

            # Update the current spend and current reward
            revised_spend = revise_spend(spend_plan, spend1, planning_months)
            current_reward = compute_target_reward(
                spend_data = revised_spend,
                planning_year= planning_year,
                lead_years= 1,
                lag_years= 1,
                target_weeks= target_weeks,
                planning_months= planning_months,
                df_time= df_time,
                df_params= df_params,
                df_curve= df_curve,
                df_adjust_grid = df_adjust_grid
            )
            reward_debit = current_goal - current_reward

            # Update allocation_rank
            allocation_rank.loc[entry, 'current_MCPT'] = new_mcpt
            allocation_rank.loc[entry, 'updates'] += 1

            if spend1[i] >= array_ub[i]:
                entries_at_upper_bound.append(i)
                print(f"Entry {i} ({array_media_entries[i]}) has reached its upper bound and will not be allocated any more budget.")

        # Special case 3: Single entry remaining
        # ...........................................................................................................
        elif special_case == 3:
            # print("Running special case 3")
            entry = remaining_entries.index[0]
            i = remaining_entries.loc[entry, 'entry']
            current_spend = spend1[i]
            distance_to_upper = array_ub[i] - current_spend
            
            while remaining_budget > 0 and spend1[i] < array_ub[i]:
                quarter_budget = min(remaining_budget / 4, distance_to_upper / 4)
                if quarter_budget < 250:  # Break if allocation becomes too small
                    quarter_budget = remaining_budget
                    
                # Update spend
                spend1[i] += quarter_budget
                remaining_budget -= quarter_budget

                        
                # Update reward and check if target reached
                revised_spend = revise_spend(spend_plan, spend1, planning_months)
                current_reward = compute_target_reward(
                    spend_data = revised_spend,
                    planning_year= planning_year,
                    lead_years= 1,
                    lag_years= 1,
                    target_weeks= target_weeks,
                    planning_months= planning_months,
                    df_time= df_time,
                    df_params= df_params,
                    df_curve= df_curve,
                    df_adjust_grid = df_adjust_grid
                )
                reward_debit = current_goal - current_reward
                
                print(f"Allocated quarter budget ${quarter_budget:.2f} to {array_media_entries[i]} (Entry {i})")
                print(f"Current reward: {current_reward:.2f}, Target: {reward_debit:.2f}")
                
                if current_reward >= reward_debit:
                    # print("Target reached after quarter allocation")
                    break

                # Update distance to upper bound
                distance_to_upper = array_ub[i] - spend1[i]
                
                if spend1[i] >= array_ub[i]:
                    entries_at_upper_bound.append(i)
                    print(f"Entry {i} ({array_media_entries[i]}) has reached its upper bound and will not be allocated any more budget.")
                    break

            # Update MCPT after all allocations
            closest_row = df_params_monthly[f'S_{i}'].sub(spend1[i]).abs().idxmin()
            new_mcpt = df_params_monthly.loc[closest_row, f'MCPT_{i}']
            allocation_rank.loc[entry, 'current_MCPT'] = new_mcpt
            allocation_rank.loc[entry, 'updates'] += 1

        # Special case 4: Multiple entries with the same lowest MCPT
        # ...........................................................................................................
        # Check for multiple entries with the same lowest MCPT
        if special_case == 4:
            print("Running Special Case 4")
            next_mcpt = allocation_rank[allocation_rank['current_MCPT'] > lowest_mcpt]['current_MCPT'].min()
            print(f"Multiple entries with lowest MCPT {lowest_mcpt:.4f}. Updating to next MCPT {next_mcpt:.4f}")
            
            # Calculate required spend for each entry
            spend_requirements = []
            for entry in lowest_mcpt_entries.index:
                i = allocation_rank.loc[entry, 'entry']
                current_spend = spend1[i]
                
                # Find required spend for target MCPT
                target_row = df_params_monthly[f'MCPT_{i}'].sub(next_mcpt).abs().idxmin()
                next_row = np.minimum(target_row + 1, df_params_monthly.shape[0] - 1)
                required_spend = df_params_monthly.loc[next_row, f'S_{i}'] - current_spend
                
                spend_requirements.append({
                    'entry': entry,
                    'required_spend': required_spend
                })

            # Sort by required spend
            spend_requirements = sorted(spend_requirements, key=lambda x: x['required_spend'])
            # print(spend_requirements)
            
            # Process entries in order of required spend
            for req in spend_requirements:
                results = update_entry(req['entry'], next_mcpt, spend1, allocation_rank, remaining_budget)
                spend1 = results[0]
                allocation_rank = results[1]
                remaining_budget = results[2]
                current_reward = results[3]
                if remaining_budget <= 0 or current_reward >= current_goal: 
                    # print("Stopping allocation as goal has been reached or budget depleted.")
                    break


        # Normal Case: Update the entry with lowest MCPT
        # ...........................................................................................................
        if special_case == 0:
            # Normal case: update the entry with the lowest MCPT
            print("Running Normal Case")
            entry = remaining_entries.index[0]
            # Look up next MCPT from full allocation_rank, not just remaining_entries
            entry_mcpt = allocation_rank.loc[entry, 'current_MCPT']
            next_mcpt = allocation_rank[allocation_rank['current_MCPT'] > entry_mcpt]['current_MCPT'].min()
            results = update_entry(entry, next_mcpt, spend1, allocation_rank, remaining_budget)
            spend1 = results[0]
            allocation_rank = results[1]
            remaining_budget = results[2]
            current_reward = results[3]

        iteration += 1
        print("")
        print("")
        print("")


    #  Step 10: Wrapping up the results
    # ================================================================================================================
    print(f"Base credit: {credit_base_year:.2f}, Current credit: {credit_current_year:.2f}, Future credit: {credit_future_year:.2f}")
    print(f"Starting Current Reward: {starting_current_reward:.2f}, Current Reward: {current_reward:.2f}, Current Goal: {current_goal:.2f}")
    print(f"Starting remaining budget: {remaining_budget0}")
    success = 0
    if current_reward >= current_goal:
        success = 1
    final_spend_plan = revise_spend(spend_plan, spend1, planning_months)

    return final_spend_plan, allocation_rank, success











def maximizer(
        media_list,spend_plan, df_base,
        df_time_filtered, df_params, df_curve, df_bounds, multiplier_threshold,
        planning_year, planning_months, planning_weeks, target_weeks
):
    df_adjust_grid = multiplier_grid(spend_plan, df_base, multiplier_threshold)

    # Step 0: Set spendings in the spending month to 0
    # ================================================================================================================
    spend_plan_wiped = spend_plan.copy()
    spend_plan_wiped.loc[spend_plan_wiped['FIS_MO_NB'].isin(planning_months), spend_plan_wiped.columns[1:]] = 0


    # Step 1: For each {media, month}, compute the potential gain percentgae (due to media timing)
    # ================================================================================================================
    result_data = []
    for month in planning_months:
        planning_start = df_time_filtered[(df_time_filtered['FIS_YR_NB'] == planning_year) & (df_time_filtered['FIS_MO_NB'] == month)].index.values[0]
        impact_start = target_weeks[0]
        if impact_start >  planning_start:
            skipping_weeks = impact_start - planning_start
            cumu_start = skipping_weeks+1
            cumu_end = skipping_weeks + len(target_weeks) - 1
            cumulative_indices = df_curve.iloc[cumu_start : cumu_end, :].sum()
            result_data.append(cumulative_indices.tolist())
        if impact_start <= planning_start:
            cumu_start = 0
            cumu_end = target_weeks[-1] - planning_start + 1
            cumulative_indices = df_curve.iloc[cumu_start : cumu_end, :].sum()
            result_data.append(cumulative_indices.tolist())

    # Now we have cumulative impact during the target period from each {media, planning month}
    cumu_timing_df = pd.DataFrame(result_data, index=planning_months)
    cumu_timing_df.columns = media_list  

    # Step 2: Flatten media timing index, media names, and media spend
    # ================================================================================================================
    array_cumu_timing = cumu_timing_df.values.flatten()
    array_media_entries = np.tile(media_list, len(planning_months))
    spend0 = []
    for month in planning_months:
        spend0.extend(spend_plan[spend_plan['FIS_MO_NB'] == month].iloc[:, 1:].values.flatten())
    spend0 = np.array(spend0)

    # Step 3: Form the lower bound array and make the array of multipliers at {month, media} level
    # ================================================================================================================
    array_lb = []
    for spend, media in zip(spend0, array_media_entries):
        lb_pct = df_bounds.loc[df_bounds['Media'] == media, 'LB'].values[0]
        ub_pct = df_bounds.loc[df_bounds['Media'] == media, 'UB'].values[0]
        
        array_lb.append(spend * lb_pct)
    array_lb = np.array(array_lb)

    spend_lb = spend_plan_wiped.copy()
    for i, month in enumerate(planning_months):
        spend_lb.loc[spend_lb['FIS_MO_NB'] == month, spend_plan.columns[1:]] = array_lb[i*len(spend_plan.columns[1:]):(i+1)*len(spend_plan.columns[1:])]


    # Step 4: Computing the month-to-year multipliers
    #         - Some medias may have 0-spend months, multipliers need to be less than 12 in this case
    # ================================================================================================================
    # multipliers = [] 
    # for x in media_list:
    #     x_sum = spend_lb[x].values.sum()
    #     multipliers.append(len([x for x in spend_lb[x].values if x/x_sum >= (multiplier_threshold / 100)]))

    multipliers = pd.DataFrame({
        'Media': media_list,
        'LB': df_adjust_grid['mty_fixed'].values
    })
    multipliers = np.tile(multipliers.LB.values, len(planning_months))


    # Step 5: Create the reward panel (i.e. 300pct table for each {media, planning month})
    #         - This table now considers the media timing impact, i.e. even for the same media
    #           and same spend, the MCPT may be different due to the timing
    #         - The multiplier is computed regarding to the planning year though
    # ================================================================================================================
    reward_panels = []
    for i, (media_entry, cumu_timing) in enumerate(zip(array_media_entries, array_cumu_timing)):
        multiplier = multipliers[i]

        # Define variable names
        spend_varname = global_names['prefix']['spend'] + media_entry
        inc_varname = global_names['prefix']['inc'] + media_entry
        cpt_varname = global_names['prefix']['cpt'] + media_entry
        mcpt_varname = global_names['prefix']['mcpt'] + media_entry

        # Create reward_panel for this entry
        reward_panel = df_params[['PCT_Change', spend_varname, inc_varname, cpt_varname, mcpt_varname]].copy()

        # Adjust spend and inc columns
        reward_panel[spend_varname] /= multiplier
        reward_panel[inc_varname] = (reward_panel[inc_varname] / multiplier) * cumu_timing

        # Create CPT column
        reward_panel[cpt_varname] = reward_panel[spend_varname] / reward_panel[inc_varname]

        # Create MCPT columns
        if cumu_timing == 0:
            reward_panel[mcpt_varname] = reward_panel[mcpt_varname] / 0.0001
        else:
            reward_panel[mcpt_varname] = reward_panel[mcpt_varname] / cumu_timing

        # Rename columns
        reward_panel.columns = ['pct', f'S_{i}', f'R_{i}', f'CPT_{i}', f'MCPT_{i}']
        
        reward_panels.append(reward_panel)
        
    # Combine all reward panels and rename to df_params_monthly
    df_params_monthly = pd.concat(reward_panels, axis=1)

    # Keep only one 'pct' column
    df_params_monthly = df_params_monthly.loc[:, ~df_params_monthly.columns.duplicated()]
    df_params_monthly = df_params_monthly.fillna(0)


    # Step 6 : form the upperbound array
    # ================================================================================================================
    array_ub = []
    array_300 = []
    for i, media in enumerate(array_media_entries):
        max_spend = df_params_monthly.loc[df_params_monthly['pct'] == 300, f'S_{i}'].values[0]
        array_300.append(max_spend)
    array_300 = np.array(array_300)
    for spend, media in zip(spend0, array_media_entries):
        ub_pct = df_bounds.loc[df_bounds['Media'] == media, 'UB'].values[0]
        array_ub.append(min(spend * ub_pct, array_300[len(array_ub)]))
    array_ub = np.array(array_ub)


    # Step 7: Initialize the iteration trackers
    # ================================================================================================================
    current_budget = np.sum(array_lb)
    remaining_budget = np.sum(spend0) - np.sum(array_lb)
    print(f"Benchmark Budget: {np.sum(spend0)}")
    print(f"Current Budget: {current_budget}") 
    print(f"Remaining Budget: {remaining_budget}")

    spend1 = array_lb.copy() # Start with the scenario where each media is at their lowest boundary

    # Create the allocation_rank dataframe
    # This is the main tracker of media saturation, i.e. which {media, month} has lowest MCPT
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    allocation_rank = pd.DataFrame({
        'entry': range(len(spend1)),
        'current_MCPT': np.zeros(len(spend1)),
        'updates' : np.zeros(len(spend1))
    })
    for i, (spend, media) in enumerate(zip(spend1, array_media_entries)):
        # Find the closest spend value in df_params_monthly
        closest_row = df_params_monthly[f'S_{i}'].sub(spend).abs().idxmin()
        # Get the corresponding reward and MCPT values
        mcpt = df_params_monthly.loc[closest_row, f'MCPT_{i}']
        # Update current_reward and allocation_rank
        allocation_rank.loc[i, 'current_MCPT'] = mcpt
    # Sort allocation_rank
    allocation_rank = allocation_rank.sort_values('current_MCPT').reset_index(drop=True)
    entries_at_upper_bound = list(allocation_rank.loc[allocation_rank.current_MCPT == 0, 'entry'].values)
    allocation_rank = allocation_rank[allocation_rank['current_MCPT'] > 0]
    allocation_rank.current_MCPT = allocation_rank.current_MCPT.round(0)


    # Function for one iterative step
    # ================================================================================================================
    def update_entry(entry, target_mcpt, spend1, allocation_rank, remaining_budget):
        i = allocation_rank.loc[entry, 'entry']
        # Skip if this entry has already hit upper bound
        if i in entries_at_upper_bound:
            print(f"Skipping Entry {i} ({array_media_entries[i]}) as it has already reached upper bound")
            return spend1, remaining_budget  # Return both values


        print("Allocating to entry", i, "Media", array_media_entries[i])
        current_spend = spend1[i]
        old_mcpt = allocation_rank.loc[entry, 'current_MCPT']
        
        # Find the row with target MCPT
        target_row = df_params_monthly[f'MCPT_{i}'].sub(target_mcpt).abs().idxmin()
        next_row = np.minimum(target_row + 1, df_params_monthly.shape[0] - 1)
        
        new_spend = df_params_monthly.loc[next_row, f'S_{i}']

        # Check if new_spend exceeds the upper bound
        if new_spend >= array_ub[i]:
            print(f"Cannot allocate to ${new_spend}, adjusting to upper bound ${array_ub[i]}") 
            new_spend = array_ub[i]
            closest_row = df_params_monthly[f'S_{i}'].sub(new_spend).abs().idxmin()
            new_mcpt = df_params_monthly.loc[closest_row, f'MCPT_{i}']
            entries_at_upper_bound.append(i)
            print(f"Entry {i} ({array_media_entries[i]}) has reached its upper bound and will not be allocated any more budget.")
        else:
            new_mcpt = df_params_monthly.loc[next_row, f'MCPT_{i}']

        spend_increase = new_spend - current_spend
        
        if spend_increase > remaining_budget:
            # Insufficient remaining budget
            new_spend = current_spend + remaining_budget
            closest_row = df_params_monthly[f'S_{i}'].sub(new_spend).abs().idxmin()
            new_reward = df_params_monthly.loc[closest_row, f'R_{i}']
            new_mcpt = df_params_monthly.loc[closest_row, f'MCPT_{i}']
            allocated_budget = remaining_budget
            remaining_budget = 0
        else:
            allocated_budget = spend_increase
            remaining_budget -= spend_increase

        spend1[i] = new_spend
        if spend1[i] == array_ub[i]:
            entries_at_upper_bound.append(i)
            print(f"Entry {i} ({array_media_entries[i]}) has reached its upper bound and will not be allocated any more budget.")

        allocation_rank.loc[entry, 'current_MCPT'] = new_mcpt
        allocation_rank.loc[entry, 'updates'] += 1

        # Find the media with the target MCPT
        target_entry = allocation_rank[allocation_rank['current_MCPT'] == target_mcpt].iloc[0].name
        target_media_index = allocation_rank.loc[target_entry, 'entry']
        target_media = array_media_entries[target_media_index]

        print(f"Media: {array_media_entries[i]} (Entry {i})" )
        print(f"Target MCPT: {target_mcpt:.4f} (from {target_media})")
        print(f"Spend: {current_spend:.2f} >>> {new_spend:.2f}  (allocated {spend_increase:.2f})")
        print(f"MCPT: {old_mcpt:.2f} >>> {new_mcpt:.2f}")
        print(f"Remaining budget: {remaining_budget:.2f}")
        print("--------------------")

        return spend1, allocation_rank, remaining_budget
    
    # Step 8: Main optimization loop
    # ================================================================================================================
    iteration = 1
    while remaining_budget > 0 and iteration < 10000:
        print(f"\nIteration {iteration}:")
        allocation_rank['current_MCPT'] = allocation_rank['current_MCPT'].round(0)
        allocation_rank = allocation_rank.sort_values('current_MCPT')

        # Instead of removing entries at upper bound, create a filtered view for allocation decisions
        remaining_entries = allocation_rank[~allocation_rank.entry.isin(entries_at_upper_bound)]
        lowest_mcpt = remaining_entries['current_MCPT'].min() if not remaining_entries.empty else 0
        lowest_mcpt_entries = remaining_entries[remaining_entries['current_MCPT'] == lowest_mcpt]

        # Deciding running case:
        special_case = 0
        if len(remaining_entries) == 0:
            special_case = 1
        if remaining_entries['current_MCPT'].nunique() == 1 and len(remaining_entries) > 1:
            special_case = 2
        if len(remaining_entries) == 1:
            special_case = 3
        if len(lowest_mcpt_entries) > 1 and remaining_entries['current_MCPT'].nunique() > 1:
            special_case = 4

        # Special Case 1: All entries have reached their upper bounds
        # .............................................................................................................
        if special_case == 1:
            print("Running Special Case 1")
            print("All entries have reached their upper bounds. Ending allocation process.")
            break

        # Special Case 2: All entries have the same MCPT
        # .............................................................................................................
        if special_case == 2:
            print("Running Special Case 2")
            print("All remaining entries have the same MCPT. Allocating fixed amount to entry with most room.")

            # Calculate remaining capacity for each entry
            remaining_capacity = []
            for idx in remaining_entries.index:
                i = allocation_rank.loc[idx, 'entry']
                capacity = array_ub[i] - spend1[i]
                remaining_capacity.append({
                    'entry_idx': idx,
                    'entry': i,
                    'capacity': capacity
                })

            # Sort by remaining capacity (descending)
            remaining_capacity = sorted(remaining_capacity, key=lambda x: x['capacity'], reverse=True)

            # Select entry with most capacity
            selected = remaining_capacity[0]
            entry = selected['entry_idx']
            i = selected['entry'] 
            print(f"Selected {array_media_entries[i]} (Entry {i}) with remaining capacity of ${selected['capacity']:.2f}")

            # Cap allocation at 3000 or distance to upper bound
            distance_to_upper = array_ub[i] - spend1[i]
            max_allocation = min(3000, distance_to_upper, remaining_budget)
            print(f"Allocating {max_allocation:.2f} to {array_media_entries[i]} (Entry {i})")

            # Update spend and lookup new reward/MCPT
            current_spend = spend1[i]
            spend1[i] += max_allocation
            remaining_budget -= max_allocation

            # Find closest row in df_params_monthly for the new spend level
            closest_row = df_params_monthly[f'S_{i}'].sub(spend1[i]).abs().idxmin()
            new_mcpt = df_params_monthly.loc[closest_row, f'MCPT_{i}']

            # Update allocation_rank
            allocation_rank.loc[entry, 'current_MCPT'] = new_mcpt
            allocation_rank.loc[entry, 'updates'] += 1

            if spend1[i] >= array_ub[i]:
                entries_at_upper_bound.append(i)
                print(f"Entry {i} ({array_media_entries[i]}) has reached its upper bound and will not be allocated any more budget.")

        # Special case 3: Single entry remaining
        # ...........................................................................................................
        elif special_case == 3:
            print("Running special case 3")
            entry = remaining_entries.index[0]
            i = remaining_entries.loc[entry, 'entry']
            current_spend = spend1[i]
            distance_to_upper = array_ub[i] - current_spend
            
            while remaining_budget > 0 and spend1[i] < array_ub[i]:
                quarter_budget = min(remaining_budget / 4, distance_to_upper / 4)
                if quarter_budget < 250:  # Break if allocation becomes too small
                    quarter_budget = remaining_budget
                    
                spend1[i] += quarter_budget
                remaining_budget -= quarter_budget
                distance_to_upper = array_ub[i] - spend1[i]
                
                if spend1[i] >= array_ub[i]:
                    entries_at_upper_bound.append(i)
                    break

            # Update MCPT after all allocations
            closest_row = df_params_monthly[f'S_{i}'].sub(spend1[i]).abs().idxmin()
            new_mcpt = df_params_monthly.loc[closest_row, f'MCPT_{i}']
            allocation_rank.loc[entry, 'current_MCPT'] = new_mcpt
            allocation_rank.loc[entry, 'updates'] += 1

        # Special case 4: Multiple entries with the same lowest MCPT
        # ...........................................................................................................
        # Check for multiple entries with the same lowest MCPT
        if special_case == 4:
            print("Running Special Case 4")
            next_mcpt = allocation_rank[allocation_rank['current_MCPT'] > lowest_mcpt]['current_MCPT'].min()
            print(f"Multiple entries with lowest MCPT {lowest_mcpt:.4f}. Updating to next MCPT {next_mcpt:.4f}")
            
            # Calculate required spend for each entry
            spend_requirements = []
            for entry in lowest_mcpt_entries.index:
                i = allocation_rank.loc[entry, 'entry']
                current_spend = spend1[i]
                
                # Find required spend for target MCPT
                target_row = df_params_monthly[f'MCPT_{i}'].sub(next_mcpt).abs().idxmin()
                next_row = np.minimum(target_row + 1, df_params_monthly.shape[0] - 1)
                required_spend = df_params_monthly.loc[next_row, f'S_{i}'] - current_spend
                
                spend_requirements.append({
                    'entry': entry,
                    'required_spend': required_spend
                })

            # Sort by required spend
            spend_requirements = sorted(spend_requirements, key=lambda x: x['required_spend'])
            print(spend_requirements)
            
            # Process entries in order of required spend
            for req in spend_requirements:
                results = update_entry(req['entry'], next_mcpt, spend1, allocation_rank, remaining_budget)
                spend1 = results[0]
                allocation_rank = results[1]
                remaining_budget = results[2]
                if remaining_budget <= 0: 
                    print("Stopping allocation as goal has been reached or budget depleted.")
                    break


        # Normal Case: Update the entry with lowest MCPT
        # ...........................................................................................................
        if special_case == 0:
            # Normal case: update the entry with the lowest MCPT
            print("Running Normal Case")
            entry = remaining_entries.index[0]
            # Look up next MCPT from full allocation_rank, not just remaining_entries
            entry_mcpt = allocation_rank.loc[entry, 'current_MCPT']
            next_mcpt = allocation_rank[allocation_rank['current_MCPT'] > entry_mcpt]['current_MCPT'].min()
            results = update_entry(entry, next_mcpt, spend1, allocation_rank, remaining_budget)
            spend1 = results[0]
            allocation_rank = results[1]
            remaining_budget = results[2]

        iteration += 1
        print("")
        print("")
        print("")

    #  Step 9: Wrapping up the results
    # ================================================================================================================
    final_spend_plan = revise_spend(spend_plan, spend1, planning_months)
    
    return final_spend_plan, allocation_rank






