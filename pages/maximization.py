import streamlit as st
import pandas as pd
import numpy as np
import io

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")


# Loading Dependencies
# =========================================================================
from library.utilities import global_names, crafting_time, revise_spend, split_months_indices
from library.scenario import multiplier_grid, compute_reward_X, compute_plan_reward, plan_forecast_craft, forecast_table_summarizer, comparison_plot, plan_wrapper, reporter
from library.optimization import compute_target_reward, maximizer

# Loading macro files
# =========================================================================
model_versions = pd.read_csv('data/model_versions.csv')
media_mapping_file = pd.read_csv('data/media_label.csv')
# time_ref  = pd.read_csv('data/DT_snowflake.csv') 
df_time = crafting_time(
    raw_time_file= 'data/DT_snowflake.csv',
    planning_years=[2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
    )
months_abbv = global_names['months_abbv']
months_full = global_names['months_full']


def show_maximization():
    st.write("")
    st.write("")
    # ==================================================================================================
    # State refreshments
    # ==================================================================================================
    # 1) Reset session keys 
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if 'maximizer_page_loaded' not in st.session_state:
        for key in ['maximizer_region_validated', 'maximizer_region', 'maximizer_region_code']:
            st.session_state.pop(key, None)
        st.session_state['maximizer_page_loaded'] = True

    # 2) Reset session keys in other pages
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    st.session_state['refresh_scenario'] = "Yes"
    st.session_state['refresh_minimizer'] = "Yes"
    st.session_state["scenario_computed"] = False

    # 3) Clear minimizer results if they exist when entering maximization page
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if 'minimizer_done' in st.session_state:
        st.session_state['minimizer_done'] = False
    if 'minimizer_results' in st.session_state:
        st.session_state.pop('minimizer_results', None)

    # ==================================================================================================
    # Initialize session state variables
    # ==================================================================================================
    if 'maximizer_done' not in st.session_state:
        st.session_state['maximizer_done'] = False
    # Password validation
    if 'maximizer_region_validated' not in st.session_state:
        st.session_state['maximizer_region_validated'] = "Not Validated"
    if 'maximizer_region' not in st.session_state:
        st.session_state['maximizer_region'] = ""
    if 'refresh_maximizer' not in st.session_state:
        st.session_state['refresh_maximizer'] = "No"


    whitespace = 15
    list_tabs = "Input Tab", "Output Tab"
    tab1, tab2 = st.tabs([s.center(whitespace,"\u2001") for s in list_tabs])
    # ==================================================================================================
    # User Interface - Inputs
    # ==================================================================================================
    with tab1:
        # 0) Check if tab has changed and reset validation if needed
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        current_tab = "No"
        if st.session_state['refresh_maximizer'] != current_tab:
            st.session_state['maximizer_region_validated'] = "Not Validated"
            st.session_state['maximizer_region'] = ""
            st.session_state['refresh_maximizer'] = current_tab

        # 1) Prompt user to provide region code
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        st.write("Please enter the region password")
        col1, col2 = st.columns([5, 5])
        with col1:
            container = st.container(border = True)
            with container:
                region_code_maximizer = st.text_input("placeholder",
                                                    label_visibility= "collapsed",
                                                    key="scenario_password_input")
        st.session_state['maximizer_region_code'] = region_code_maximizer
        with col2:
            st.write("")

        # Validate password
        if region_code_maximizer:
            # Assuming model_versions is a DataFrame with 'password' and 'maximizer_region' columns
            check_maximizer = model_versions.loc[model_versions.password == region_code_maximizer, 'region'].values
            
            if len(check_maximizer) == 1:
                st.session_state['maximizer_region_validated'] = "Validated"
                st.session_state['maximizer_region'] = check_maximizer[0]
            else:
                st.session_state['maximizer_region_validated'] = "Not Validated"

        # Display messages
        if st.session_state['maximizer_region_validated'] == "Not Validated" and region_code_maximizer:
            st.error("Please enter the correct region password to proceed")

        elif st.session_state['maximizer_region_validated'] == "Validated":
            # 2) Load region-level parameters and files
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            region = st.session_state['maximizer_region']
            file_params = 'data/' + region + "/input_300pct.csv"
            file_curve = 'data/' + region + "/input_mediaTiming.csv"
            file_base = 'data/' + region + "/input_base.csv"    

            media_mapping = media_mapping_file[media_mapping_file.region == region].set_index('media_code').to_dict()['media_label']
            media_mapping_inverse = {value: key for key, value in media_mapping.items()}
            media_labels = list(media_mapping.values())

            mmm_year = model_versions.loc[model_versions.region == region, 'update'].values[0]
            adjust_ratio = model_versions.loc[model_versions.region == region, 'adjust'].values[0]
            price = model_versions.loc[model_versions.region == region, 'price'].values[0]
            currency = model_versions.loc[model_versions.region == region, 'currency'].values[0]
            target_var = model_versions.loc[model_versions.region == region, 'target'].values[0]

            message = f"🤵🏾‍♀️🎙️ {region} scenarios will be based on model results for fiscal year {mmm_year}"
            st.markdown(
                f"<p style='font-size: 6px; color: #4e98ff'>{message}</p>",
                unsafe_allow_html=True
            )

            df_base = pd.read_csv(file_base)
            df_base = df_base[~df_base.Media.isna()] # In case contains empty rows
            df_base = df_base.T.iloc[1:, :]
            df_base.columns = media_mapping.keys()
            df_base.reset_index(inplace=True)
            df_base.rename(columns={'index': 'FIS_MO_NB'}, inplace=True)
            df_base['FIS_MO_NB'] = np.arange(1, 13)

            df_curve  = pd.read_csv(file_curve)
            media_list = df_curve.columns.tolist()
            filler = []
            for x in df_curve.columns:
                shard = np.zeros(104)
                values = df_curve[x].values
                shard[:len(values)] = values
                filler.append(shard)
            df_curve = pd.DataFrame(np.array(filler).T, columns=df_curve.columns)
            df_curve = df_curve.loc[~(df_curve == 0).all(axis=1)]
            df_curve.fillna(0, inplace=True)


            df_params  = pd.read_csv(file_params)
            df_params.columns = [x.replace("TlncT", 'TIncT') for x in df_params.columns]
            names = list(df_params.columns)
            names2 = [s.replace("FABING", "ING") for s in names]
            names2 = [s.replace("DIS_BAN", "BAN") for s in names2]
            names2 = [s.replace("DIS_AFF", "AFF") for s in names2]
            df_params.columns = names2

            # 3) User select {planning year, planning months} 
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            col1, col2, col3 = st.columns([2, 1, 7])
            with col1:
                planning_year = st.number_input(
                    "Planning Year",
                    value = "min", 
                    min_value = mmm_year + 1,
                )
            with col3:
                start_month, end_month = st.select_slider(
                    "Planning Period",
                    options= months_abbv,
                    value=("Oct", "Sep"),
                )
            planning_period = [
                f"{planning_year} {start_month}",
                f"{planning_year} {end_month}",
            ]

            # 4) User select {target period} 
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            col1, col2, col3 = st.columns([2, 1, 7])
            with col1:
                st.write("")
            with col3:
                st.write("")
                array_year = [planning_year] * 12 + [planning_year + 1] * 12
                slideBar_values = [f"{planning_year} {month}" for month in months_abbv] + [f"{planning_year + 1} {month}" for month in months_abbv]
                planning_start = str(planning_year) + " " + start_month
                index_start = slideBar_values.index(planning_start)
                slideBar_values = slideBar_values[index_start:] 
                
                target_start, target_end = st.select_slider(
                        "Target Period",
                        options= slideBar_values,
                        value=(slideBar_values[0], slideBar_values[-1]),
                    )
                target_period = [target_start, target_end]

            # 5) Compute planning weeks and target weeks, outputs:
            #    1. df_time_filtered: DataFrame with with 3 relevant years
            #    2. planning_weeks: list of indices for planning weeks in df_time_filtered
            #    3. target_weeks: list of indices for target weeks in df_time_filtered
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>           
            timeline = []
            for yr in [planning_year - 1, planning_year, planning_year + 1]:
                timeline.append([f"{yr} {month}" for month in months_abbv])
            timeline = list(np.array(timeline).flatten())

            planning_start = timeline.index(planning_period[0])
            planning_end = timeline.index(planning_period[1]) 
            target_start = timeline.index(target_period[0])
            target_end = timeline.index(target_period[1])

            planning_months = timeline[planning_start:planning_end+1]
            target_months = timeline[target_start:target_end+1]

            month_label = {}
            for i, item in enumerate(months_abbv):
                month_label[i + 1] = item
            month_label_reverse = {v: k for k, v in month_label.items()}

            df_time_filtered = df_time[df_time['FIS_YR_NB'].isin([planning_year - 1, planning_year, planning_year + 1])].reset_index(drop=True)
            df_time_filtered['month_label'] = df_time_filtered.FIS_MO_NB.map(month_label)
            df_time_filtered['year_month'] = df_time_filtered.apply(lambda x: f"{x['FIS_YR_NB']} {x['month_label']}", axis=1)

            planning_weeks = df_time_filtered[df_time_filtered['year_month'].isin(planning_months)].index.tolist()
            target_weeks = df_time_filtered[df_time_filtered['year_month'].isin(target_months)].index.tolist() 
            # Convert months back to fiscal mo number
            planning_months = [month_label_reverse[month.split()[-1]] for month in planning_months]


            # 6) User input 3: choosing baseline spending, 
            #    using MMM year as default
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            st.write("")  
            st.write("Please enter the initial spending plan")

            spend_blueprint = df_base.T.iloc[1:,:].reset_index()
            spend_blueprint.columns = ['Media'] + months_full
            for x in spend_blueprint.columns[1:]:
                spend_blueprint[x] = spend_blueprint[x].astype(float).round(0)
            spend_blueprint['Media'] = spend_blueprint['Media'].replace(media_mapping)

            num_cols = [c for c in spend_blueprint.columns if c != "Media"]
            spend_blueprint[num_cols] = spend_blueprint[num_cols].astype("string")

            col_cfg = {
                "Media": st.column_config.Column(disabled=True),
                **{c: st.column_config.TextColumn() for c in num_cols},
            }

            container = st.container()
            with container:
                num_rows = spend_blueprint.shape[0]
                spend_plan = st.data_editor(
                    spend_blueprint,
                    height=(num_rows + 1) * 35 + 3,
                    disabled=["Media"],
                    hide_index=True,
                    column_config=col_cfg,
                    column_order=['Media'] + months_full,   # keep stable order
                    key="spend_editor",
                )

            # --- Sanitize pasted values to numbers (commas, NBSP, thin spaces, etc.)
            def _to_num(x):
                s = "" if x is None else str(x)
                s = (s.replace(",", "")
                    .replace(" ", "")
                    .replace("\u00A0", "")   # NBSP
                    .replace("\u202F", "")   # narrow NBSP (common in desktop Excel)
                    .replace("\u2009", "")   # thin space
                    .strip())
                return pd.to_numeric(s, errors="coerce")

            for c in num_cols:
                spend_plan[c] = spend_plan[c].map(_to_num)

            # optional: fill NaNs from bad cells or leave as NaN
            spend_plan[num_cols] = spend_plan[num_cols].fillna(0).round(0)

            # --- Your original post-processing
            spend_plan['Media'] = spend_plan['Media'].replace(media_mapping_inverse)
            spend_plan = spend_plan.T
            spend_plan.columns = spend_plan.iloc[0]
            spend_plan = spend_plan.iloc[1:].reset_index()
            spend_plan.rename(columns={'index': 'FIS_MO_NB'}, inplace=True)
            spend_plan['FIS_MO_NB'] = spend_plan['FIS_MO_NB'].apply(lambda x: months_full.index(x) + 1)
            df_future = spend_plan.copy()


            # 6.5) Compute and display media-driven attendance given current spend plan
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            default_grid = multiplier_grid(spend_plan, df_base, adjust_ratio)
            mo_base = plan_forecast_craft(df_base, planning_year - 1, 0, 3, 
                                          df_time, df_params, df_curve, default_grid
                                          )[0]
            mo_scnr1 = plan_forecast_craft(spend_plan, planning_year, 1, 2, 
                                          df_time, df_params, df_curve, default_grid
                                          )[0]
            mo_future = plan_forecast_craft(df_future, planning_year + 1, 2, 1, 
                                            df_time, df_params, df_curve, default_grid
                                          )[0]
            forecast_craft_mo_s1 = pd.concat([mo_base, mo_scnr1, mo_future], axis = 0)
            forecast_craft_mo_s1.iloc[:, 1:] = forecast_craft_mo_s1.iloc[:, 1:].round(1)
            forecast_craft_mo_s1 = forecast_table_summarizer(forecast_craft_mo_s1, target_var)


            current_gain = forecast_craft_mo_s1[target_months].iloc[0, :].values.astype(float).sum()
            current_gain = int(np.round(current_gain, 0))
            message = f"{target_var} from current plan = {current_gain}"
            st.markdown(
                f"<p style='font-size: 6px; color: #4e98ff; font-style: italic;'>{message}</p>",
                unsafe_allow_html=True
            )


            # 7) Choosing media bounds
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            st.write("") 
            st.write("Please choose adjusted bounds for each media channel as percentage.")
            df_bounds = pd.DataFrame({
                'Media' : media_list,
                'Lower Bound (%)' : 80,
                'Upper Bound (%)' : 120
            })

            df_bounds_user = df_bounds.copy()
            df_bounds_user['Media'] = df_bounds_user['Media'].replace(media_mapping)


            container = st.container()
            with container:
                num_rows = df_bounds_user.shape[0]
                df_bounds_user = st.data_editor(
                    df_bounds_user,
                    height = (num_rows + 1) * 35 + 3,    
                    hide_index = True            
                ) 
            
            df_bounds_coded = df_bounds_user.copy()
            df_bounds_coded['Media'] = df_bounds_coded['Media'].replace(media_mapping_inverse)
            df_bounds_coded.columns = ['Media', 'LB', 'UB']
            for x in df_bounds_coded.columns[1:]:
                df_bounds_coded[x] = df_bounds_coded[x].astype(float) / 100

            st.write("")
            st.write("")
            st.write("")
            st.divider()

            # *********************************************************************************************
            # Runnning the optimizer
            # *********************************************************************************************
            if st.button("Let's begin!"):
                with st.spinner("I'm working on it ..."):
                    final_spend_plan, allocation_rank = maximizer(
                        media_list = media_list,
                        spend_plan = spend_plan,
                        df_base = df_base,
                        df_time_filtered = df_time_filtered,
                        df_params = df_params,
                        df_curve = df_curve,
                        df_bounds = df_bounds_coded,
                        multiplier_threshold = adjust_ratio,
                        planning_year = planning_year,
                        planning_months = planning_months,
                        planning_weeks = planning_weeks,
                        target_weeks = target_weeks
                    )

                    optimization_artifacts = reporter(
                        spend_plans=[spend_plan, final_spend_plan],
                        planning_year=planning_year,
                        planning_period=planning_months,
                        target_period=target_months,
                        target_weeks=target_weeks,
                        price= price,
                        currency_symbol= currency,
                        threshold= adjust_ratio,
                        media_list=media_list,
                        media_labels=media_labels,
                        media_mapping=media_mapping,
                        df_base = df_base,
                        df_params = df_params,
                        df_time = df_time,
                        df_curve = df_curve,
                        target = target_var
                    ) 

                st.success("Optimization performed successfully! Please check the results in the output tab 👉")
                st.session_state['maximizer_artifacts'] = optimization_artifacts
                st.session_state["maximizer_done"] = True

    # ==================================================================================================
    # User Interface - Outputs
    # ==================================================================================================
    with tab2:
        scenario_status = st.session_state['maximizer_done']
        if scenario_status:
            artifacts = st.session_state['maximizer_artifacts']
            st.write("")
            st.write("")
            viewing = st.selectbox("Select result format to view",['Aggregate Summary', 'Optimized Spend', 'Detailed Summary by Media', 'Increment Forecast'])

            # Tab 1: Aggregate Summary
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if viewing == 'Aggregate Summary':

                # --------- Verbal Summary ---------------
                summary = artifacts['summary']
                summary['Change'] = summary.Optimized - summary.Original
                summary = summary[[' ', 'Original', 'Optimized', 'Change', 'Change(%)']]
                nrows = summary.shape[0]
                ncols = summary.shape[1]
                attendance_change = summary["Change(%)"].values[1] 
                color_attendance_change = f'<span style="color:blue">{attendance_change}%</span>'
                roas_change = summary["Change(%)"].values[-3]
                color_roas_change = f'<span style="color:blue">{roas_change}%</span>'
                # Verbal summary draft
                verbal_summary1 = f"Now this is nice. With the optimized spend plan, we can use the same total budget but achieve {color_attendance_change} more {target_var} than the original plan. That's {color_roas_change} increase in return on ad spend!"
                verbal_summary2 = "See below for the aggregate level details or use the dropdown menu to explore the detailed summary by media."

                # Verbal Summary
                col1, spacing_col, col2 = st.columns([5, 1, 5]) 
                with col1:
                    st.markdown(f'<p style="font-size: 8px;">{verbal_summary1}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p style="font-size: 8px;">{verbal_summary2}</p>', unsafe_allow_html=True)
                    container = st.container()
                    with container:
                        st.dataframe(summary, 
                                    height = (nrows + 1) * 35 + 3, width= (ncols + 1) * 200 + 3,
                                    hide_index=True)
                with col2:
                    st.image("static_files/images/soul-joe-mr-mittens3.png", width= 350)


            # 2) Tab 2: Optimized Spend
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if viewing == "Optimized Spend":
                # Table
                optimized_spend = artifacts['plans'][1]
                optimized_spend = optimized_spend.T.iloc[1:, :]
                optimized_spend.columns = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 
                                           'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'] 
                optimized_spend['Media'] = media_labels
                shard1 = optimized_spend['Media']
                shard2 = optimized_spend.drop(columns = ['Media'])
                optimized_spend = pd.concat([shard1, shard2], axis=1)
                nrows = optimized_spend.shape[0]

                container = st.container()
                with container:
                    st.dataframe(optimized_spend, 
                                height = (nrows + 1) * 35 + 3, 
                                hide_index=True)
                    
                # Plots
                st.plotly_chart(artifacts['plots'][0])
                st.plotly_chart(artifacts['plots'][1])

            # 3) Tab 3: Detailed Summary by Media
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if viewing == 'Detailed Summary by Media':
                dash_media = artifacts['media_dash']
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("**Original Spend**")
                    table = dash_media[0]
                    nrows = table.shape[0]
                    container = st.container()
                    with container:
                        st.dataframe(table, 
                                    height = (nrows + 1) * 35 + 3, 
                                    hide_index=True)
                with col2:
                    st.write("**Optimized Spend**")
                    table = dash_media[1]
                    nrows = table.shape[0]
                    container = st.container()
                with container:
                    st.dataframe(table, 
                                height = (nrows + 1) * 35 + 3, 
                                hide_index=True)
                    
            # 4) Tab 4: Increment Forecast
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if viewing == 'Increment Forecast':
                col1, col2 = st.columns(2)
                with col1:
                    scenario = st.radio("", ['Original', 'Optimized'])
                with col2:
                    timeframe  = st.radio("", ['Monthly', 'Quarterly'])

                if (scenario == 'Original') & (timeframe == 'Monthly'):
                    table = artifacts['forecasts']['plan1'][0]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    table['Spend'] = [str(x).split('.')[0] for x in table['Spend'].values]
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3,  hide_index=True)

                if (scenario == 'Original') & (timeframe == 'Quarterly'):
                    table = artifacts['forecasts']['plan1'][1]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    table['Spend'] = [str(x).split('.')[0] for x in table['Spend'].values]
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)

                if (scenario == 'Optimized') & (timeframe == 'Monthly'):
                    table = artifacts['forecasts']['plan2'][0]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    table['Spend'] = [str(x).split('.')[0] for x in table['Spend'].values]
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)

                if (scenario == 'Optimized') & (timeframe == 'Quarterly'):
                    table = artifacts['forecasts']['plan2'][1]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    table['Spend'] = [str(x).split('.')[0] for x in table['Spend'].values]
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)



        else:
            st.write("Please complete the steps in the Input Tab and run optimization")