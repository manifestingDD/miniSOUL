import streamlit as st
import pandas as pd
import numpy as np
import io

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")


# Loading Dependencies
# =========================================================================
from library.utilities import global_names, crafting_time, revise_spend, split_months_indices, metric_compare
from library.scenario import multiplier_grid, compute_reward_X, compute_plan_reward, plan_forecast_craft, comparison_plot, plan_wrapper, reporter



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


def show_scenario():
    st.write("")
    st.write("")
    # ==================================================================================================
    # State refreshments
    # ==================================================================================================
    # 1) Reset session keys 
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if 'scenario_page_loaded' not in st.session_state:
        # Clear password related states on first load of the page
        for key in ['scenario_region_validated', 'scenario_region', 'scenario_region_code']:
            st.session_state.pop(key, None)
        st.session_state['scenario_page_loaded'] = True

    # 2) Refresh other functionalities, requiring password again
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    st.session_state['refresh_minimizer'] = "Yes"
    st.session_state['refresh_maximizer'] = "Yes"

    # 3) Clear minimizer & maximizer results if they exist when entering maximization page
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if 'minimizer_done' in st.session_state:
        st.session_state['minimizer_done'] = False
    if 'minimizer_results' in st.session_state:
        st.session_state.pop('minimizer_results', None)
    if 'maximizer_done' in st.session_state:
        st.session_state['maximizer_done'] = False
    if 'maximizer_results' in st.session_state:
        st.session_state.pop('maximizer_results', None)

    # ==================================================================================================
    # Initialize session state variables
    # ==================================================================================================
    if 'scenario_computed' not in st.session_state:
        st.session_state['scenario_computed'] = False
    # Initialize password validation state and add a new state for tracking active tab
    if 'scenario_region_validated' not in st.session_state:
        st.session_state['scenario_region_validated'] = "Not Validated"
    if 'scenario_region' not in st.session_state:
        st.session_state['scenario_region'] = ""
    if 'refresh_scenario' not in st.session_state:
        st.session_state['refresh_scenario'] = "No"


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
        if st.session_state['refresh_scenario'] != current_tab:
            st.session_state['scenario_region_validated'] = "Not Validated"
            st.session_state['scenario_region'] = ""
            st.session_state['refresh_scenario'] = current_tab

        # 1) Prompt user to provide region code
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        st.write("Please enter the region password")
        col1, col2 = st.columns([5, 5])
        with col1:
            container = st.container(border = True)
            with container:
                region_code_scenario = st.text_input("placeholder",
                                                    label_visibility= "collapsed",
                                                    key="scenario_password_input")
        st.session_state['scenario_region_code'] = region_code_scenario
        with col2:
            st.write("")
        
        # Validate password
        if region_code_scenario:
            # Assuming model_versions is a DataFrame with 'password' and 'scenario_region' columns
            check_scenario = model_versions.loc[model_versions.password == region_code_scenario, 'region'].values
            
            if len(check_scenario) == 1:
                st.session_state['scenario_region_validated'] = "Validated"
                st.session_state['scenario_region'] = check_scenario[0]
            else:
                st.session_state['scenario_region_validated'] = "Not Validated"

        # Display messages
        if st.session_state['scenario_region_validated'] == "Not Validated" and region_code_scenario:
            st.error("Please enter the correct region password to proceed")

        elif st.session_state['scenario_region_validated'] == "Validated":
            # 2) Load region-level parameters and files
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            region = st.session_state['scenario_region']
            file_params = 'data/' + region + "/input_300pct.csv"
            file_curve = 'data/' + region + "/input_mediaTiming.csv"
            file_base = 'data/' + region + "/input_base.csv"

            media_mapping = media_mapping_file[media_mapping_file.region == region].set_index('media_code').to_dict()['media_label']
            media_mapping_inverse = {value: key for key, value in media_mapping.items()}
            media_labels = media_mapping.values()

            mmm_year = model_versions.loc[model_versions.region == region, 'update'].values[0]
            adjust_ratio = model_versions.loc[model_versions.region == region, 'adjust'].values[0]
            price = model_versions.loc[model_versions.region == region, 'price'].values[0]
            currency = model_versions.loc[model_versions.region == region, 'currency'].values[0]
            target_var = model_versions.loc[model_versions.region == region, 'target'].values[0]
            st.session_state['scenario_currency'] = currency
            st.session_state['scenario_target'] = target_var

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

            spend_template = pd.read_csv(file_base)
            spend_template['Media'] = media_mapping.values()
            spend_template = spend_template.to_csv(index = False).encode("utf-8")
            

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

            # 3) user Load region-level parameters and files, choose planning year
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if 'results_df' not in st.session_state:
                st.session_state['results_df'] = pd.DataFrame()
            col1, spacing_col, col2 = st.columns([5, 1, 5])

            with col1:
                uploaded_files = st.file_uploader("Spend Scenarios Files", type="csv", accept_multiple_files=True)
                st.download_button(
                                    label = "Download File Template",
                                    data = spend_template,
                                    file_name = f"spend_plan_template_{region}.csv"
                                )
                
            with col2:
                planning_year = st.number_input(
                    "Planning Year",
                    value = "min", 
                    min_value = mmm_year + 1,
                )

                # Preparing list of periods for future use
                # planning_period (list): List of fiscal month numbers for the planning period, e.g. [1, 2, ..., 12]
                # target_period (list): list of strings representing target weeks, e.g. ['2025 Jul', '2025 'Aug', ..., '2026 Dec'].
                # target_weeks (list): list of index of target weeks in df_time_filtered (defined previously), e.g. [90, 91, 92, ..., 104]
                planning_period = list(range(1, 13))  # Fiscal months from 1 to 12
                target_period = [f"{planning_year} {months_abbv[m - 1]}" for m in planning_period]
                target_weeks = list(range(52, 105))



            if uploaded_files:
                dfs = []
                file_names = []
                for file in uploaded_files:
                    df = pd.read_csv(file)
                    df = df.T.iloc[1:, :]
                    df.columns = media_mapping.keys()
                    df.reset_index(inplace=True)
                    df.rename(columns={'index': 'FIS_MO_NB'}, inplace=True)
                    df['FIS_MO_NB'] = np.arange(1, 13)



                    dfs.append(df)
                    file_names.append(file.name) 



                # 4) User choosing file for scenario 1 & scenario 2 
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                st.write("")
                col1, col2 = st.columns(2)

                with col1:
                    scnr1_name = st.selectbox("Scenario 1", file_names)
                    scnr1_index = file_names.index(scnr1_name)
                    scnr1 = dfs[scnr1_index] 
                    st.session_state['scnr1_file'] = file_names[scnr1_index]
                    st.session_state['scnr1_table'] = scnr1  

                    scnr1_summary = []
                    for x in scnr1.columns[1:]:
                        scnr1_summary.append([x, int(np.round(scnr1[x].sum(), 0))]) 
                    scnr1_summary = pd.DataFrame(scnr1_summary, columns=['Media', 'Spending']) 
                    scnr1_summary = pd.DataFrame(scnr1_summary.values.T, columns = scnr1.columns[1:]).fillna(0)
                    scnr1_summary = scnr1_summary.iloc[1:, :]
                    scnr1_summary.rename(columns = media_mapping, inplace=True)
                    scnr1_summary = scnr1_summary.T.reset_index()
                    scnr1_summary.columns = ['Media', 'Annual Spending - Scenario 1']

                    scnr1_revised = scnr1.copy()
                    scnr1_revised.columns = ['FIS_MO_NB'] + list(media_mapping.keys())  
                    scnr1_revised['FIS_MO_NB'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

                    #Forming month-to-year multiplier grid
                    df_adjust_grid1 = multiplier_grid(df_plan= scnr1_revised, df_base= df_base, threshold= adjust_ratio)

                with col2:
                    scnr2_name = st.selectbox("Scenario 2", file_names)
                    scnr2_index = file_names.index(scnr2_name)
                    scnr2 = dfs[scnr2_index] 
                    st.session_state['scnr2_file'] = file_names[scnr2_index]
                    st.session_state['scnr2_table'] = scnr2  

                    scnr2_summary = []
                    for x in scnr2.columns[1:]:
                        scnr2_summary.append([x, int(np.round(scnr2[x].sum(), 0))])
                    scnr2_summary = pd.DataFrame(scnr2_summary, columns=['Media', 'Spending'])
                    scnr2_summary = pd.DataFrame(scnr2_summary.values.T, columns = scnr2.columns[1:]).fillna(0)
                    scnr2_summary = scnr2_summary.iloc[1:, :]
                    scnr2_summary.rename(columns = media_mapping, inplace=True)
                    scnr2_summary = scnr2_summary.T.reset_index()
                    scnr2_summary.columns = ['Media', 'Annual Spending - Scenario 2']

                    scnr2_revised = scnr2.copy()
                    scnr2_revised.columns = ['FIS_MO_NB'] + list(media_mapping.keys())
                    scnr2_revised['FIS_MO_NB'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                    #Forming month-to-year multiplier grid
                    df_adjust_grid2 = multiplier_grid(df_plan= scnr2_revised, df_base= df_base, threshold= adjust_ratio)


                    # Aggregate Summary
                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    scenarios_summary = scnr1_summary.merge(scnr2_summary, on = 'Media', how = 'left') 
                    # scenarios_summary = scenarios_summary.T
                    # names = scenarios_summary.iloc[0, :].values
                    # scenarios_summary.columns = names
                    # scenarios_summary = scenarios_summary.iloc[1:, :]

                container = st.container()
                with container:
                    numRows = scenarios_summary.shape[0]
                    st.dataframe(scenarios_summary, height = (numRows + 1) * 35 + 3, hide_index= True)

                # # 5) Error handling
                # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # inv_map = {v: k for k, v in media_mapping.items()}

                # error_medias_scnr1 = [] 
                # medias_UB_scnr1 = []
                # for x in scenarios_summary.columns:
                #     x_code = inv_map[x]
                #     spending = scenarios_summary[x].values[0] 
                #     spending_UB = df_params.loc[df_params['PCT_Change'] == 300, 'M_P_' + x_code].values[0]
                #     if spending > spending_UB:
                #         error_medias_scnr1.append(x) 
                #         medias_UB_scnr1.append(spending_UB)
                # if len(error_medias_scnr1) > 0:
                #     st.error("The following medias in Scenario 1 exeeded upper bound for annnual spending, please adjust the spending plans before running the analysis -- ")
                #     for i in np.arange(len(error_medias_scnr1)):
                #         x = error_medias_scnr1[i]
                #         x_ub = np.round(medias_UB_scnr1[i], 0)
                #         x_ub = format(x_ub, ",")
                #         st.error(x + " exceeded annual upper bound of $" + str(x_ub)) 

                # error_medias_scnr2 = [] 
                # medias_UB_scnr2 = []
                # for x in scenarios_summary.columns:
                #     x_code = inv_map[x]
                #     spending = scenarios_summary[x].values[1] 
                #     spending_UB = df_params.loc[df_params['PCT_Change'] == 300, 'M_P_' + x_code].values[0]
                #     if spending > spending_UB:
                #         error_medias_scnr2.append(x) 
                #         medias_UB_scnr2.append(spending_UB)
                # if len(error_medias_scnr2) > 0:
                #     st.error("The following media in Scenario 2 exeeded upper bound for annnual spending, please adjust the spending plans before running the analysis -- ")
                #     for i in np.arange(len(error_medias_scnr2)):
                #         x = error_medias_scnr2[i]
                #         x_ub = np.round(medias_UB_scnr2[i], 0)
                #         x_ub = format(x_ub, ",")
                #         st.error(x + " exceeded annual upper bound of $" + x_ub)


                st.write("")
                st.write("")
                st.write("")
                st.divider()



            
                # 6) Analysis begins
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if st.button("Run Analysis", help = 'Click on the button to run scenario analysis, this should take 10 - 15 seconds'):
                    # if len(error_medias_scnr1) > 0 or len(error_medias_scnr2) > 0:
                    #     st.error("Please address the spending plan errors above before running the analysis")
                    # else:
                    with st.spinner("I'm working on it ..."):   
                        artifacts = reporter(
                            spend_plans=[scnr1_revised, scnr2_revised],
                            planning_year=planning_year,
                            planning_period=planning_period,
                            target_period=target_period,
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

                    st.session_state['artifacts'] = artifacts
                    st.session_state["scenario_computed"] = True
                    st.success("Success! Please check the results in the following tabs 👉")



    #------------------------------------------------------------------------------------------------------------
    # Output Tab
    #-----------------------------------------------------------------------------------------------------------
    with tab2:
        scenario_status = st.session_state['scenario_computed']
        if scenario_status == False:
            st.write("Please upload scenario files and run the analysis in the first tab")
        else:
            currency_symbo = st.session_state['scenario_currency']
            target_var = st.session_state['scenario_target']

            artifacts = st.session_state['artifacts']
            agg0, media0, forecasts0 = artifacts['plan_summary'][0], artifacts['media_dash'][0], artifacts['forecasts']['plan1']
            agg1, media1, forecasts1 = artifacts['plan_summary'][1], artifacts['media_dash'][1], artifacts['forecasts']['plan2']

            st.write("")
            st.write("")
            viewing = st.selectbox("Select result format to view", ['Media Summary', 'Increment Forecast'])


            # Summary Tab
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if viewing == 'Media Summary':
                # Verbal Summary
                col1, spacing_col, col2 = st.columns([5, 1.5, 5]) 
                with col1:
                    summary = artifacts['summary']
                    changes_spend = metric_compare(summary, 'plan period spend')
                    changes_plan_attendance = metric_compare(summary, f'plan-driven total {target_var} (target period + future months)')
                    changes_attendance = metric_compare(summary, f'total {target_var} in target period')
                    changes_cpa = metric_compare(summary, 'cpa')
                    changes_mroas = metric_compare(summary, 'mroas')
                    

                    message0 = "Let's see..." 
                    message1_0 = "Comparing to Scenario 1, Scenario 2 has "
                    message1_1 = f"an {changes_spend[0]} of {changes_spend[1]} ({changes_spend[2]}) in total spend, "
                    message1_2 = f"which results in {changes_plan_attendance[0]} of {changes_plan_attendance[1]} ({changes_plan_attendance[2]}) in plan-driven {target_var}. "
                    message1_3 = f"That amounts to {currency_symbo}{changes_cpa[1]} ({changes_cpa[2]}) {changes_cpa[0]} in cost per {target_var} or "
                    message1_4 = f"{currency_symbo}{changes_mroas[1]} ({changes_mroas[2]}) {changes_mroas[0]} in MROAS. "
                    message1 = message1_0 + message1_1 + message1_2 + message1_3 + message1_4 

                    if changes_cpa[-1] < 0:
                        winner = "Scenario 2"
                    if changes_cpa[-1] == 0:
                        winner = "No winner"
                    if changes_cpa[-1] > 0:
                        winner = "Scenario 1"
                    if winner != "No winner":
                        message2 = f"Given the lower media cost per {target_var}, I would recommend {winner} as the better spend plan."
                    else:
                        message2 = f"Since both scenarios have the same overall media cost per {target_var}, you can pick either one of them that favors business need from other perspective."

                    message3 = f"Please check down below for more detailed summaries, or use the drop down menu to see the forecast of media driven {target_var}" 

                    st.markdown(f'<p style="font-size: 8px;">{message0}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p style="font-size: 8px;">{message1}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p style="font-size: 8px;">{message2}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p style="font-size: 8px;">{message3}</p>', unsafe_allow_html=True)

                with col2:
                    st.image("static_files/images/soul-joe-mr-mittens3.png", width= 350)

                # Media Summary
                col1, col2 = st.columns(2)
                with col1:
                    st.write("")
                    if winner == "Scenario 1":
                        st.markdown("## Scenario 1 🎷🎶")
                    else:
                        st.markdown("## Scenario 1")
                    container = st.container()
                    table = artifacts['media_dash'][0]
                    table['Media'] = table['Media'].replace(media_mapping).fillna(table['Media'])
                    table.drop(columns=[f'Target period {target_var}'], inplace=True, errors='ignore')
                    agg_info = [
                        'Aggregate',
                        agg0['plan period spend'],
                        agg0[f'plan-driven total {target_var} (target period + future months)'],
                        agg0['cpa'], agg0['mcpa'], agg0['roas'], agg0['mroas']
                    ]
                    # table.loc[-1] = agg_info
                    # table = pd.concat([table.iloc[[-1]], table.iloc[:-1]], axis = 0, ignore_index=True)
                    with container:
                        numRows = table.shape[0]
                        numCols = table.shape[1]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, width= (numCols + 1) * 85 + 3,  hide_index=True)
                with col2:
                    st.write("")
                    if winner == "Scenario 2":
                        st.markdown("## Scenario 2 🎷🎶")
                    else:
                        st.markdown("## Scenario 2")
                    container = st.container()
                    table = artifacts['media_dash'][1]
                    table['Media'] = table['Media'].replace(media_mapping).fillna(table['Media'])
                    table.drop(columns=['Target period {target_var}'], inplace=True, errors='ignore')
                    agg_info = [
                        'Aggregate',
                        agg1['plan period spend'],
                        agg1[f'plan-driven total {target_var} (target period + future months)'],
                        agg1['cpa'], agg1['mcpa'], agg1['roas'], agg1['mroas']
                    ]
                    # table.loc[-1] = agg_info
                    # table = pd.concat([table.iloc[[-1]], table.iloc[:-1]], axis = 0, ignore_index=True)
                    with container:
                        numRows = table.shape[0]
                        numCols = table.shape[1]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, width= (numCols + 1) *85 + 3,  hide_index=True)

                # Plots
                st.plotly_chart(artifacts['plots'][0])
                st.plotly_chart(artifacts['plots'][1])


            # Increment Forecast Tab
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if viewing == 'Increment Forecast':
                col1, col2 = st.columns(2)
                with col1:
                    scenario = st.radio("", ['Scenario 1', 'Scenario 2'])
                with col2:
                    timeframe  = st.radio("", ['Monthly', 'Quarterly'])

                if (scenario == 'Scenario 1') & (timeframe == 'Monthly'):
                    table = forecasts0[0]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    table['Spend'] = [str(x).split('.')[0] for x in table['Spend'].values]
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)

                if (scenario == 'Scenario 1') & (timeframe == 'Quarterly'):
                    table = forecasts0[1]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    table['Spend'] = [str(x).split('.')[0] for x in table['Spend'].values]
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)

                if (scenario == 'Scenario 2') & (timeframe == 'Monthly'):
                    table = artifacts['forecasts']['plan2'][0]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    table['Spend'] = [str(x).split('.')[0] for x in table['Spend'].values]
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)

                if (scenario == 'Scenario 2') & (timeframe == 'Quarterly'):
                    table = artifacts['forecasts']['plan2'][1]
                    table.iloc[:, 2:] = table.iloc[:, 2:].astype(float).astype(int)
                    table['Spend'] = [str(x).split('.')[0] for x in table['Spend'].values]
                    container = st.container()
                    with container:
                        numRows = table.shape[0]
                        st.dataframe(table, height = (numRows + 1) * 35 + 3, hide_index=True)

                