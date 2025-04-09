import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import IPython
import pickle as pkl
import max26_func
import max26_opti_func

# Data source
file_path = r"C:\Users\micha\Desktop\Models\3M\Meridian_files\mmm_dump.pkl"
with open(file_path, 'rb') as file:
    mmm = pkl.load(file)

# Configure page
st.set_page_config(layout="wide")
# Create a two-column layout
col1, col2 = st.columns([2, 1])  # Adjust the proportions as needed

with col2:  # Right column
    st.image(r"C:\Users\micha\Downloads\Cap_logo.png", width=350)
    
with col1:  # Left column
    # Custom style for the header
    st.markdown("<h2 style='font-size:46px; color:black; text-decoration:underline red;'>Marketing Mix Modelling Dashboard</h2>", unsafe_allow_html=True)
    

# Styl zakładki bocznej
st.html("""<style>[data-testid="stSidebarContent"] {color: black; background-color: #30B700} </style>""")
st.sidebar.subheader('Configuration parameters', divider="grey") 
# definicja kontentu strony
show_results = st.sidebar.checkbox('Model Results Summary', value=True)
if show_results:
    from max26_func import rr_write
    rr_text = rr_write(mmm)
    st.markdown(f"<h1 style='font-size:26px; color:black; text-decoration:underline; text-decoration-color:blue;'>Model's R² {rr_text}</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Expected revenu vs. actual revenue', divider="blue") 
    from max26_func import fit_data, fit_chart
    fit_tabel = fit_data(mmm)
    fit_geo_list = set(fit_tabel['geo'])
    fit_geo_radio = list(fit_geo_list)
    fit_m = st.radio('Choose a region:', fit_geo_radio, horizontal=True, key='fit_radio')
    fit_chart(fit_tabel, fit_m)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Channel Contribution', divider="blue") 
    from max26_func import marketing_contribution, marketing_contribution_chart, media_activities
    chart_col, table_col = st.columns([1.8, 1.2])
    
    with chart_col:
        final_df = marketing_contribution(mmm)
        marketing_contribution_chart(final_df)
    
    with table_col:
         st.markdown("<br>", unsafe_allow_html=True)
         st.write('Impressions and spend per media channel')
         media_data = media_activities(mmm)
         st.markdown("<br>", unsafe_allow_html=True)
         st.dataframe(media_data, use_container_width=True)
    
    from max26_func import prior_posterior_data, prior_posterior_chart
    combined_df = prior_posterior_data(mmm)
    channel_list = [column for column in combined_df.columns if column != 'distribution']
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Channel Prior and Posterior Distributions', divider="blue") 
    prior = st.radio('Choose a channel:', channel_list, horizontal=True, key='prior_radio')
    prior_posterior_chart(combined_df, prior)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Hill saturation curves', divider="blue") 
    from max26_func import hill_curves, hill_curves_chart
    K_hill_chart_data = hill_curves(mmm)
    hill_channel_list = set(K_hill_chart_data['channel'])
    hill = st.radio('Choose a channel:', list(hill_channel_list), horizontal=True, key='hill_radio')
    hill_curves_chart(K_hill_chart_data, hill)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Adstock saturation curves', divider="blue") 
    from max26_func import adstock_curves, adstock_curves_chart
    K_adstock_chart_data = adstock_curves(mmm)
    adstock_channel_list = set(K_adstock_chart_data['channel'])
    adstock = st.radio('Choose a channel:', list(adstock_channel_list), horizontal=True, key='adstock_radio')
    adstock_curves_chart(K_adstock_chart_data, adstock)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('ROI vs Effectiveness', divider="blue") 
    from max26_func import cont_total, cont_chart
    co_data = cont_total(mmm)
    cont_chart(co_data)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Response curves by marketing channel', divider="blue") 
    from max26_func import response_hill, response_hill_chart
    res_hi = response_hill(mmm)
    response_hill_chart(res_hi)

show_opti = st.sidebar.checkbox('Recommended budget allocation', value=False , key='key_opti')
if show_opti:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Standard optimization scenario', divider="blue") 
    from max26_opti_func import chacha_data, chacha_data_chart, opti_budget_tab  
    cha_data = chacha_data(mmm)
    chart_col, tab_col = st.columns([2, 3])  # Dwie kolumny: lewa większa
    with chart_col:
        st.markdown("<br>", unsafe_allow_html=True)
        chacha_data_chart(cha_data)

    with tab_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write('**Optimized budget allocation**')
        opti_budget_tab(mmm, cha_data)
            


show_custom_opti = st.sidebar.checkbox('Customized budget allocation', value=False , key='key_custom_opti')
if show_custom_opti:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('Customized optimization scenario', divider="blue") 
    #from max26_opti_func import chacha_data, chacha_data_chart, opti_budget_tab

    #with st.sidebar.form("my_form"):
    #    st.write("Enter new budget value")
    #    budget_num = st.number_input(' ', value=1000000, placeholder="Type a number...", format="%d")       
    #    submitted = st.form_submit_button("Submit")

    #if submitted:
    #    cha_data = chacha_data(mmm, budget_num)
    #    chart_col, tab_col = st.columns([2, 3])  # Dwie kolumny: lewa większa
    #    with chart_col:
    #        st.markdown("<br>", unsafe_allow_html=True)
    #        chacha_data_chart(cha_data)

    #    with tab_col:
    #        st.markdown("<br>", unsafe_allow_html=True)
    #        opti_budget_tab(mmm, cha_data)