import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import tensorflow as tf
import tensorflow_probability as tfp
import arviz as az

import IPython
import pickle as pkl
import meridian

from meridian import constants
from meridian.data import load
from meridian.data import test_utils
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import optimizer
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.analysis import summarizer
from meridian.analysis import formatter
 
def chacha_data(mmm):
    from meridian.analysis.optimizer import BudgetOptimizer
    
    optimizer = BudgetOptimizer(mmm)  # mmm = twój model/analyzer
    results = optimizer.optimize()  # budżet np. 1 mln
    box_c = results.optimized_data 
    output_box_df_c = box_c.to_dataframe().reset_index()
    change_in_channel = output_box_df_c[['channel', 'metric', 'spend']]
    chacha_data = change_in_channel[change_in_channel['metric'] == 'mean']
    cha_data = chacha_data[['channel', 'spend']]
    cha_data = cha_data.rename(columns={"spend": "Optimized Spend"})
    
    return cha_data

def chacha_data_chart(cha_data):
    # Dodaj kolory: niebieski dla >0, czerwony dla <0
    cha_data["color"] = cha_data["Optimized Spend"].apply(lambda x: "#4ECDE6" if x > 0 else "#F28B82")
    # Formatowanie wartości jako tekst na słupkach
    cha_data["label"] = cha_data["Optimized Spend"].apply(lambda x: f'{x/1e6:.1f}M')
    fig_cha = go.Figure()
    fig_cha.add_trace(go.Bar(
        x=cha_data["channel"],
        y=cha_data["Optimized Spend"],
        marker_color=cha_data["color"],
        text=cha_data["label"],
        textposition="outside",
        textfont=dict(color="#3C4043"),
        hovertemplate='<b>%{x}</b><br>Spend: %{y:$,.0f}<extra></extra>',
    ))
    
    # Stylizacja
    fig_cha.update_layout(
        title={
            "text": "Change in optimized spend for each channel",
            "x": 0.01,
            "xanchor": "left",
            "font": dict(size=18, color="#3C4043", family="Google Sans Display")
        },
        plot_bgcolor='white',
        xaxis=dict(
            title='',
            tickangle=-45,
            tickfont=dict(size=12, family="Roboto", color="#5F6368"),
            showline=True,
            linecolor="#DADCE0"
        ),
        yaxis=dict(
            title="$",
            titlefont=dict(size=12, family="Roboto", color="#5F6368"),
            tickfont=dict(size=12, family="Roboto", color="#5F6368"),
            zeroline=True,
            zerolinecolor="#DADCE0",
            gridcolor="#EFEFEF",
        ),
        height=450,
        width=434,
        margin=dict(t=60, b=80, l=40, r=20),
    )
    
    st.plotly_chart(fig_cha, use_container_width=True)
  
def opti_budget_tab(mmm, cha_data):
    from meridian.analysis.optimizer import BudgetOptimizer
    optimizer = BudgetOptimizer(mmm)
    results = optimizer.optimize()  # This runs the optimization
    box_b = results.nonoptimized_data  # This gets the response 
    output_box_df_b = box_b.to_dataframe().reset_index()
    nonopti_budget = output_box_df_b[['channel', 'metric', 'spend']]
    nopti_budget = nonopti_budget[nonopti_budget['metric'] == 'mean']
    nop_budget = nopti_budget[['channel', 'spend']]
    nop_budget = nop_budget.rename(columns={"spend": "Non-optimized Spend"})
    budget_allocation_tab = pd.merge(nop_budget, cha_data[['channel', 'Optimized Spend']], on='channel', how='left')
    # Formatowanie danych w tabeli
    budget_allocation_tab['Non-optimized Spend'] = budget_allocation_tab['Non-optimized Spend'].apply(lambda x: '{:,.0f}'.format(x).replace(',', ' '))
    budget_allocation_tab['Optimized Spend'] = budget_allocation_tab['Optimized Spend'].apply(lambda x: '{:,.0f}'.format(x).replace(',', ' '))
    budget_allocation_tab = budget_allocation_tab.reset_index()
    # Tworzenie HTML z tabelą
    st.table(budget_allocation_tab)





    



