from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from main import name, p_name

print(f"loading results/{p_name}_{name}_out.csv")

df = pd.read_csv(f'results/{p_name}_{name}_out.csv', index_col=0, header=[0,1])
df_raw = df

yaxis_name = 'MCal food'


app = Dash(__name__)

app.layout = html.Div([
    html.H4('results of simulation'),
    dcc.Dropdown(
        id="dropdown",
        options=[c for c in df.columns.get_level_values(0).unique()] + ['all'],
        value='all'
    ),
    dcc.Graph(id="graph"),
    dcc.Graph(id="graph_2"),
    dcc.Graph(id='graph_sc_vs_sc'),
])


@app.callback(
    Output("graph", "figure"), 
    Input("dropdown", "value"))
def update_figure(input_value):
    print(df)
    if input_value=='all':
        dff = df.copy()
        dff.columns = [' '.join(col).strip() for col in df.columns.values]  # flatten multiindex
    else:
        dff = df[input_value].copy()
    fig = px.line(dff,
                  labels={
                     "index": "time step (days)",
                     "value": "MCal food",
                 },)
    fig.update_layout(
        title=dict(text=f"time plot of variables", 
                   font=dict(size=22), automargin=True, yref='paper')
    )
    return fig

@app.callback(
    Output("graph_2", "figure"), 
    Input("dropdown", "value"))
def update_figure(input_value):
    dff = df.copy()
    if input_value=='all':
        dff = dff.copy()
    else:
        dff = dff[[input_value]]
    df_new = pd.DataFrame(0, index=dff.columns, columns=['sum', 'type'])
    df_new['sum'] = dff.sum(axis=0)
    for c in dff.columns:
        if c[1]=='input flow':
            df_new.loc[c, 'type'] = 'inputs'
        if 'foodwaste'==c[1] or 'food waste'==c[1] or 'self consumption' in c[1]:
            df_new.loc[c, 'type'] = 'outputs'
        if c[1].startswith('flow'):
            df_new.loc[c, 'type'] = 'flows'
    df_new['node'] = [c[0] for c in df_new.index]
    df_new.index = [' '.join(i).strip() for i in df_new.index]  # flatten multiindex
    df_new['names'] = df_new.index
    df_new.rename(columns={'sum': yaxis_name}, inplace=True)
    try:
        fig = px.bar(df_new, x='type', y=yaxis_name, barmode='stack',
                    labels='names', text='names', color='node',
                    category_orders={'type': ['inputs', 'flows', 'outputs']})
    except:
        fig = px.bar(df_new, x='type', y=yaxis_name, barmode='stack',
                 labels='names', text='names', color='node',
                 category_orders={'type': ['inputs', 'flows', 'outputs']})
    fig.update_xaxes(type='category')
    fig.update_layout(
        title=dict(text=f"distribution of input, flows and outputs", 
                   font=dict(size=22), automargin=True, yref='paper')
    )
    return fig

@app.callback(
    Output("graph_sc_vs_sc", "figure"), 
    Input("dropdown", "value"))
def update_figure(input_value):
    dff = pd.DataFrame()
    if input_value=='all':
        dff = df_raw.copy()
    else:
        dff = df_raw[[input_value]]
    df_new = pd.DataFrame(0, index=dff.columns, columns=['sum', 'type'])
    df_new['sum'] = dff.sum(axis=0)
    for c in dff.columns:
        if c[1]=='input flow':
            df_new.drop(c, axis=0, inplace=True)
        elif 'foodwaste'==c[1] or 'food waste'==c[1]:
            df_new.loc[c, 'type'] = 'foodwaste'
        elif 'self consumption' in c[1]:
            df_new.loc[c, 'type'] = 'self consumption'
        elif c[1].startswith('flow'):
            df_new.drop(c, axis=0, inplace=True)
    df_new['node'] = [c[0] for c in df_new.index]
    df_new.index = [' '.join(i).strip() for i in df_new.index]  # flatten multiindex
    df_new['names'] = df_new.index
    # calculate total foodwaste
    df_fw = df_new[df_new['type']=='foodwaste']
    fw_sum = np.round(df_fw['sum'].sum(),1)
    print("total foodwaste is ", df_fw['sum'].sum())
    # calculate total self consumption
    df_sc = df_new[df_new['type']=='self consumption']
    df_new.fillna(0, inplace=True)
    sc_sum = np.round(df_sc['sum'].sum(),1)
    print("total self consumption is ", df_sc['sum'].sum())
    df_new.rename(columns={'sum': yaxis_name}, inplace=True)
    df_new.fillna(0, inplace=True)
    fig = px.bar(df_new, x='type', y=yaxis_name, barmode='stack',
                 labels='names', text='names', color='node',
                 category_orders={'type': ['foodwaste', 'self consumption']})
    fig.update_xaxes(type='category')
    fig.update_layout(
        title=dict(text=f"summed foodwaste: {fw_sum} MCal                         self-consumption: {sc_sum} MCal", 
                   font=dict(size=22), automargin=True, yref='paper')
    )
    return fig


app.run_server(debug=True)
