from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

df = pd.read_csv('results/out.csv', index_col=0, header=[0,1])

# new

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
    fig = px.line(dff)
    return fig

@app.callback(
    Output("graph_2", "figure"), 
    Input("dropdown", "value"))
def update_figure(i):
    dff = df.copy()
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
    print(df_new)
    fig = px.bar(df_new, x='type', y='sum', barmode='stack',
                 labels='names', text='names', color='node',
                 category_orders={'type': ['inputs', 'flows', 'outputs']})
    fig.update_xaxes(type='category')
    return fig

@app.callback(
    Output("graph_sc_vs_sc", "figure"), 
    Input("dropdown", "value"))
def update_figure(i):
    dff = df.copy()
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
    print(df_new)
    fig = px.bar(df_new, x='type', y='sum', barmode='stack',
                 labels='names', text='names', color='node',
                 category_orders={'type': ['foodwaste', 'self consumption']})
    fig.update_xaxes(type='category')
    return fig


app.run_server(debug=True)
