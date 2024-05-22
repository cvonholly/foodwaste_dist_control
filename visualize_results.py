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


app.run_server(debug=True)