# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd
import base64
import io
import random

import networkx as nx
from algorithms import dijkstra, temp_func

app = dash.Dash(__name__)


# GLOBAL VARIABLES
df1 = pd.DataFrame({'source': [random.randint(1, 10) for _ in range(10)],
                    'target': [random.randint(1, 10) for _ in range(10)],
                    'weight': [random.lognormvariate(mu=0, sigma=0.5) for _ in range(10)]})
G1 = nx.DiGraph(df1)
df2 = pd.DataFrame({'source': [random.randint(1, 10) for _ in range(10)],
                    'target': [random.randint(1, 10) for _ in range(10)],
                    'weight': [random.lognormvariate(mu=0, sigma=0.5) for _ in range(10)]})
G2 = nx.DiGraph(df2)
datasets = [df1, df2]

algorithm_types = ['Shortest path', 'Minimal spanning tree', 'Matching']
algorithms_per_type = {'Shortest path': ['Dijkstra', 'Bellman-Ford', 'Floyd-Warshall'],
                       'Minimal spanning tree': ['Kruskal', 'Prim'],
                       'Matching': ['Ford-Fulkerson']}


# APP LAYOUT
app.layout = html.Div(id='main-body', children=[
    # INPUT PANEL
    html.Div(className='input-panel', children=[
        # Data loading
        html.H1('Input panel'),
        html.P('Select file type'),
        dcc.Dropdown(id="dropdown-filetype",
            options=[
                {'label': 'CSV', 'value': 'csv'},
                {'label': 'XLS', 'value': 'xls'}
            ],
            value='csv'
        ),
        dcc.Upload(
            id='upload-field',
            className='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            # Do not allow multiple files to be uploaded
            multiple=False
        ),
        html.Div(id='head-data-upload'),

        # Algorithm selection
        html.Hr(),
        html.P('Select algorithm type'),
        dcc.Dropdown(
            id="algorithm-type-dropdown",
            options=[{'label': x, 'value': x} for x in algorithm_types],
            value=algorithm_types[0]
        ),
        html.P('Select algorithm to run'),
        dcc.Dropdown(id='algorithm-dropdown'),
        # Algorithm settings: everything present; will be filled in when necessary with callbacks
        html.P('Settings'),
        html.Div(id='algorithm-settings', children=[
            html.Div(id='dijkstra-settings'),
            html.Div(id='temp_func-settings')
        ]),
        html.Button('Run algorithm', id='run-algorithm-button')
    ]),

    # VISUAL ANALYTICS PANEL
    html.Div(className='vis-panel', children=[
        'Visual analytics panel',
        html.Div(id='vis-test')
    ]),

    # OUTPUT PANEL
    html.Div(className='output-panel', children=[
        'Right panel'
    ])
])


# FUNCTIONS
def parse_contents(contents, filetype, filename):
    # If we want to give the option to upload multiple files
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing ' + filename + '. The given '
            'error is ' + str(e) + '.'
        ])

    return


# CALLBACK FUNCTIONS
@app.callback(Output('head-data-upload', 'children'),
              [Input('upload-field', 'contents'),
               Input('dropdown-filetype', 'value')],
               [State('upload-field', 'filename')]
              )
def load_data(content, filetype, name):
    if content is not None:
        children = [
            parse_contents(content, filetype, name)]
        return children

    return html.Div([
        'There was no file uploaded'
    ])

# Algorithm selection callbacks
@app.callback(Output('algorithm-dropdown', 'options'),
              [Input('algorithm-type-dropdown', 'value')])
def set_algorithm_options(selected_type):
    return [{'label': x, 'value': x} for x in algorithms_per_type[selected_type]]

@app.callback(Output('algorithm-dropdown', 'value'),
              [Input('algorithm-dropdown', 'options')])
def set_algorithm_value(available_options):
    return available_options[0]['value']


if __name__ == '__main__':
    app.run_server(debug=True)
