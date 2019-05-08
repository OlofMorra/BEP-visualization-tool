# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State
import dash_table
from dash.exceptions import PreventUpdate
import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd
import base64
import io
import random
import json
import itertools as it

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
datasets = {'df1': df1, 'df2': df2}

# structure: {algorithm type: {algorithm: [html structure]}}
algorithms = {
    'Shortest path': ['Dijkstra', 'Bellman-Ford', 'Floyd-Warshall'],
    'Minimal spanning tree': ['Kruskal', 'Prim'],
    'Matching': ['Ford-Fulkerson']
}

network_layouts = ['breadthfirst', 'circle', 'concentric', 'cose', 'grid', 'random']

app.config['suppress_callback_exceptions'] = True


# APP LAYOUT
app.layout = html.Div(id='main-body', children=[
    # INPUT PANEL
    html.Div(className='input-panel', children=[
        # Data loading
        html.H1('Input panel'),
        html.Div('Supported file types are csv, json, '
                 'xls, dta, xpt and pkl.'),

        dcc.Upload(
            id='upload-field',
            className='upload-data',
            children=[html.Div(['Drag and Drop or ',
                                html.A('Select Files')
                                ]),
                      html.Div(id='upload-message')],
            # Do allow multiple files to be uploaded
            multiple=True
        ),
        html.Button('Click to store in memory', id='memory-button'),
        html.Div(id='test'),
        dcc.Store(id='dataset1', storage_type='memory'),
        dcc.Store(id='dataset2', storage_type='memory'),
        dcc.Store(id='dataset3', storage_type='memory'),
        dcc.Store(id='dataset4', storage_type='memory'),
        dcc.Store(id='dataset5', storage_type='memory'),
        # Storage components for datasets
        html.Div(id='datasets', children=[
            html.Div(id='nr-of-datasets', children='0', style={'display': 'none'})
        ]),
        html.Div(id='head-data-upload'),

        # Algorithm selection
        html.Hr(),
        html.H1('Algorithm settings'),
        html.P('Select dataset'),
        dcc.Dropdown(id='dataset-dropdown',
                     options=[{'label': 'df1', 'value': 'df1'}, {'label': 'df2', 'value': 'df2'}],
                     value='df1'),
        html.P('Select algorithm type'),
        dcc.Dropdown(
            id="algorithm-type-dropdown",
            options=[{'label': x, 'value': x} for x in algorithms.keys()],
            value=[x for x in algorithms.keys()][0]  # dumb workaround; dict.keys() doesn't support indexing
        ),
        html.P('Select algorithm to run'),
        dcc.Dropdown(id='algorithm-dropdown'),
        # Algorithm settings: everything present from the beginning and will be filled in when necessary with callbacks
        html.H3('Settings'),
        html.Div(id='algorithm-settings', children=[
            # Dijkstra
            html.Div(id='dijkstra-settings', style={'display': 'none'}, children=[
                html.P('Select start node'),
                dcc.Dropdown(id='dijkstra-start-dropdown'),
                html.P('Select weight column'),
                dcc.Dropdown(id='dijkstra-weight-dropdown'),
                html.Button(id='dijkstra-run-button', n_clicks=0, children='Run algorithm', type='submit')
            ]),
            # Bellman-Ford
            html.Div(id='bellman-ford-settings', style={'display': 'none'}, children=[]),
            # Floyd Warshall
            html.Div(id='floyd-warshall-settings', style={'display': 'none'}, children=[]),
            # Kruskal
            html.Div(id='kruskal-settings', style={'display': 'none'}, children=[]),
            # Prim
            html.Div(id='prim-settings', style={'display': 'none'}, children=[]),
            # Ford-Fulkerson
            html.Div(id='ford-fulkerson-settings', style={'display': 'none'}, children=[])
        ]),
    ]),

    # VISUAL ANALYTICS PANEL
    html.Div(className='vis-panel', children=[
        html.H1('Visual analytics panel'),
        html.Div(id='vis-graphs')
    ]),

    # OUTPUT PANEL
    html.Div(className='output-panel', children=[
        html.H1('Right panel'),
        html.P('Layout:', style={'width': '25%', 'display': 'inline-block'}),
        dcc.Dropdown(id='network-layout-dropdown',
                     options=[{'label': x, 'value': x} for x in network_layouts],
                     value=network_layouts[0],
                     style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'middle'}),
        html.Div(id='network')
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

#########################
# INPUT PANEL CALLBACKS #
#########################
def validate_dataset(i, contents, filename):
    df = []
    # Splitting at start of file for content type and the actual data
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    # Select last 5 characters from filename as extension is in there
    file_type = filename[-5:]

    try:
        if 'csv' in file_type:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=" ")
        elif 'json' in file_type:
            # Assume that the user uploaded a JSON file
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in file_type:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'dta' in file_type:
            # Assume that the user uploaded an stata data file
            df = pd.read_stata(io.BytesIO(decoded))
        elif 'xpt' in file_type:
            # Assume that the user uploaded an SAS file
            df = pd.read_stata(io.BytesIO(decoded))
        elif 'pkl' in file_type:
            # Assume that the user uploaded an python pickle file
            df = pd.read_pickle(io.BytesIO(decoded))
        else:
            raise Exception('File format is not supported ')

    except Exception as e:
        return html.Div([
            'There was an error processing ' + filename +
            '. ' + str(e) + '.'
        ]), None

    data = df.to_dict()

    return html.Div(['Upload of ' + filename + ' (dataset ' + str(i) + ' was successful.']), data

# Callback functions; functions that execute when something is changed
@app.callback([Output('dataset1', 'data')],
              [Input('upload-field', 'contents')],
              [State('upload-field', 'filename')]
              )
def load_data(contents, names):
    childrenUplMess = list()
    datasets = list()
    count = 0

    if contents is None:
        raise PreventUpdate

    if contents is not None:
        for i, (dataset, name) in enumerate(zip(it.islice(contents, 5), it.islice(names, 5))):
            child, data = validate_dataset(i, dataset, name)
            childrenUplMess.extend([child])
            datasets.extend([data])
            count += 1

        for j in range(count, 5):
            datasets.extend([{}])

        print(type(datasets[0]))
        print(type(datasets[1]))
        print(type(datasets[2]))
        print(type(datasets[3]))
        print(type(datasets[4]))
        print(type(datasets))
        return datasets[0]
            #, datasets[1], datasets[2], datasets[3], datasets[4]

    try:
        return {}, {}, {}, {}, {}
    except Exception as e:
        print(e)

@app.callback(Output('test', 'children'),
                  [Input('memory-button', 'n_clicks')],
                  [State('dataset1', 'data')])
def on_click(n_clicks, data):
    if n_clicks is None:
        # Preventing the None callbacks is important with the store component,
        # you don't want to update the store for nothing.
        raise PreventUpdate

    return html.Div(str(data) + str(n_clicks))

# Algorithm selection callbacks
@app.callback(Output('algorithm-dropdown', 'options'),
              [Input('algorithm-type-dropdown', 'value')])
def set_algorithm_options(selected_type):
    return [{'label': x, 'value': x} for x in algorithms[selected_type]]

@app.callback(Output('algorithm-dropdown', 'value'),
              [Input('algorithm-dropdown', 'options')])
def set_algorithm_value(available_options):
    return available_options[0]['value']

@app.callback([Output('dijkstra-settings', 'style'),
               Output('bellman-ford-settings', 'style'),
               Output('floyd-warshall-settings', 'style'),
               Output('kruskal-settings', 'style'),
               Output('prim-settings', 'style'),
               Output('ford-fulkerson-settings', 'style')],
              [Input('algorithm-dropdown', 'value')])
def show_settings(selected_algorithm):
    result_dict = {x: {'display': 'none'} for y in algorithms.values() for x in y}  # hide all divs
    result_dict[selected_algorithm] = {}  # remove the selected algorithm's div's style settings
    return [x for x in result_dict.values()]


# Dijkstra callbacks
# TODO account for possible changes in Graph object
@app.callback(Output('dijkstra-start-dropdown', 'options'),
              [Input('dijkstra-settings', 'style')])
def set_dijkstra_start_options(style):
    if 'display' in style.keys() and style['display'] == 'none':
        return []
    else:
        # TODO fix graph/dataset met callbacks
        return [{'label': str(x), 'value': x} for x in sorted(G1.nodes)]

@app.callback(Output('dijkstra-start-dropdown', 'value'),
              [Input('dijkstra-start-dropdown', 'options')])
def set_dijkstra_start_value(options):
    if len(options) > 0:
        return options[0]['value']
    else:
        return ''

@app.callback(Output('dijkstra-weight-dropdown', 'options'),
              [Input('dijkstra-settings', 'style')])
def set_dijkstra_weight_options(style):
    if 'display' in style.keys() and style['display'] == 'none':
        return []
    else:
        # TODO fix graph/dataset met callbacks
        return [{'label': x, 'value': x} for x in df1.columns]

@app.callback(Output('dijkstra-weight-dropdown', 'value'),
              [Input('dijkstra-weight-dropdown', 'options')])
def set_dijkstra_start_value(options):
    if len(options) > 0:
        if 'weight' in [x['value'] for x in options]:
            return 'weight'
        else:
            return options[0]['value']
    else:
        return ''


####################################
# VISUAL ANALYTICS PANEL CALLBACKS #
####################################
@app.callback(Output('vis-graphs', 'children'),
              [Input('dijkstra-run-button', 'n_clicks')],
              [State('dijkstra-start-dropdown', 'value'), State('dijkstra-weight-dropdown', 'value'),
               State('dataset-dropdown', 'value'), State('vis-graphs', 'children')])
def run_dijkstra(n_clicks, start, weight, df, current_graphs):
    graph = dcc.Graph(figure={
        'data': [{'x': datasets[df]['source'], 'y': datasets[df]['target'], 'type': 'line', 'name': 'SF'}]
    })
    if n_clicks > 0:
        if current_graphs is None:
            return list([graph])
        else:
            current_graphs.append(graph)
            return current_graphs


##########################
# OUTPUT PANEL CALLBACKS #
##########################
@app.callback(Output('network', 'children'),
              [Input('dataset-dropdown', 'value'), Input('network-layout-dropdown', 'value')])
def show_network(df_name, layout):
    df = datasets[df_name]
    set_of_nodes = set(df['source']) | set(df['target'])  # union of the sets of source and target nodes
    nodes = [{'data': {'id': x, 'label': x}} for x in set_of_nodes]
    edges = [{'data': {'source': row['source'], 'target': row['target']}} for i, row in df[['source', 'target']].iterrows()]
    elements = nodes + edges

    return html.H3(df_name), cyto.Cytoscape(
        id='cytoscape-layout-1',
        elements=elements,
        style={'width': '100%', 'height': '350px'},
        layout={
            'name': layout
        }
    )

if __name__ == '__main__':
    app.run_server(debug=True)