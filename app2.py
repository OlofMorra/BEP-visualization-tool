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
import sys
import time
from heapq import heapify, heappush, heappop
import datetime

import networkx as nx
from algorithms2 import Dijkstra

####################
#   APP SETTINGS   #
####################
app = dash.Dash(__name__)

app.config['suppress_callback_exceptions'] = True
app.scripts.config.serve_locally = True

####################
# GLOBAL VARIABLES #
####################
# Use only for     #
# standard options #
# of cell          #
####################
algorithms = {
    'Shortest path': ['Dijkstra', 'Bellman-Ford', 'Floyd-Warshall'],
    'Minimal spanning tree': ['Kruskal', 'Prim'],
    'Matching': ['Ford-Fulkerson']
}

network_layouts = ['breadthfirst', 'circle', 'concentric', 'cose', 'grid', 'random']

###################
# CREATING PANELS #
###################


def create_input_panel():
    return html.Div(className='input-panel', children=[
        # Data uploading and saving
        html.H1('Input panel'),
        html.Div('Supported file types are csv, json, '
                 'xls, dta, xpt and pkl.'),

        dcc.Loading(id="loading-data",
                    children=[dcc.Upload(
                        id='upload-field',
                        className='upload-data',
                        children=[html.Div(['Drag and Drop or ',
                                    html.A('Select Files')])],
                        # Do allow multiple files to be uploaded
                        multiple=True
                    ), html.Div(id='upload-message')],
                    type="default"),
        dcc.Store(id='datasets', storage_type='memory'),

        # Algorithm selection
        html.Hr(),
        html.H1('Algorithm settings'),
        html.P('Select dataset'),
        dcc.Dropdown(id='dataset-dropdown',
                     options=[],
                     value=''),
        html.P('Select algorithm type'),
        dcc.Dropdown(
            id="algorithm-type-dropdown",
            options=[{'label': x, 'value': x} for x in algorithms.keys()],
            value=[x for x in algorithms.keys()][0]  # dumb workaround; dict.keys() doesn't support indexing
        ),
        html.P('Select algorithm to run'),
        dcc.Dropdown(id='algorithm-dropdown'),

        # Algorithm settings: everything present from the beginning and will be filled in when necessary with callbacks
        html.Div(id='algorithm-settings', children=[
            dcc.ConfirmDialog(
                id='settings-missing-dialog',
                message='Please select a value for all settings'),
            # Dijkstra
            html.Div(id='dijkstra-settings', style={'display': 'none'}, children=[
                html.H3('Settings'),
                html.P('Select start node'),
                dcc.Dropdown(id='dijkstra-start-dropdown'),
                html.P('Edge weights:'),
                dcc.RadioItems(
                    id='dijkstra-weight-radio',
                    options=[{'label': 'Edge weights all equal to 1', 'value': 'no'},
                             {'label': 'Use column in dataset as edge weights', 'value': 'yes'}],
                    value='no',
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.Dropdown(id='dijkstra-weight-dropdown', style={'display': 'none'}),
                html.Button(id='dijkstra-run-button', n_clicks=0, children='Run algorithm', type='submit')
            ]),
            # Bellman-Ford
            html.Div(id='bellman-ford-settings', style={'display': 'none'}, children=[
                html.H3('Settings'),
            ]),
            # Floyd Warshall
            html.Div(id='floyd-warshall-settings', style={'display': 'none'}, children=[
                html.H3('Settings'), ]),
            # Kruskal
            html.Div(id='kruskal-settings', style={'display': 'none'}, children=[
                html.H3('Settings'), ]),
            # Prim
            html.Div(id='prim-settings', style={'display': 'none'}, children=[
                html.H3('Settings'), ]),
            # Ford-Fulkerson
            html.Div(id='ford-fulkerson-settings', style={'display': 'none'}, children=[
                html.H3('Settings'), ])
        ]),
    ])


def create_visualisation_panel():
    return html.Div(className='visualisation-panel', children=[
                dcc.Store(id='dijkstra-info', storage_type='memory'),
                dcc.Store(id='graph-info', storage_type='memory'),
                dcc.Store(id='data-info', storage_type='memory'),
                html.Div(className='vis-panel', children=[
                    html.H1('Visual analytics panel'),
                    dcc.Dropdown(id='show-graphs-dropdown', multi=True,
                                 style={'vertical-align': 'middle'}),
                    html.Div(id='shown-vis-graphs')
                ])])


def create_output_panel():
    return html.Div(className='output-panel', children=[
                html.H1('Right panel'),
                html.Div(id='show-div-content'),
                html.P('Draw network graph:'),
                dcc.RadioItems(id='draw-network-radio',
                               options=[{'label': 'yes', 'value': 'yes'}, {'label': 'no', 'value': 'no'}],
                               value='no'),
                html.Div(id='network', style={'display': 'none'}, children=[
                    html.P('Layout:', style={'width': '25%', 'display': 'inline-block'}),
                    dcc.Dropdown(id='network-layout-dropdown',
                                 options=[{'label': x, 'value': x} for x in network_layouts],
                                 value=network_layouts[0],
                                 style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                    html.Div(id='network-graph')])
            ])


#############
# FUNCTIONS #
#############
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

    return html.Div(['Upload of ' + filename + ' was successful.']), df.to_dict('records')


def getDataFrame(datasets, i):
    df = pd.DataFrame.from_records(datasets[i])
    return df


def createDiGraph(df, weight):
    G1 = nx.DiGraph()
    if weight in df.columns:
        G1.add_weighted_edges_from(zip(df['source'], df['target'], df[weight]))
    else:
        G1.add_edges_from(zip(df['source'], df['target']))
    return G1


def append_new_graph(current_graphs, name, data, xlab, ylab):
    if current_graphs is None:
        name = name + str(1)  # id number corresponding to index in the list of graphs
        graph = dcc.Graph(
            id=name,
            figure={
                'data': data,
                'layout': go.Layout(
                    title={'text': name},
                    xaxis={'title': xlab,
                           'rangeslider' : {'visible': True},
                            'type' : 'date',
                           'tickformat':'%M~%S~%L'},
                    yaxis={'title': ylab}
                )}
        )
        return list([graph])
    else:
        name = name + str(len(current_graphs) + 1)  # id number corresponding to index in the list of graphs
        graph = dcc.Graph(
            id=name,
            figure={
                'data': data,
                'layout': go.Layout(
                    title={'text': name}, xaxis={'title': xlab,
                           'rangeslider' : {'visible': True},
                            'type' : 'date',
                           'tickformat':'%M~%S~%L'}, yaxis={'title': ylab}
                )}
        )
        current_graphs.append(graph)
        return current_graphs


def clean_dict(mydict):
    """
    Robust function that changes all indices to strings in a dict

    :param mydict:      dictionary which might contain non-string keys
    :return: mydict:    dictionary with only string keys
    """
    keys_to_change = []
    for key, value in mydict.items():
        if type(key) is not str:
            keys_to_change.append(key)
        if value == float('inf'):
            mydict[key] = 'inf'

    for key in keys_to_change:
        mydict[str(key)] = mydict[key]
        del mydict[key]

    return mydict


def get_memory_used(*args):
    result = 0
    for x in args:
        result += sys.getsizeof(x)
    return result

def column(matrix, i):
    return [row[i] for row in matrix]

#########################
# INPUT PANEL CALLBACKS #
#########################
# Callback functions; functions that execute when something is changed
@app.callback([Output('upload-message', 'children'),
               Output('datasets', 'data'),
               Output('dataset-dropdown', 'options'),
               Output('dataset-dropdown', 'value')],
              [Input('upload-field', 'contents')],
              [State('upload-field', 'filename')]
              )
def load_data(contents, filenames):
    childrenUplMess = list()
    datasets = list()
    count = 0

    if contents is None:
        raise PreventUpdate

    if contents is not None:
        for i, (dataset, name) in enumerate(zip(it.islice(contents, 5), it.islice(filenames, 5))):
            uplMess, data = validate_dataset(i, dataset, name)
            childrenUplMess.extend([uplMess])
            datasets.extend([data])
            count += 1

        for j in range(count, 5):
            datasets.extend([{}])

        options = [{'label': lab, 'value': i} for i, lab in enumerate(filenames)]

        return childrenUplMess, datasets, options, ""

    return html.Div(['No dataset is uploaded.']), [{}, {}, {}, {}, {}], [], ""


@app.callback(Output('settings-missing-dialog', 'displayed'),
              [Input('dijkstra-run-button', 'n_clicks')],
              [State('dijkstra-start-dropdown', 'value'), State('dijkstra-weight-dropdown', 'value'),
               State('dataset-dropdown', 'value')])
def display_settings_reminder(n_clicks, start, weight, df_name):
    if n_clicks > 0 and None in (start, weight, df_name):
        return True
    else:
        return False


# Algorithm selection callbacks
@app.callback(Output('algorithm-dropdown', 'options'),
              [Input('algorithm-type-dropdown', 'value')])
def set_algorithm_options(selected_type):
    if selected_type is None:
        return []
    else:
        return [{'label': x, 'value': x} for x in algorithms[selected_type]]


@app.callback(Output('algorithm-dropdown', 'value'),
              [Input('algorithm-dropdown', 'options')])
def set_algorithm_value(available_options):
    if available_options is None or len(available_options) == 0:
        return None
    else:
        return available_options[0]['value']



@app.callback([Output('dijkstra-settings', 'style'),
               Output('bellman-ford-settings', 'style'),
               Output('floyd-warshall-settings', 'style'),
               Output('kruskal-settings', 'style'),
               Output('prim-settings', 'style'),
               Output('ford-fulkerson-settings', 'style')],
              [Input('algorithm-dropdown', 'value')])
def show_settings(selected_algorithm):
    alg_names = [x for y in algorithms.values() for x in y]
    result_dict = {x: {'display': 'none'} for x in alg_names}  # hide all divs
    if selected_algorithm in alg_names:
        result_dict[selected_algorithm] = {}  # remove the selected algorithm's div's style settings
    return [x for x in result_dict.values()]


# Dijkstra callbacks
# TODO account for possible changes in Graph object
@app.callback(Output('dijkstra-start-dropdown', 'options'),
              [Input('dijkstra-settings', 'style'),
               Input('dataset-dropdown', 'value')],
              [State('datasets', 'data')])
def set_dijkstra_start_options(style, i, datasets):
    if datasets is None or i is "":
        raise PreventUpdate

    if 'display' in style.keys() and style['display'] == 'none':
        return []
    else:
        # TODO fix graph/dataset met callbacks
        df = getDataFrame(datasets, i)
        G1 = createDiGraph(df, "")
        return [{'label': str(x), 'value': x} for x in sorted(G1.nodes)]


@app.callback(Output('dijkstra-start-dropdown', 'value'),
              [Input('dijkstra-start-dropdown', 'options')])
def set_dijkstra_start_value(options):
    if len(options) > 0:
        return options[0]['value']
    else:
        return None


@app.callback(Output('dijkstra-weight-dropdown', 'options'),
              [Input('dijkstra-settings', 'style'),
               Input('dataset-dropdown', 'value')],
              [State('datasets', 'data')])
def set_dijkstra_weight_options(style, i, datasets):
    if datasets is None or i is "":
        raise PreventUpdate

    df = getDataFrame(datasets, i)

    if 'display' in style.keys() and style['display'] == 'none':
        return []
    else:
        # TODO fix graph/dataset met callbacks
        return [{'label': x, 'value': x} for x in df.columns]


@app.callback([Output('dijkstra-weight-dropdown', 'style'), Output('dijkstra-weight-dropdown', 'value')],
              [Input('dijkstra-weight-radio', 'value')],
              [State('dijkstra-weight-dropdown', 'options')])
def set_dijkstra_weight_value(use_weight_column, options):
    if use_weight_column == 'yes':
        if len(options) > 0:
            if 'weight' in [x['value'] for x in options]:
                return {}, 'weight'
            else:
                return {}, options[0]['value']
    else:
        return {'display': 'none'}, ''


####################################
# VISUAL ANALYTICS PANEL CALLBACKS #
####################################
@app.callback([Output('dijkstra-info', 'data'),
               Output('shown-vis-graphs', 'children'),
               Output('graph-info', 'data'),
               Output('data-info', 'data')],
              [Input('dijkstra-run-button', 'n_clicks'),
               Input('graph-info', 'modified_timestamp')],
              [State('datasets', 'data'),
               State('dijkstra-start-dropdown', 'value'),
               State('dijkstra-weight-dropdown', 'value'),
               State('dataset-dropdown', 'value'),
               State('dijkstra-weight-radio', 'value'),
               State('dijkstra-info', 'data'),
               State('graph-info', 'data')])
def dijkstra(n_clicks, time_stamp, datasets, start, weight, i, use_weight_column, prev_data, iterations):
    if n_clicks > 0 and i not in ("", None):
        if prev_data in ("", None):
            df = getDataFrame(datasets, i)

            if use_weight_column == 'no':
                df['weight'] = 1  # list of ones
                weight = 'weight'

            G = createDiGraph(df, weight)
            return init_dijkstra(G, start), [], [], []
        else:
            data = {'t_added': time.time(),
                    'dataset_number': i,
                    'start': start,
                    'weight': weight,
                    'use_weight_column': use_weight_column,
                    'iterations': []}
            Q, dist, prev, neighs = prev_data
            if iterations is None:
                iterations = []
            output = iter_dijkstra(Q, dist, prev, neighs, iterations)

            data['iterations'] = output[3]

            return output[0], output[2], output[1], data
    else:
        raise PreventUpdate


def init_dijkstra(G, start):
    Q = list()
    dist = dict()
    prev = dict()
    neighs = dict()

    dist[start] = 0
    for v in G.nodes:
        neighs[v] = []
        for x in nx.neighbors(G, v):
            neighs[v].append([x, G.edges[v, x]['weight']])
        if v != start:
            dist[v] = float('inf')
        prev[v] = None
        heappush(Q, (dist[v], v))  # insert v, maintaining min heap property

    alg_output = [Q, clean_dict(dist), clean_dict(prev), clean_dict(neighs)]

    return alg_output


def iter_dijkstra(Q, dist, prev, neighs, iterations):
    data = dict()
    if Q not in (None, []):
        t_start = time.time()  # keep track of time
        for q in Q:
            if q[0] is None:
                q[0] = float("inf")

        dist_u, u = heappop(Q)  # extract minimum, maintaining min heap property
        neighs_u = neighs[str(u)]
        for v_inf in neighs_u:
            alt = dist_u + v_inf[1]  # dist(source, u) + dist(u, v)
            if dist[str(v_inf[0])] == "inf" or alt < dist[str(v_inf[0])]:
                dist[str(v_inf[0])] = alt
                prev[str(v_inf[0])] = u
                heappush(Q, [alt, v_inf[0]])

            t_elapsed = time.time() - t_start
            timestamp = datetime.datetime.now()
            memory_used = get_memory_used(Q, dist, prev, neighs_u)
            iterations.append([len(iterations), t_elapsed, memory_used])
            data = {'t': t_elapsed,
                    'memory': memory_used,
                    'Q': Q,
                    'u': u,
                    'v': neighs_u[0],
                    'neighs_u': neighs_u,
                    'dist': dist,
                    'prev': prev}
    else:
        raise PreventUpdate

    alg_output = [[Q, clean_dict(dist), clean_dict(prev), clean_dict(neighs)], iterations,
                    append_new_graph(
                        [],
                        name='Alg:dijkstra | Data:{} | Type:Runtime | Run:'.format(1),
                        data=[{'x': column(iterations, 0), 'y': column(iterations, 2), 'type': 'bar', 'name': 'SF'}],
                        xlab='iteration number',
                        ylab='time (s)'
                    ), data]

    return alg_output


##############
# APP LAYOUT #
##############
app.layout = html.Div(id='main-body', children=[
    # INPUT PANEL
    create_input_panel(),

    # VISUAL ANALYTICS PANEL
    create_visualisation_panel(),

    # OUTPUT PANEL
    create_output_panel(),

    # HIDDEN DIVS
    html.Div(id='saved-vis-graphs', style={'display': 'none'}),
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8000) # Might want to switch to processes=4
