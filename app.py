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
from operator import itemgetter
import copy

import networkx as nx
from algorithms import Dijkstra, Prim

####################
#   APP SETTINGS   #
####################
app = dash.Dash(__name__)

app.config['suppress_callback_exceptions'] = True
#app.scripts.config.serve_locally = True

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

##############
# APP LAYOUT #
##############
app.layout = html.Div(id='main-body', children=[
    # INPUT PANEL
    html.Div(className='input-panel', children=[
        # Data uploading and saving
        html.H1('Input panel'),
        html.Div('Supported file types are csv, json, '
                 'xls, dta, xpt and pkl.'),

        dcc.Loading(id="loading-data",
                    children=[dcc.Upload(
                                id='upload-field',
                                className='upload-data',
                                children=[html.Div(['Drag and Drop or ',
                                                    html.A('Select Files')
                                                    ]
                                            )],
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
            value=[x for x in algorithms.keys()][1]  # dumb workaround; dict.keys() doesn't support indexing
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
                dcc.Loading(id="dijkstra-running", children=[
                    html.Button(id='dijkstra-run-button', n_clicks=0, children='Run algorithm',
                                type='submit')]
                            , type="circle"),
            ]),
            # Bellman-Ford
            html.Div(id='bellman-ford-settings', style={'display': 'none'}, children=[
                html.H3('Settings'),
            ]),
            # Floyd Warshall
            html.Div(id='floyd-warshall-settings', style={'display': 'none'}, children=[
                html.H3('Settings')]),
            # Kruskal
            html.Div(id='kruskal-settings', style={'display': 'none'}, children=[
                html.H3('Settings')]),
            # Prim
            html.Div(id='prim-settings', style={'display': 'none'}, children=[
                html.H3('Settings'),
                html.P('Select start node'),
                dcc.Dropdown(id='prim-start-dropdown'),
                html.P('Edge weights:'),
                dcc.RadioItems(
                    id='prim-weight-radio',
                    options=[{'label': 'Edge weights all equal to 1', 'value': 'no'},
                             {'label': 'Use column in dataset as edge weights', 'value': 'yes'}],
                    value='no',
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.Dropdown(id='prim-weight-dropdown', style={'display': 'none'}),
                html.Button(id='prim-run-button', n_clicks=0, children='Run algorithm', type='submit')]),
            # Ford-Fulkerson
            html.Div(id='ford-fulkerson-settings', style={'display': 'none'}, children=[
                html.H3('Settings')])
        ]),
    ]),

    # VISUAL ANALYTICS PANEL
    dcc.Store(id='graph-info', storage_type='memory'),
    html.Div(className='vis-panel', children=[
        html.H1('Visual analytics panel'),
        dcc.Dropdown(id='show-graphs-dropdown', multi=True,
                     style={'vertical-align': 'middle'}),
        html.Div(id='shown-vis-graphs')
    ]),

    # OUTPUT PANEL
    html.Div(className='output-panel', children=[
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
            html.Div(id='network-graph')]),

        # Algorithm animation
        dcc.Store(id='stored-alg-output', storage_type='memory'),

        dcc.Dropdown(id='algorithm-runs-dropdown',
                     style={'width': '90%', 'display': 'inline-block', 'vertical-align': 'middle'}),
        html.Button(id='choose-run-button', children='Submit', type='submit',
                    style={'display': 'inline-block', 'vertical-align': 'middle'},
                    n_clicks=0, n_clicks_timestamp=0),

        html.Div(id='network-animation'),
        html.Div(id='iteration-range-slider-block', children=[
            dcc.RangeSlider(id='iteration-range-slider', pushable=0),
        ], style={'height': '40px'}),
        html.Div(id='animation-buttons', children=[
            html.Button(id='animation-run-button', children='Run animation', type='submit',
                        n_clicks=0, n_clicks_timestamp=0),
            html.Button(id='animation-stop-button', n_clicks=0, children='Stop animation', type='submit')
        ]),
        dcc.Interval(id='animation-interval', disabled=True)
    ]),

    # HIDDEN DIVS
    html.Div(id='saved-vis-graphs', style={'display': 'none'}),
    html.Div(id='prim-vis-graphs', style={'display': 'none'}),
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


def createGraph(df, weight):
    G1 = nx.Graph()
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


#########################
# INPUT PANEL CALLBACKS #
#########################
# Callback functions; functions that execute when something is changed
@app.callback([Output('upload-message', 'children'),
               Output('datasets', 'data'),
               Output('dataset-dropdown', 'options'),
               Output('dataset-dropdown', 'value')],
              [Input('upload-field', 'contents')],
              [State('upload-field', 'filename')])
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

        return childrenUplMess, datasets, options, options[-1]['value']

    return html.Div(['No dataset is uploaded.']), [{}, {}, {}, {}, {}], None, None


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
        return available_options[1]['value']

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
    if datasets is None or i in ("", None):
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
    if datasets is None or i in ("", None):
        raise PreventUpdate

    df = getDataFrame(datasets, i)

    if 'display' in style.keys() and style['display'] == 'none':
        return []
    else:
        # TODO fix graph/dataset met callbacks
        return [{'label': x, 'value': x} for x in df.columns]


@app.callback([Output('dijkstra-weight-dropdown', 'style'),
               Output('dijkstra-weight-dropdown', 'value')],
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


# Prim callbacks
# TODO account for possible changes in Graph object
@app.callback(Output('prim-start-dropdown', 'options'),
              [Input('prim-settings', 'style'),
               Input('dataset-dropdown', 'value')],
              [State('datasets', 'data')])
def set_prim_start_options(style, i, datasets):
    if datasets is None or i is "":
        raise PreventUpdate

    if 'display' in style.keys() and style['display'] == 'none':
        return []
    else:
        # TODO fix graph/dataset met callbacks
        df = getDataFrame(datasets, i)
        G1 = createDiGraph(df, "")
        return [{'label': str(x), 'value': x} for x in sorted(G1.nodes)]


@app.callback(Output('prim-start-dropdown', 'value'),
              [Input('prim-start-dropdown', 'options')])
def set_prim_start_value(options):
    if len(options) > 0:
        return options[0]['value']
    else:
        return None


@app.callback(Output('prim-weight-dropdown', 'options'),
              [Input('prim-settings', 'style'),
               Input('dataset-dropdown', 'value')],
              [State('datasets', 'data')])
def set_prim_weight_options(style, i, datasets):
    if datasets is None or i is "":
        raise PreventUpdate

    df = getDataFrame(datasets, i)

    if 'display' in style.keys() and style['display'] == 'none':
        return []
    else:
        # TODO fix graph/dataset met callbacks
        return [{'label': x, 'value': x} for x in df.columns]


@app.callback([Output('prim-weight-dropdown', 'style'),
               Output('prim-weight-dropdown', 'value')],
              [Input('prim-weight-radio', 'value')],
              [State('prim-weight-dropdown', 'options')])
def set_prim_weight_value(use_weight_column, options):
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
def append_new_graph(current_graphs, name, data, xlab, ylab):
    if current_graphs is None:
        name = name + str(1)  # id number corresponding to index in the list of graphs
        graph = dcc.Graph(
            id=name,
            figure={
                'data': data,
                'layout': go.Layout(
                    title={'text': name}, xaxis={'title': xlab}, yaxis={'title': ylab}
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
                    title={'text': name}, xaxis={'title': xlab}, yaxis={'title': ylab}
                )}
        )
        current_graphs.append(graph)
        return current_graphs


def clean_dict(mydict):
    for key, value in mydict.items():
        if type(key) is not str:
            try:
                mydict[str(key)] = mydict[key]
            except Exception as e:
                print(e)
                try:
                    mydict[repr(key)] = mydict[key]
                except Exception as e2:
                    print(e2, str(2))
                    pass
            del mydict[key]
    return mydict


@app.callback(Output('stored-alg-output', 'data'),
              [Input('dijkstra-run-button', 'n_clicks')],
              [State('dataset-dropdown', 'value'),
               State('datasets', 'data'),
               State('dijkstra-start-dropdown', 'value'),
               State('dijkstra-weight-dropdown', 'value'),
               State('stored-alg-output', 'data'),
               State('dijkstra-weight-radio', 'value')])
def run_dijkstra(n_clicks, df_name, datasets, start, weight, i, current_graphs, use_weight_column):
    if n_clicks > 0 and i not in ("", None):
        df = getDataFrame(datasets, i)

        if use_weight_column == 'no':
            df['weight'] = 1  # list of ones
            weight = 'weight'

        G = createDiGraph(df, weight)
        dijkstra = Dijkstra(G, start, weight).dijkstra()  # Dijkstra's algorithm as generator
        timestamp = []
        time = []
        memory_use = []

        for memory, t, tstamp, Q, u, neighs_u, dist, prev in dijkstra:
            time.append(t)
            timestamp.append(tstamp)
            memory_use.append(memory/1000000)  # in megabytes
            result = dist, prev

        current_graphs = append_new_graph(
            current_graphs,
            name='Alg:dijkstra | Data:{} | Type:Runtime | Run:'.format(df_name),
            data=[{'x': timestamp, 'y': time, 'type': 'bar', 'name': 'SF'}],
            xlab='iteration number',
            ylab='time (s)'
        )
        current_graphs = append_new_graph(
            current_graphs,
            name='Alg:dijkstra | Data:{} | Type:Memory | Run:'.format(df_name),
            data=[{'x': timestamp, 'y': memory_use, 'type': 'bar', 'name': 'SF'}],
            xlab='iteration number',
            ylab='memory (MB)'
        )
        return current_graphs
    else:
        raise PreventUpdate


@app.callback(Output('prim-vis-graphs', 'children'),
              [Input('prim-run-button', 'n_clicks')],
              [State('prim-start-dropdown', 'label'),
               State('datasets', 'data'),
               State('prim-start-dropdown', 'value'),
               State('prim-weight-dropdown', 'value'),
               State('dataset-dropdown', 'value'),
               State('prim-vis-graphs', 'children'),
               State('prim-weight-radio', 'value')])
def run_prim(n_clicks, df_name, datasets, start, weight, i, current_graphs, use_weight_column):
    if n_clicks > 0 and i not in ("", None):
        df = getDataFrame(datasets, i)

        if use_weight_column == 'no':
            df['weight'] = 1  # list of ones
            weight = 'weight'

        G = createGraph(df, weight)
        prim = Prim(G, start, weight).prim()  # Prim's algorithm as generator
        timestamp = []
        time = []
        memory_use = []

        for memory, t, tstamp, Q, v, neighs_u, dist, order in prim:
            time.append(t)
            timestamp.append(tstamp)
            memory_use.append(memory/1000000)  # in megabytes

        current_graphs = append_new_graph(
            current_graphs,
            name='Alg:Prim | Data:{} | Type:Runtime | Run:'.format(df_name),
            data=[{'x': timestamp, 'y': time, 'type': 'bar', 'name': 'SF'}],
            xlab='iteration number',
            ylab='time (s)'
        )
        current_graphs = append_new_graph(
            current_graphs,
            name='Alg:Prim | Data:{} | Type:Memory | Run:'.format(df_name),
            data=[{'x': timestamp, 'y': memory_use, 'type': 'bar', 'name': 'SF'}],
            xlab='iteration number',
            ylab='memory (MB)'
        )
        return current_graphs
    else:
        raise PreventUpdate


@app.callback(Output('show-graphs-dropdown', 'options'),
              [Input('saved-vis-graphs', 'children'),
               Input('prim-vis-graphs', 'children')])
def set_show_visualizations_dropdown_options(current_dijkstra_graphs, current_prim_graphs):
    current_graphs = []

    if current_dijkstra_graphs is not None:
        current_graphs.extend(current_dijkstra_graphs)

    if current_prim_graphs is not None:
        current_graphs.extend(current_prim_graphs)

    return [{'label': graph['props']['id'], 'value': graph['props']['id']} for graph in current_graphs]


@app.callback(Output('show-graphs-dropdown', 'value'),
              [Input('show-graphs-dropdown', 'options')],
              [State('show-graphs-dropdown', 'value')])
def set_show_visualizations_dropdown_value(options, current_values):
    if len(options) > 0:
        num_values_to_add = 2
        values_to_add = list([options[-i]['value'] for i in reversed(range(num_values_to_add))])
        if current_values is None:  # no value set
            return values_to_add  # set last added option as value
        else:  # at least one value present
            current_values.extend(values_to_add)
            return current_values


@app.callback(Output('shown-vis-graphs', 'children'),
              [Input('show-graphs-dropdown', 'value')],
              [State('saved-vis-graphs', 'children'),
               State('prim-vis-graphs', 'children')])
def hide_visualizations(selected_graph_ids, saved_dijkstra_graphs, saved_prim_graphs):
    result = []
    saved_graphs = []

    if saved_dijkstra_graphs is not None:
        saved_graphs.extend(saved_dijkstra_graphs)

    if saved_prim_graphs is not None:
        saved_graphs.extend(saved_prim_graphs)

    for graph in saved_graphs:
        if graph['props']['id'] in selected_graph_ids:
            result.append(graph.copy())
    return result


##########################
# OUTPUT PANEL CALLBACKS #
##########################
@app.callback([Output('network', 'style'),
               Output('network-graph', 'children')],
              [Input('draw-network-radio', 'value'),
               Input('network-layout-dropdown', 'value'),
               Input('dataset-dropdown', 'value')],
              [State('dataset-dropdown', 'label'),
               State('datasets', 'data')])
def show_network(draw_network, layout, i, df_name, datasets):
    if draw_network == 'yes' and i not in ("", None):
        df = getDataFrame(datasets, i)
        set_of_nodes = set(df['source']) | set(df['target'])  # union of the sets of source and target nodes
        elements = [{'data': {'id': x, 'label': x}} for x in set_of_nodes]  # nodes
        edges = [{'data': {'source': row['source'], 'target': row['target']}} for _, row in df[['source', 'target']].iterrows()]
        elements.extend(edges)

        return {}, [html.H3(df_name), cyto.Cytoscape(
            id='cytoscape-layout-1',
            elements=elements,
            style={'width': '100%', 'height': '350px'},
            layout={
                'name': layout
            })]
    else:
        return {'display': 'none'}, []


@app.callback(Output('algorithm-runs-dropdown', 'options'),
              [Input('stored-alg-output', 'data')])
def set_algorithm_runs_dropdown_options(runs):
    if runs is None:
        raise PreventUpdate
    else:
        return [{'label': x, 'value': x} for x in list(runs.keys())]


@app.callback(Output('algorithm-runs-dropdown', 'value'),
              [Input('algorithm-runs-dropdown', 'options')])
def set_algorithm_runs_dropdown_value(options):
    if options is None or len(options) == 0:
        raise PreventUpdate
    else:
        return options[0]['value']


@app.callback(Output('network-animation', 'children'),
              [Input('algorithm-runs-dropdown', 'value')],
              [State('stored-alg-output', 'data'),
               State('datasets', 'data')])
def draw_full_animation_network(run_name, run_data, datasets):
    if None in (run_name, run_data):
        raise PreventUpdate

    df = getDataFrame(datasets, run_data[run_name]['dataset_number'])
    weight = run_data[run_name]['weight']

    use_weight_column = run_data[run_name]['use_weight_column']
    if use_weight_column == 'no':
        df['weight'] = 1  # list of ones
        weight = 'weight'

    set_of_nodes = set(df['source']) | set(df['target'])  # union of the sets of source and target nodes
    elements = [{'data': {'id': x, 'label': x}} for x in set_of_nodes]  # nodes
    edges = [{'data': {'id': str(row['source']) + str(row['target']),
                       'label': 'inf',
                       'source': row['source'],
                       'target': row['target']}}
             for _, row in df[['source', 'target', weight]].iterrows()]
    elements.extend(edges)

    return cyto.Cytoscape(
        id='cytoscape-network-animation',
        elements=elements,
        layout={'name': 'cose'},
        style={'width': 'auto', 'height': '350px'},
        stylesheet=[
            {'selector': 'edge',
             'style': {
                 'opacity': 0.3,
                 'content': 'data(label)',
                 'curve-style': 'bezier',
                 'target-arrow-shape': 'vee',
                 'arrow-scale': 2}
             },
            {'selector': 'node',
             'style': {
                 'opacity': 0.3,
                 'content': 'data(id)'}
             }
        ]
    ),


@app.callback([Output('iteration-range-slider', 'min'),
               Output('iteration-range-slider', 'max'),
               Output('iteration-range-slider', 'marks'),
               Output('iteration-range-slider', 'value')],
              [Input('choose-run-button', 'n_clicks_timestamp'),
               Input('animation-run-button', 'n_clicks_timestamp')],
              [State('algorithm-runs-dropdown', 'value'),
               State('stored-alg-output', 'data'),
               State('iteration-range-slider', 'value'),
               State('animation-interval', 'n_intervals')])
def set_iteration_range_slider(t_choice, t_animation, run_name, run_data, iteration_range, n_intervals):
    if None in (run_data, run_name):
        raise PreventUpdate

    if t_choice > t_animation:
        iterations = run_data[run_name]['iterations']

        slider_min = 0
        slider_max = len(iterations)-1
        slider_value = [slider_min, slider_min, slider_max]
        if slider_max <= 10:
            step = 1
        else:
            step = int((slider_max+1 - slider_min) / 10)
        slider_marks = {i: str(i) for i in range(slider_min, slider_max+1, step)}
        slider_marks[slider_max] = str(slider_max)  # add last element in case range step is too high

        return slider_min, slider_max, slider_marks, slider_value
    else:
        print(n_intervals)
        return 0, 0, 0, 0


@app.callback(Output('cytoscape-network-animation', 'stylesheet'),
              [Input('iteration-range-slider', 'value')],
              [State('stored-alg-output', 'data'),
               State('algorithm-runs-dropdown', 'value'),
               State('datasets', 'data'),
               State('cytoscape-network-animation', 'stylesheet'),
               State('cytoscape-network-animation', 'elements')])
def draw_animation_iteration(iteration_range, run_data, run_name, datasets, stylesheet, elements):
    if None in (run_name, run_data, iteration_range, datasets, stylesheet):
        raise PreventUpdate

    i = int(iteration_range[1])  # iteration_range = (min, value, max)
    iteration = run_data[run_name]['iterations'][i]  # iteration data: dictionary containing Q, u, neighs_u, dist, prev
    df = getDataFrame(datasets, run_data[run_name]['dataset_number'])
    weight = run_data[run_name]['weight']

    use_weight_column = run_data[run_name]['use_weight_column']
    if use_weight_column == 'no':
        df['weight'] = 1  # list of ones
        weight = 'weight'

    all_nodes = [node['data']['id'] for node in elements if 'source' not in node['data'].keys()]
    all_edges = [edge['data']['id'] for edge in elements if 'source' in edge['data'].keys()]

    # key: edge id, value: edge weight (label)
    # assumption only one entry of edge (x,y) in a dataset
    edges = {str(src) + tg: df[(df['source'] == src) & (df['target'] == int(tg))][weight].iloc[0]
             for tg, src in iteration['prev'].items() if src is not None}
    nodes = [str(node) for node, dist in iteration["dist"].items() if dist < 1e12]

    if len(nodes) > 0:
        stylesheet.extend([
            {'selector': '#' + str(node),
             'style': {'opacity': 1}
             } if node in nodes else {
                'selector': '#' + str(node),
                'style': {'opacity': 0.3}
            } for node in all_nodes
        ])
    if len(edges) > 0:
        stylesheet.extend([
            {'selector': '#' + str(edge_id),
             'style': {
                 'content': edges[edge_id],
                 'opacity': 1}
             } if edge_id in edges.keys() else {
                'selector': '#' + str(edge_id),
                'style': {
                    'content': 'inf',
                    'opacity': 0.3}
            } for edge_id in all_edges
        ])

    start = str(run_data[run_name]['start'])
    stylesheet.append({'selector': '#' + start,
                       'style': {'content': start + '\n(start)'}})
    return stylesheet


@app.callback([Output('animation-interval', 'interval'),
               Output('animation-interval', 'n_intervals'),
               Output('animation-interval', 'max_intervals'),
               Output('animation-interval', 'disabled')],
              [Input('animation-run-button', 'n_clicks')],
              [State('iteration-range-slider', 'value')])
def run_animation(n_clicks, iteration_range):
    if n_clicks == 0 or iteration_range is None:
        raise PreventUpdate

    interval = 1000  # 1 second
    n_intervals = iteration_range[0]
    max_intervals = iteration_range[2] - iteration_range[0]
    return interval, n_intervals, max_intervals, False


#########################
# Testing with strings! #
#########################
@app.callback(Output('show-div-content', 'children'),
              [Input('dataset-dropdown', 'value')],
              [State('datasets', 'data')])
def show_div_content(i, datasets):
    if None in (i, datasets):
        raise PreventUpdate

    df = getDataFrame(datasets, i)
    return "{} has size {} MB".format(str(i), str(round(sys.getsizeof(df)/1000000, 2)))


if __name__ == '__main__':
    app.run_server(debug=True) # Might want to switch to processes=4
