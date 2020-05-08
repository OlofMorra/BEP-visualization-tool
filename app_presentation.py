# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State
import dash_table
from dash.exceptions import PreventUpdate
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
from operator import itemgetter

import networkx as nx

####################
#   APP SETTINGS   #
####################
app = dash.Dash(__name__)

app.config['suppress_callback_exceptions'] = True
app.scripts.config.serve_locally = True
app.css.config.serve_locally = False
app.css.append_css({'external_url': 'https://use.fontawesome.com/releases/v5.7.0/css/all.css'})

####################
# GLOBAL VARIABLES #
####################
# Use only for     #
# standard options #
# of cell          #
####################
algorithms = {
    'Shortest path': ['Dijkstra'],  # 'Bellman-Ford', 'Floyd-Warshall'
    'Minimal spanning tree': ['Prim']  # 'Kruskal'
    # 'Matching': ['Ford-Fulkerson']
}

network_layouts = ['breadthfirst', 'circle', 'concentric', 'cose', 'grid', 'random']

DELTA_T = 0.02
MARKER_SIZE = 14
LINE_WIDTH = 2
MAX_OPACITY = 0.9

GRAPH_TYPES = ['Dynamic', 'Memory', 'Time']

###################
# CREATING PANELS #
###################
def create_header():
    return html.Div(id='header', className='header', children=[
        html.Abbr(children=[html.I(id='hide-input-panel-button', n_clicks=0,
                                   style={'font-size': '24px', 'cursor': 'pointer'})],
                  title='Hide input panel', id='hide-input-panel-msg'),
        html.H1("Visualization of Graph Network Algorithms")
    ])

def create_input_panel():
    return html.Div(id='input-panel', className='input-panel', children=[
        # Data uploading and saving
        html.H2('Upload Dataset'),
        # html.Div('Supported file types are csv, json, '
        #          'xls, dta, xpt and pkl.'),

        dcc.Loading(id="loading-data",
                    children=[
                        html.Abbr(dcc.Upload(
                            id='upload-field',
                            className='upload-data',
                            children=[html.Div(['Drag and Drop or ',
                                                html.A('Select Files')])],
                            # Do allow multiple files to be uploaded
                            multiple=True
                        ), title='Supported file types are csv, json, xls, dta, xpt and pkl.'),
                        html.Div(id='upload-message')],
                    type="default"),
        dcc.Store(id='datasets', storage_type='memory'),

        # Algorithm selection
        html.Hr(),

        # Algorithm selection
        html.H2('Settings'),
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
                html.H2('Parameters'),
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
                html.Button(id='dijkstra-run-button', n_clicks=0, children='Run algorithm', type='submit',
                            style={'cursor': 'pointer'})
            ]),
            # Bellman-Ford
            html.Div(id='bellman-ford-settings', style={'display': 'none'}, children=[
                html.H2('Parameters')
            ]),
            # Floyd Warshall
            html.Div(id='floyd-warshall-settings', style={'display': 'none'}, children=[
                html.H2('Parameters')
            ]),
            # Kruskal
            html.Div(id='kruskal-settings', style={'display': 'none'}, children=[
                html.H2('Parameters')
            ]),
            # Prim
            html.Div(id='prim-settings', style={'display': 'none'}, children=[
                html.H2('Parameters'),
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
                html.Button(id='prim-run-button', n_clicks=0, children='Run algorithm', type='submit',
                            style={'cursor': 'pointer'})]),
            # Ford-Fulkerson
            html.Div(id='ford-fulkerson-settings', style={'display': 'none'}, children=[
                html.H2('Parameters')
            ]),
        ]),

        # Graph selection
        html.P('Select graph types'),
        dcc.Checklist(
            id='graph-type-selection',
            options=[{'label': x, 'value': x} for x in GRAPH_TYPES],
            value=GRAPH_TYPES
        ),

        html.Div(id='interval-graph-block', children=[
            html.P('Update speed:', style={'padding-right': '5px', 'display': 'inline-block'}),
            dcc.Input(id='graph-update-speed', type='number', value=10, min=1, style={'width': '33px'}),
            html.P('iterations', style={'padding-left': '5px', 'display': 'inline-block'}),
        ]),

        html.Div(id='interval-lines-block', children=[
            html.P('Show all lines per:', style={'padding-right': '5px', 'display': 'inline-block'}),
            dcc.Input(id='line-update-speed', type='number', value=0, min=0, style={'width': '33px'}),
            html.P('iterations', style={'padding-left': '5px', 'display': 'inline-block'}),
        ]),
    ])


def create_visualisation_panel():
    return html.Div(className='visualization-panel', children=[
        # Dijkstra store components
        dcc.Store(id='dijkstra-info', storage_type='memory'),
        dcc.Store(id='dijkstra-graph-info', storage_type='memory'),
        dcc.Store(id='dijkstra-data-info', storage_type='memory'),
        dcc.Store(id='dijkstra-dynamic-graph-info', storage_type='memory'),
        #Prim store components
        dcc.Store(id='prim-info', storage_type='memory'),
        dcc.Store(id='prim-graph-info', storage_type='memory'),
        dcc.Store(id='prim-data-info', storage_type='memory'),
        dcc.Store(id='prim-dynamic-graph-info', storage_type='memory'),
        html.Div(className='vis-panel', children=[
            html.H2('Visual Analytics'),
            dcc.Dropdown(id='show-graphs-dropdown', multi=True,
                         style={'vertical-align': 'middle'}),
            html.Div(id='shown-vis-graphs'),
            dcc.Store(id='selected-range', storage_type='memory'),
            dcc.Store(id='selected-range-1', storage_type='memory'),
            dcc.Store(id='selected-range-2', storage_type='memory'),
            dcc.Store(id='selected-range-3', storage_type='memory'),
            dcc.Store(id='selected-range-4', storage_type='memory'),
            dcc.Store(id='selected-range-5', storage_type='memory'),
            dcc.Store(id='selected-range-6', storage_type='memory'),
            dcc.Store(id='selected-range-7', storage_type='memory'),
            dcc.Store(id='selected-range-8', storage_type='memory'),
            dcc.Store(id='selected-range-9', storage_type='memory')
        ])])


def create_output_panel():
    return html.Div(className='output-panel', children=[
        html.H2('Network Animation'),
        # Algorithm animation
        dcc.Store(id='dijkstra-stored-alg-output', storage_type='memory', data={}),
        dcc.Store(id='prim-stored-alg-output', storage_type='memory', data={}),
        dcc.Store(id='stored-alg-output', storage_type='memory', data={}),

        dcc.Dropdown(id='algorithm-runs-dropdown'),

        html.Div([
            html.P('Layout:', style={'width': '55px', 'display': 'inline-block'}),
            dcc.Dropdown(id='animation-network-layout-dropdown',
                         options=[{'label': x, 'value': x} for x in network_layouts],
                         value=network_layouts[0],
                         style={'width': '80%', 'display': 'inline-block', 'vertical-align': 'middle'})]),
        html.Div(id='network-animation'),
        html.Div(id='iteration-range-slider-block', children=[
            dcc.RangeSlider(id='iteration-range-slider', pushable=1),
        ], style={'height': '40px', 'margin': '10px'}),
        html.Div(id='animation-buttons', children=[
            html.Button(id='animation-run-button', n_clicks=0, children='Run animation', type='submit',
                        style={'cursor': 'pointer'}),
            html.Button(id='animation-stop-button', n_clicks=0, children='Pause animation', type='submit',
                        style={'cursor': 'pointer'}),
            html.Button(id='animation-reset-button', n_clicks=0, children='Reset slider', type='submit',
                        style={'cursor': 'pointer'}),
        ]),
        html.Div(id='interval-block', children=[
            html.P('Animation speed:', style={'display': 'inline-block'}),
            dcc.Input(id='interval-length-input', type='number', value=1, min=1, max=99, style={'width': '33px'}),
            html.P('seconds', style={'display': 'inline-block'}),
        ]),
        html.Div(id='output-container-range-slider'),
        dcc.Interval(id='animation-interval', interval=1000, disabled=True),

        html.Div(id='animation-legend', children=[
            html.Hr(),
            html.I(id='hide-legend-button', n_clicks=0,
                   style={'font-size': '16px', 'cursor': 'pointer', 'display': 'inline-block', 'padding': '5px'}),
            html.H3("Legend  ", style={'display': 'inline-block'}),
            html.Div(className='animation-legend', id='animation-legend-table', children=[
                html.P("Green square"), html.P("start node"),
                html.P("Purple"), html.P("node with lowest distance from start node, so the algorithm currently considers its neighbours"),
                html.P("Blue"), html.P("node has been considered"),
                html.P("Orange"), html.P("neighbours of the purple node that are currently being considered"),
                html.P("Grey nodes"), html.P("nodes that haven't been looked at"),
                html.P("Grey edges"), html.P("edges that either haven't been looked at or are not part of any shortest path")
            ])]),

        html.Hr(),

        html.H2('Network information'),
        html.Div(id='network-statistics-dijkstra'),
        html.Div(id='network-statistics-prim'),

        html.Hr(),

        html.P('Draw network graph:', style={'display': 'inline'}),
        dcc.RadioItems(id='draw-network-radio',
                       options=[{'label': 'yes', 'value': 'yes'}, {'label': 'no', 'value': 'no'}],
                       value='no', style={'display': 'inline'}),
        html.Div(id='network', style={'display': 'none'}, children=[
            html.P('Layout:', style={'width': '55px', 'display': 'inline-block'}),
            dcc.Dropdown(id='network-layout-dropdown',
                         options=[{'label': x, 'value': x} for x in network_layouts],
                         value=network_layouts[0],
                         style={'width': '80%', 'display': 'inline-block', 'vertical-align': 'middle'}),
            dcc.Loading(html.Div(id='network-graph'))
        ]),
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


def append_new_time_series(current_graphs, length, name, id, data, xlab, ylab, run):
    name = name + str(run)  # id number corresponding to index in the list of graphs

    graph = dcc.Graph(
        id=id + '-graph-{}'.format(run),
        figure={
            'data': data,
            'layout': go.Layout(
                title={'text': name},
                xaxis={'title': xlab,
                       'rangeslider': {'visible': True},
                       'tickmode': 'linear',
                       'dtick' : int(length/50)},
                yaxis={'title': ylab}
            )}
    )

    current_graphs.append(graph)
    return current_graphs


def append_new_dynamic_graph(current_graphs, name, run):
    name = name + str(run)  # id number corresponding to index in the list of graphs

    graph = dcc.Graph(
        id="dynamic-graph-{}".format(run),
        figure={
            'data': [],
            'layout': go.Layout(
                title=name,
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=40, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           rangeslider=dict(visible=True)),
                yaxis=dict(showgrid=False, zeroline=False, title='Nodes',
                           showticklabels=False))}
    )

    current_graphs.append(graph)
    return current_graphs


def extend_dynamic_graph(dynamic_graph, data, lines):
    if len(data) > 0:
        if dynamic_graph['props']['figure']['data'] != []:
            l = dynamic_graph['props']['figure']['data'][-1]['x'][0]
        else:
            l = 0

        prev_iter = data[0]

        for i, iteration in enumerate(data, 1):
            if iteration == prev_iter and i % lines != 0:
                continue
            for node in iteration.keys():
                opacity = 0.2
                if iteration[node] is not None:
                    if iteration[node] != prev_iter[node]:
                        if prev_iter[node] is None:
                            marker_color = 'red'
                        else:
                            marker_color = ['blue', 'green']

                        change_marker = go.Scatter(
                            x=[l + (i*DELTA_T), l + (i*DELTA_T)],
                            y=[prev_iter[node], iteration[node]],
                            hoverinfo='text',
                            text='Change in iteration: ' + str(int(l/DELTA_T+i)),
                            mode='markers',
                            marker=dict(
                                size=MARKER_SIZE,
                                color=marker_color
                            )
                        )
                        dynamic_graph['props']['figure']['data'].append(change_marker)
                        opacity = MAX_OPACITY
                    if opacity == MAX_OPACITY or i % lines == 0:
                        color = get_color(int(iteration[node]))
                        edge_trace = go.Scatter(
                            x=[l + (i*DELTA_T), l + (i*DELTA_T) + 1],
                            y=[iteration[node], int(node)],
                            line=dict(width=LINE_WIDTH, color=color),
                            hoverinfo='text',
                            text='[' + str(iteration[node]) + ',' + str(node) + ']',
                            mode='lines',
                            opacity=opacity)
                        dynamic_graph['props']['figure']['data'].append(edge_trace)
            prev_iter = iteration

    return dynamic_graph


def get_color(node):
    red = ((int(node)%5) * 50) % 255
    green = ((int(node)%7) * 35) % 255
    blue = ((int(node)%11) * 22) % 255

    return 'rgb(' + str(red) + ', ' + str(green) + ', ' + str(blue) + ')'


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
    return result/1000000


def column(matrix, i):
    return [row[i] for row in matrix]


#########################
# INPUT PANEL CALLBACKS #
#########################
@app.callback([Output('input-panel', 'style'),
               Output('hide-input-panel-button', 'className'),
               Output('hide-input-panel-msg', 'title'),
               Output('content', 'className'),
               Output('cytoscape-network-animation', 'style')],
              [Input('hide-input-panel-button', 'n_clicks')])
def hide_input_panel(n_clicks):
    if n_clicks % 2 != 0:
        return {'display': 'none'}, 'fas fa-chevron-right', 'Show input panel', 'content_hidden', \
               {'width': '100%', 'height': '600px'}
    else:
        return {}, 'fas fa-chevron-left', 'Hide input panel', 'content', \
               {'width': '99%', 'height': '600px'}


# Callback functions; functions that execute when something is changed
@app.callback([Output('upload-message', 'children'),
               Output('datasets', 'data'),
               Output('dataset-dropdown', 'options'),
               Output('dataset-dropdown', 'value')],
              [Input('upload-field', 'contents')],
              [State('upload-field', 'filename'),
               State('datasets', 'data'),
               State('dataset-dropdown', 'options')])
def load_data(contents, filenames, datasets, options):
    childrenUplMess = list()
    count = 0

    if datasets is None:
        datasets = list()
        options = list()

    if contents is None:
        raise PreventUpdate

    if contents is not None:
        for i, (dataset, name) in enumerate(zip(it.islice(contents, 5), it.islice(filenames, 5))):
            uplMess, data = validate_dataset(i, dataset, name)
            childrenUplMess.extend([uplMess])
            options.append({'label': name, 'value': len(options)})
            datasets.extend([data])
            count += 1

        return childrenUplMess, datasets, options, options[-1]['value']

    return html.Div(['No dataset is uploaded.']), [], [], ""


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
               # Output('bellman-ford-settings', 'style'),
               # Output('floyd-warshall-settings', 'style'),
               Output('prim-settings', 'style'),
               # Output('kruskal-settings', 'style'),
               # Output('ford-fulkerson-settings', 'style'),
               ],
              [Input('algorithm-dropdown', 'value')])
def show_settings(selected_algorithm):
    alg_names = [x for y in algorithms.values() for x in y]
    result_dict = {x: {'display': 'none'} for x in alg_names}  # hide all divs
    if selected_algorithm in alg_names:
        result_dict[selected_algorithm] = {}  # remove the selected algorithm's div's style settings
    return [x for x in result_dict.values()]


# Dijkstra callbacks
# TODO account for possible changes in Graph object
@app.callback([Output('dijkstra-start-dropdown', 'options'),
               Output('network-statistics-dijkstra', 'children')],
              [Input('dijkstra-settings', 'style'),
               Input('dataset-dropdown', 'value')],
              [State('datasets', 'data')])
def set_dijkstra_start_options(style, i, datasets):
    if datasets is None:
        raise PreventUpdate

    if 'display' in style.keys() and style['display'] == 'none':
        return [], []
    elif i is None:
        return [], html.Div('No dataset is uploaded.')
    else:
        df = getDataFrame(datasets, i)
        G1 = createDiGraph(df, "")
        info = html.Div([html.P('Number of nodes: ' + str(G1.number_of_nodes())),
                      html.P('Number of edges: ' + str(G1.number_of_edges()))])

        return [{'label': str(x), 'value': x} for x in sorted(G1.nodes)], info


@app.callback(Output('dijkstra-start-dropdown', 'value'),
              [Input('dijkstra-start-dropdown', 'options')])
def set_dijkstra_start_value(options):
    if options is not None:
        if len(options) > 0:
            return options[0]['value']
        else:
            return None
    else:
        return None


@app.callback(Output('dijkstra-weight-dropdown', 'options'),
              [Input('dijkstra-settings', 'style'),
               Input('dataset-dropdown', 'value')],
              [State('datasets', 'data')])
def set_dijkstra_weight_options(style, i, datasets):
    if datasets is None:
        raise PreventUpdate

    if 'display' in style.keys() and style['display'] == 'none':
        return []
    elif i is None:
        return []
    else:
        df = getDataFrame(datasets, i)
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
@app.callback([Output('prim-start-dropdown', 'options'),
               Output('network-statistics-prim', 'children')],
              [Input('prim-settings', 'style'),
               Input('dataset-dropdown', 'value')],
              [State('datasets', 'data')])
def set_prim_start_options(style, i, datasets):
    if datasets is None or i is "":
        raise PreventUpdate

    if 'display' in style.keys() and style['display'] == 'none':
        return [], []
    elif i is None:
        return [], html.Div('No dataset is uploaded.')
    else:
        # TODO fix graph/dataset met callbacks
        df = getDataFrame(datasets, i)
        G1 = createDiGraph(df, "")
        info = html.Div([html.P('Number of nodes: ' + str(G1.number_of_nodes())),
                      html.P('Number of edges: ' + str(G1.number_of_edges()))])
        return [{'label': str(x), 'value': x} for x in sorted(G1.nodes)], info


@app.callback(Output('prim-start-dropdown', 'value'),
              [Input('prim-start-dropdown', 'options')])
def set_prim_start_value(options):
    if options is not None:
        if len(options) > 0:
            return options[0]['value']
        else:
            return None
    else:
        raise PreventUpdate


@app.callback(Output('prim-weight-dropdown', 'options'),
              [Input('prim-settings', 'style'),
               Input('dataset-dropdown', 'value')],
              [State('datasets', 'data')])
def set_prim_weight_options(style, i, datasets):
    if datasets is None or i in ("", None):
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
# Dijkstra callbacks
@app.callback(Output('dijkstra-run-button', 'n_clicks'),
              [Input('dijkstra-info', 'data')])
def end_dijkstra(data):
    print("Data:" + data.__str__())
    if data in (None, []):
        raise PreventUpdate
    elif data[0][0][0] is None:
        return 0
    else:
        return 2


@app.callback([Output('dijkstra-info', 'data'),
               Output('dijkstra-data-info', 'data'),
               Output('dijkstra-dynamic-graph-info', 'data'),
               Output('dijkstra-stored-alg-output', 'data'),
               Output('dijkstra-graph-info', 'data')],
              [Input('dijkstra-run-button', 'n_clicks'),
               Input('dijkstra-data-info', 'modified_timestamp')],
              [State('datasets', 'data'),
               State('dijkstra-start-dropdown', 'value'),
               State('dijkstra-weight-dropdown', 'value'),
               State('dataset-dropdown', 'value'),
               State('dijkstra-weight-radio', 'value'),
               State('dijkstra-info', 'data'),
               State('dijkstra-graph-info', 'data'),
               State('dijkstra-stored-alg-output', 'data'),
               State('prim-stored-alg-output', 'data'),
               State('graph-type-selection', 'value'),
               State('graph-update-speed', 'value'),
               State('dijkstra-dynamic-graph-info', 'data')])
def dijkstra(n_clicks, time_stamp, datasets, start, weight, i, use_weight_column, prev_data, iterations,
             dijkstra_data, prim_data, graph_types, update_speed, dynamic_graph_info):
    if 'Dynamic' in graph_types:
        dynamic = True
    else:
        dynamic = False

    if n_clicks > 0 and i not in ("", None):
        if n_clicks == 1:
            df = getDataFrame(datasets, i)

            if use_weight_column == 'no':
                df['weight'] = 1  # list of ones
                weight = 'weight'

            G = createDiGraph(df, weight)

            if dijkstra_data is not None:
                l = len(dijkstra_data)
            else:
                l = 0

            if prim_data is not None:
                l = l + len(prim_data)

            dijkstra_data["Dijkstra run {}".format(l+1)] = {'Run': l+1,
                                                't_added': time.time(),
                                                'dataset_number': i,
                                                'start': start,
                                                'weight': weight,
                                                'use_weight_column': use_weight_column,
                                                'iterations': list()}
            alg_output = init_dijkstra(G, start)

            return alg_output, [], [alg_output[2]], dijkstra_data, []
        else:
            time.sleep(0.5)
            run = list(dijkstra_data.keys())[-1]

            Q, dist, prev, neighs = prev_data
            if iterations is None:
                iterations = []

            output = iter_dijkstra(Q, dist, prev, neighs, iterations, dynamic, update_speed)

            dijkstra_data[run]['iterations'].extend(output[2])

            if len(dynamic_graph_info) == 1:
                dynamic_graph_info.extend(output[3])
            else:
                dynamic_graph_info = output[3]

            return output[0], dijkstra_data, dynamic_graph_info, dijkstra_data, output[1]
    else:
        return [], [], dynamic_graph_info, dijkstra_data, iterations


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


def iter_dijkstra(Q, dist, prev, neighs, iterations, dynamic, i):
    dynamic_graph_data = []
    iter_data = list()
    for i in range(0, i):
        if Q not in (None, []) and Q[0][0] != float("inf"):
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

                t_elapsed = (time.time() - t_start)*1000
                timestamp = datetime.datetime.now()
                memory_used = get_memory_used(Q, dist, prev, neighs_u)

                iterations.append([len(iterations), t_elapsed, memory_used])
                iter_data.append({'t': t_elapsed,
                        'memory': memory_used,
                        'Q': Q,
                        'u': u,
                        'v': v_inf[0],
                        'neighs_u': column(neighs_u,0),
                        'dist': dist.copy(),
                        'prev': prev.copy()})

                if dynamic:
                    dynamic_graph_data.append(prev.copy())
        else:
            break

    alg_output = [[Q, clean_dict(dist), clean_dict(prev), clean_dict(neighs)], iterations,
                  iter_data, dynamic_graph_data]

    return alg_output


# Prim callbacks
@app.callback(Output('prim-run-button', 'n_clicks'),
              [Input('prim-info', 'data')])
def end_prim(data):
    if data in (None, []):
        raise PreventUpdate
    elif data[0][0][0] is None:
        return 0
    else:
        return 2


@app.callback([Output('prim-info', 'data'),
               Output('prim-data-info', 'data'),
               Output('prim-dynamic-graph-info', 'data'),
               Output('prim-stored-alg-output', 'data'),
               Output('prim-graph-info', 'data')],
              [Input('prim-run-button', 'n_clicks'),
               Input('prim-data-info', 'modified_timestamp')],
              [State('datasets', 'data'),
               State('prim-start-dropdown', 'value'),
               State('prim-weight-dropdown', 'value'),
               State('dataset-dropdown', 'value'),
               State('prim-weight-radio', 'value'),
               State('prim-info', 'data'),
               State('prim-graph-info', 'data'),
               State('dijkstra-stored-alg-output', 'data'),
               State('prim-stored-alg-output', 'data'),
               State('graph-update-speed', 'value'),
               State('prim-dynamic-graph-info', 'data')])
def prim(n_clicks, time_stamp, datasets, start, weight, i, use_weight_column, prev_data, iterations,
            dijkstra_data, prim_data, update_speed, dynamic_graph_info):
    if n_clicks > 0 and i not in ("", None):
        if n_clicks == 1:
            df = getDataFrame(datasets, i)

            if use_weight_column == 'no':
                df['weight'] = 1  # list of ones
                weight = 'weight'

            G = createGraph(df, weight)

            if dijkstra_data is not None:
                l = len(dijkstra_data)
            else:
                l = 0

            if prim_data is not None:
                l = l + len(prim_data)

            prim_data["Prim run {}".format(l+1)] = {'Run': l+1,
                                                't_added': time.time(),
                                                'dataset_number': i,
                                                'start': start,
                                                'weight': weight,
                                                'use_weight_column': use_weight_column,
                                                'iterations': list()}

            return init_prim(G, start), [], [], prim_data, []
        else:
            time.sleep(0.5)
            run = list(prim_data.keys())[-1]

            Q, dist, prev, neighs, MST = prev_data
            if iterations is None:
                iterations = []

            output = iter_prim(Q, dist, prev, neighs, MST, iterations, update_speed)

            prim_data[run]['iterations'].extend(output[2])

            return output[0], prim_data, output[3], prim_data, output[1]
    else:
        return [], [], dynamic_graph_info, prim_data, iterations


def init_prim(G, start):
    Q = list()
    dist = dict()
    prev = dict()
    neighs = dict()
    MST = list()

    dist[start] = 0
    for v in G.nodes:
        neighs[v] = []
        for x in nx.neighbors(G, v):
            neighs[v].append([x, G.edges[v, x]['weight']])
        if v != start:
            dist[v] = float('inf')
        prev[v] = None
        heappush(Q, (dist[v], v))  # insert v, maintaining min heap property

    alg_output = [Q, clean_dict(dist), clean_dict(prev), clean_dict(neighs), MST]

    return alg_output


def iter_prim(Q, dist, prev, neighs, MST, iterations, i):
    dynamic_graph_data =[]
    iter_data = list()
    for i in range(0, i):
        if Q not in (None, []) and Q[0][0] != float("inf"):
            t_start = time.time()  # keep track of time
            for q in Q:
                if q[0] is None:
                    q[0] = float("inf")

            dist_u, u = heappop(Q)  # extract minimum, maintaining min heap property

            if u not in MST:
                MST.append(u)
                neighs_u = neighs[str(u)]
                for neighbor in neighs_u:
                    if neighbor[0] not in MST:
                        dist[str(neighbor[0])] = neighbor[1]
                        heappush(Q, [int(neighbor[1]), neighbor[0]])
                    else:
                        if prev[str(u)] is None:
                            prev[str(u)] = neighbor[0]
                            dist[str(u)] = neighbor[1]
                        elif int(neighbor[1]) < int(dist[str(u)]):
                            prev[str(u)] = neighbor[0]

                    t_elapsed = (time.time() - t_start)*1000
                    timestamp = datetime.datetime.now()
                    memory_used = get_memory_used(Q, dist, prev, neighs_u, MST)

                    iterations.append([len(iterations), t_elapsed, memory_used])
                    iter_data.append({'t': t_elapsed,
                            'memory': memory_used,
                            'Q': Q.copy(),
                            'u': u,
                            'v': neighbor[0],
                            'neighs_u': column(neighs_u,0),
                            'dist': dist.copy(),
                            'prev': prev.copy()})

                    dynamic_graph_data.append(prev.copy())

    alg_output = [[Q, clean_dict(dist), clean_dict(prev), clean_dict(neighs), MST], iterations,
                  iter_data, dynamic_graph_data]

    return alg_output


# Add graph based on saved data
@app.callback(Output('saved-vis-graphs', 'children'),
              [Input('dijkstra-graph-info', 'modified_timestamp'),
               Input('prim-graph-info', 'modified_timestamp')],
              [State('saved-vis-graphs', 'children'),
               State('dijkstra-graph-info', 'data'),
               State('dijkstra-dynamic-graph-info', 'data'),
               State('prim-graph-info', 'data'),
               State('prim-dynamic-graph-info', 'data'),
               State('graph-type-selection', 'value'),
               State('line-update-speed', 'value')])
def add_graphs(t1, t2, current_graphs, dijkstra_graph_data,
                        dijkstra_dynamic_graph_data, prim_graph_data,
                        prim_dynamic_graph_data, graph_types, lines):
    print("Properties: " + dash.callback_context.triggered.__str__())
    if lines == 0:
        lines = float('inf')
    if 'dijkstra' in dash.callback_context.triggered[0]['prop_id']:
        graph_data = dijkstra_graph_data
        dynamic_graph_data = dijkstra_dynamic_graph_data
        name = "Dijkstra"
    elif 'prim' in dash.callback_context.triggered[0]['prop_id']:
        graph_data = prim_graph_data
        dynamic_graph_data = prim_dynamic_graph_data
        name = "Prim"
    else:
        raise PreventUpdate


    print("Graph data: " + graph_data.__str__())
    print("Graph data: " + dynamic_graph_data.__str__())
    if graph_data is not None:
        if not graph_data:
            if current_graphs is None:
                current_graphs = []
                run = 1
            else:
                run = int(current_graphs[-1]['props']['id'][-1]) + 1

            if 'Dynamic' in graph_types:
                current_graphs = append_new_dynamic_graph(current_graphs,
                                                      name='Alg: ' + name + ' | Data: {} | Type: Dynamic Graph | Run: '.format(1),
                                                      run=run)
            if 'Time' in graph_types:
                current_graphs = append_new_time_series(current_graphs, 1,
                                                        name='Alg: ' + name + ' | Data: {} | Type: Time | Run: '.format(1),
                                                        id='time',
                                                        data=[],
                                                        xlab='iteration number',
                                                        ylab='Time (ms)',
                                                        run=run)
            if 'Memory' in graph_types:
                current_graphs = append_new_time_series(current_graphs, 1,
                                                        name='Alg: ' + name + ' | Data: {} | Type: Memory | Run: '.format(1),
                                                        id='memory',
                                                        data=[],
                                                        xlab='iteration number',
                                                        ylab='Memory (mb)',
                                                        run=run)
        else:
            print("Graphs: " + current_graphs.__str__())
            if current_graphs is not None:
                run = int(current_graphs[-1]['props']['id'][-1])
                if 'Memory' in graph_types and 'Time' in graph_types:
                    current_graphs = current_graphs[:-2]
                elif 'Memory' in graph_types or 'Time' in graph_types:
                    current_graphs = current_graphs[:-1]

                if 'Dynamic' in graph_types:
                    current_graphs[-1] = extend_dynamic_graph(
                        current_graphs[-1],
                        data=dynamic_graph_data,
                        lines=lines
                    )
            else:
                run = 1

            length = len(graph_data)

            if 'Time' in graph_types:
                time_trace = go.Scatter(x=column(graph_data, 0), y=column(graph_data, 1))
                current_graphs = append_new_time_series(
                    current_graphs,
                    length,
                    name='Alg: ' + name + ' | Data: {} | Type: Runtime | Run: '.format(1),
                    id='time',
                    data=[time_trace],
                    xlab='iteration number',
                    ylab='Time (ms)',
                    run=run
                )

            if 'Memory' in graph_types:
                memory_trace = go.Scatter(x=column(graph_data, 0), y=column(graph_data, 2))
                current_graphs = append_new_time_series(
                    current_graphs,
                    length,
                    name='Alg: ' + name + ' | Data: {} | Type: Memory | Run: '.format(1),
                    id='memory',
                    data=[memory_trace],
                    xlab='iteration number',
                    ylab='Memory (MB)',
                    run=run
                )

        return current_graphs
    else:
        raise PreventUpdate


# Visualisation callbacks
@app.callback([Output('show-graphs-dropdown', 'options'),
               Output('show-graphs-dropdown', 'value')],
              [Input('saved-vis-graphs', 'children')],
              [State('graph-type-selection', 'value')])
def set_show_visualizations_dropdown_options(current_graphs, graph_types):
    if current_graphs is None:
        current_graphs = []

    options = [{'label': graph['props']['figure']['layout']['title']['text'], 'value': graph['props']['id']} for graph in current_graphs]
    value = []

    if len(graph_types) == 3:
        value = [options[-3]["value"], options[-2]["value"], options[-1]["value"]]
    elif len(graph_types) == 2:
        value = [options[-2]["value"], options[-1]["value"]]
    elif len(graph_types) == 1:
        value = [options[-1]["value"]]

    return options, value


@app.callback(Output('shown-vis-graphs', 'children'),
              [Input('show-graphs-dropdown', 'value')],
              [State('saved-vis-graphs', 'children')])
def hide_visualizations(selected_graph_ids, saved_graphs):
    result = []

    if saved_graphs is None:
        saved_graphs = []

    for graph in saved_graphs:
        if graph['props']['id'] in selected_graph_ids:
            result.append(graph.copy())
    return result


for i in range(1,10):
    @app.callback(Output('selected-range-{}'.format(i), 'data'),
                  [Input('dynamic-graph-{}'.format(i), 'relayoutData'),
                   Input('memory-graph-{}'.format(i), 'relayoutData'),
                   Input('time-graph-{}'.format(i), 'relayoutData')])
    def update_range(dynamic_graph_range, memory_graph_range, time_graph_range):
        trigger_comp = dash.callback_context.triggered[0]['prop_id'][:-15]
        slide_range = []

        if trigger_comp == "dynamic-graph":
            if dynamic_graph_range is None:
                raise PreventUpdate
            elif len(dynamic_graph_range) != 1:
                x1 = (dynamic_graph_range["xaxis.range[0]"])/DELTA_T
                x2 = (dynamic_graph_range["xaxis.range[1]"])/DELTA_T
                slide_range = [x1, x2]
        elif trigger_comp == "time-graph":
            if time_graph_range is None:
                raise PreventUpdate
            elif len(time_graph_range) != 1:
                slide_range = [time_graph_range["xaxis.range[0]"], time_graph_range["xaxis.range[1]"]]
        elif trigger_comp == "memory-graph":
            if memory_graph_range is None:
                raise PreventUpdate
            elif len(memory_graph_range) != 1:
                slide_range = [memory_graph_range["xaxis.range[0]"], memory_graph_range["xaxis.range[1]"]]

        return slide_range


@app.callback(Output('selected-range', 'data'),
              [Input("selected-range-1", 'modified_timestamp'),
               Input("selected-range-2", 'modified_timestamp'),
               Input("selected-range-3", 'modified_timestamp'),
               Input("selected-range-4", 'modified_timestamp'),
               Input("selected-range-5", 'modified_timestamp'),
               Input("selected-range-6", 'modified_timestamp'),
               Input("selected-range-7", 'modified_timestamp'),
               Input("selected-range-8", 'modified_timestamp'),
               Input("selected-range-9", 'modified_timestamp')],
              [State("selected-range-1", 'data'),
               State("selected-range-2", 'data'),
               State("selected-range-3", 'data'),
               State("selected-range-4", 'data'),
               State("selected-range-5", 'data'),
               State("selected-range-6", 'data'),
               State("selected-range-7", 'data'),
               State("selected-range-8", 'data'),
               State("selected-range-9", 'data')])
def set_range_of_iter_slider(*args):
    trigger_comp = int(dash.callback_context.triggered[0]['prop_id'][-20])
    iter_range = []

    if args[8 + trigger_comp] is None:
        raise PreventUpdate

    if args[8 + trigger_comp] != []:
        x1 = int(args[8+trigger_comp][0])
        x2 = int(args[8 + trigger_comp][1]) + 1
        iter_range = [x1, x2]

    return iter_range


##########################
# OUTPUT PANEL CALLBACKS #
##########################
@app.callback(Output('stored-alg-output', 'data'),
              [Input('dijkstra-stored-alg-output', 'modified_timestamp'),
               Input('prim-stored-alg-output', 'modified_timestamp')],
              [State('dijkstra-stored-alg-output', 'data'),
               State('prim-stored-alg-output', 'data')])
def combine_store_components(t1, t2, dijkstra, prim):
    return {**dijkstra, **prim}


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
        edges = [{'data': {'source': row['source'],
                           'target': row['target']}} for _, row in df[['source', 'target']].iterrows()]
        elements.extend(edges)

        return {}, [html.H3(df_name), cyto.Cytoscape(
            id='cytoscape-layout-1',
            elements=elements,
            stylesheet=[
                {'selector': 'edge',
                 'style': {
                     'width': '2',
                     # 'content': 'data(label)',
                     'curve-style': 'bezier'}
                 },
                {'selector': 'node',
                 'style': {
                     'content': 'data(label)'}
                 }
            ],
            style={'width': '99%', 'height': '350px'},
            layout={
                'name': layout,
                'animate': True,
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
        i = 0
        for run in options:
            if int(run['value'][-1]) > i:
                i = int(run['value'][-1])
                last = run
        return last['value']


@app.callback(Output('cytoscape-network-animation', 'layout'),
              [Input('animation-network-layout-dropdown', 'value'),
               Input('hide-input-panel-button', 'n_clicks')],
              [State('network-animation', 'children')])
def set_animation_network_layout(layout, n_clicks, cur_animation):
    if cur_animation is None:
        raise PreventUpdate

    # triggers = [trigger['prop_id'] for trigger in dash.callback_context.triggered]
    # if 'hide-input-panel-button.n_clicks' in triggers:
    #     return {'name': layout, 'animate': True, 'fit': True, 'padding': 3}

    return {'name': layout, 'animate': True}


@app.callback(Output('output-container-range-slider', 'children'),
            [Input('iteration-range-slider', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


@app.callback(Output('network-animation', 'children'),
              [Input('algorithm-runs-dropdown', 'value')],
              [State('stored-alg-output', 'data'),
               State('datasets', 'data'),
               State('animation-network-layout-dropdown', 'value')])
def draw_full_animation_network(run_name, run_data, datasets, layout):
    if None in (run_name, run_data, layout):
        raise PreventUpdate

    df = getDataFrame(datasets, run_data[run_name]['dataset_number'])
    weight = run_data[run_name]['weight']

    use_weight_column = run_data[run_name]['use_weight_column']
    if use_weight_column == 'no':
        df['weight'] = 1  # list of ones
        weight = 'weight'

    set_of_nodes = set(df['source']) | set(df['target'])  # union of the sets of source and target nodes
    elements = [{'data': {'id': str(x), 'label': str(x)}} for x in set_of_nodes]  # nodes
    edges = [{'data': {'id': "{}-{}".format(int(row['source']), int(row['target'])),
                       'label': 'inf',
                       'source': int(row['source']),
                       'target': int(row['target'])}}
             for _, row in df[['source', 'target', weight]].iterrows()]
    elements.extend(edges)

    stylesheet = [{'selector': 'edge',
                   'style': {
                       'width': '2',
                       'content': 'data(label)',
                       'curve-style': 'bezier'}},
                  {'selector': 'node',
                   'style': {
                       'content': 'data(label)'}}]

    if 'dijkstra' in run_name.lower():
        stylesheet.append({'selector': 'edge', 'style': {'target-arrow-shape': 'triangle'}})

    return cyto.Cytoscape(
        id='cytoscape-network-animation',
        elements=elements,
        layout={'name': layout, 'animate': True},
        zoom=1,
        minZoom=0.2,
        maxZoom=5,
        boxSelectionEnabled=True,
        style={'width': 'auto', 'height': '600px'},
        stylesheet=stylesheet
    )


@app.callback([Output('iteration-range-slider', 'min'),
               Output('iteration-range-slider', 'max')],
              [Input('algorithm-runs-dropdown', 'value')],
              [State('stored-alg-output', 'data')])
def set_iteration_range_min_max_marks(run_name, run_data):
    if None in (run_data, run_name):
        raise PreventUpdate

    iterations = run_data[run_name]['iterations']
    slider_min = -1
    slider_max = len(iterations)
    return slider_min, slider_max


@app.callback([Output('iteration-range-slider', 'value'),
               Output('animation-interval', 'disabled')],
              [Input('iteration-range-slider', 'min'),
               Input('iteration-range-slider', 'max'),
               Input('animation-stop-button', 'n_clicks'),
               Input('animation-run-button', 'n_clicks'),
               Input('animation-reset-button', 'n_clicks'),
               Input('animation-interval', 'n_intervals'),
               Input('selected-range', 'modified_timestamp')],
              [State('iteration-range-slider', 'value'),
               State('selected-range', 'data')])
def set_iteration_range_slider_value(slider_min, slider_max, stop_clicks, run_clicks,
                                     reset_clicks, n_intervals, t, cur_iter_range, sel_range):
    if None in [trigger['value'] for trigger in dash.callback_context.triggered]:
        raise PreventUpdate

    triggers = [trigger['prop_id'] for trigger in dash.callback_context.triggered]

    if 'iteration-range-slider.min' in triggers or 'iteration-range-slider.max' in triggers:
        return [slider_min, slider_min + 1, slider_max], True
    elif 'animation-stop-button.n_clicks' in triggers:  # stop animation
        return cur_iter_range, True
    elif 'animation-run-button.n_clicks' in triggers:  # start animation by enabling the interval
        return cur_iter_range, False
    elif 'animation-interval.n_intervals' in triggers:  # interval increments value of the slider unless end is reached
        range_min = cur_iter_range[0]
        range_value = cur_iter_range[1]
        range_max = cur_iter_range[2]
        if range_value < range_max - 1:  # last mark is not part of the animation
            return [range_min, range_value + 1, range_max], False
        else:
            return cur_iter_range, True
    elif 'animation-reset-button.n_clicks' in triggers:
        return [slider_min, slider_min + 1, slider_max], True  # first mark is not part of the animation
    elif 'selected-range.modified_timestamp' in triggers:
        if slider_min is None:
            raise PreventUpdate
        elif sel_range == []:
            return [slider_min, slider_min + 1, slider_max], True
        else:
            if sel_range[0] < slider_min:
                sel_range[0] = slider_min
            if sel_range[1] >= slider_max:
                sel_range[1] = slider_max
            if sel_range[0] == sel_range[1]:
                sel_range[0] = sel_range[1] - 1
            return [sel_range[0], sel_range[0] + 1, sel_range[1]], True


@app.callback(Output('iteration-range-slider', 'marks'),
              [Input('iteration-range-slider', 'value')],
              [State('iteration-range-slider', 'min'),
               State('iteration-range-slider', 'max')])
def set_iteration_range_marks(iteration_range, slider_min, slider_max):
    if None in [trigger['value'] for trigger in dash.callback_context.triggered]:
        raise PreventUpdate

    range_min = iteration_range[0]
    range_max = iteration_range[2]
    if slider_max <= 10:
        step = 1
    else:
        step = int((slider_max + 1 - slider_min) / 10)
    slider_marks = {i: str(i) for i in range(range_min + 1, range_max, step)}
    slider_marks[range_max - 1] = str(range_max - 1)  # add last element in case range step is too high
    slider_marks[range_min] = '['
    slider_marks[range_max] = ']'

    return slider_marks


@app.callback(Output('animation-interval', 'interval'),
              [Input('interval-length-input', 'value')])
def set_interval_length(length):
    return length * 1000  # in seconds


@app.callback(Output('cytoscape-network-animation', 'stylesheet'),
              [Input('iteration-range-slider', 'value')],
              [State('stored-alg-output', 'data'),
               State('algorithm-runs-dropdown', 'value'),
               State('datasets', 'data')])
def draw_animation_iteration(iteration_range, run_data, run_name, datasets):
    if None in (run_name, run_data, iteration_range, datasets) or run_data[run_name]['iterations'] == []:
        raise PreventUpdate

    COL_VISITED = '#2980B9'  # blue
    COL_UNVISITED = '#95A5A6'  # grey
    COL_CURRENT_NODE = 'purple'
    COL_CONSIDERING = '#DC7633'  # orange

    i = int(iteration_range[1])  # iteration_range = (min, value, max)

    if i >= len(run_data[run_name]['iterations']):
        raise PreventUpdate

    iteration = run_data[run_name]['iterations'][i]  # iteration data: dictionary containing Q, u, neighs_u, dist, prev
    df = getDataFrame(datasets, run_data[run_name]['dataset_number'])
    edges_visited = []

    weight = run_data[run_name]['weight']

    use_weight_column = run_data[run_name]['use_weight_column']
    if use_weight_column == 'no':
        df['weight'] = 1  # list of ones
        weight = 'weight'

    if 'prim' in run_name.lower():  # add all edges from target to source, because the graph is undirected
        df_rev = pd.DataFrame({'source': df['target'], 'target': df['source'], 'weight': df['weight']})
        df = df.append(df_rev, ignore_index=True)

    nodes_visited = [str(node) for node, dist in iteration["dist"].items() if dist != 'inf']
    nodes_unvisited = [str(node) for node, dist in iteration["dist"].items() if dist == 'inf']

    edges_visited = [{'id': "{}-{}".format(int(src), int(tg)),
                      'weight': df[(df['source'] == int(src)) & (df['target'] == int(tg))][weight].iloc[0]
                      } for tg, src in iteration['prev'].items() if src is not None]
    edges_unvisited = ["{}-{}".format(src, tg) for tg, src in iteration['prev'].items() if src is None]

    if 'prim'in run_name.lower():  # add all edges from target to source, because the graph is undirected
        edges_visited.extend([{'id': "{}-{}".format(int(tg), int(src)),
                               'weight': df[(df['source'] == int(src)) & (df['target'] == int(tg))][weight].iloc[0]
                               } for tg, src in iteration['prev'].items() if src is not None])
        edges_unvisited.extend(["{}-{}".format(tg, src) for tg, src in iteration['prev'].items() if src is None])

    if 'dijkstra' in run_name.lower():
        stylesheet = [{'selector': 'edge',
                       'style': {
                           # 'content': 'data(label)',
                           'curve-style': 'bezier',
                           'target-arrow-shape': 'triangle'
                       }}]
    else:
        stylesheet = [{'selector': 'edge',
                       'style': {
                           # 'content': 'data(label)',
                           'curve-style': 'bezier'
                       }}]

    stylesheet.extend([  # visited nodes
        {'selector': '#' + node_id,
         'style': {
             'content': node_id,
             'background-color': COL_VISITED}
         } for node_id in nodes_visited
    ])
    stylesheet.extend([  # unvisited nodes
        {'selector': '#' + node_id,
         'style': {
             'content': node_id,
             'background-color': COL_UNVISITED}
         } for node_id in nodes_unvisited
    ])
    stylesheet.extend([  # visited edges
        {'selector': '#{}'.format(edge['id']),
         'style': {
             'width': 5,
             'content': edge['weight'],
             'target-arrow-color': COL_VISITED,
             'line-color': COL_VISITED}
         } for edge in edges_visited
    ])
    stylesheet.extend([  # unvisited edges
        {'selector': '#{}'.format(edge_id),
         'style': {
             'width': 2,
             'content': 'inf',
             'target-arrow-color': COL_UNVISITED,
             'line-color': COL_UNVISITED}
         } for edge_id in edges_unvisited
    ])
    start = run_data[run_name]['start']
    stylesheet.append({
        'selector': '#{}'.format(start),
        'style': {
            'background-color': 'darkgreen',
            'shape': 'rectangle'}})
    stylesheet.append({
        'selector': '#{}'.format(iteration['u']),
        'style': {
            'background-color': COL_CURRENT_NODE}})
    stylesheet.extend([  # current node u
        {'selector': '#{}'.format(neighbor),
         'style': {
             'background-color': COL_CONSIDERING}
         } for neighbor in iteration['neighs_u']
    ])
    stylesheet.append({  # edges from u to neighbours that are still in Q
        'selector': '#{}-{}'.format(iteration['u'], iteration['v']),
        'style': {
            'width': 10,
            'target-arrow-color': COL_CONSIDERING,
            'line-color': COL_CONSIDERING}
    })
    return stylesheet


@app.callback([Output('animation-legend-table', 'style'),
               Output('hide-legend-button', 'className')],
              [Input('hide-legend-button', 'n_clicks')])
def hide_legend(n_clicks):
    if n_clicks % 2 == 0:
        return {'display': 'none'}, 'fas fa-chevron-down'
    else:
        return {}, 'fas fa-chevron-up'


##############
# APP LAYOUT #
##############
app.layout = html.Div(className='main', id='main', children=[
    # Header
    create_header(),

    # Content
    html.Div(className='content', id='content', children=[
        # INPUT PANEL
        create_input_panel(),

        # VISUAL ANALYTICS PANEL
        create_visualisation_panel(),

        # OUTPUT PANEL
        create_output_panel(),

        # # HIDDEN DIVS
        html.Div(id='saved-vis-graphs', style={'display': 'none'})
    ])
])


if __name__ == '__main__':
    app.run_server(debug=True, port=7000)  # Might want to switch to processes=4
