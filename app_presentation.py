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
from operator import itemgetter

import networkx as nx

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
                dcc.Store(id='dynamic-graph-info', storage_type='memory'),
                html.Div(className='vis-panel', children=[
                    html.H1('Visual analytics panel'),
                    dcc.Dropdown(id='show-graphs-dropdown', multi=True,
                                 style={'vertical-align': 'middle'}),
                    html.Div(id='shown-vis-graphs'),
                    dcc.Store(id='selected-range', storage_type='memory')
                ])])


def create_output_panel():
    return html.Div(className='output-panel', children=[
            html.H1('Network animation'),

            # Algorithm animation
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
            ], style={'height': '40px'}),
            html.Div(id='animation-buttons', children=[
                html.Button(id='animation-run-button', n_clicks=0, children='Run animation', type='submit'),
                html.Button(id='animation-stop-button', n_clicks=0, children='Pause animation', type='submit'),
                html.Button(id='animation-reset-button', n_clicks=0, children='Reset slider', type='submit'),
            ]),
            html.Div(id='interval-block', children=[
                html.P('Animation speed:', style={'display': 'inline-block'}),
                dcc.Input(id='interval-length-input', type='number', value=1, min=1, max=99, style={'width': '33px'}),
                html.P('seconds', style={'display': 'inline-block'}),
            ]),
            dcc.Interval(id='animation-interval', interval=1000, disabled=True),

            html.Div(id='show-div-content'),
            html.P('Draw network graph:'),
            dcc.RadioItems(id='draw-network-radio',
                           options=[{'label': 'yes', 'value': 'yes'}, {'label': 'no', 'value': 'no'}],
                           value='no'),
            html.Div(id='network', style={'display': 'none'}, children=[
                html.P('Layout:', style={'width': '55px', 'display': 'inline-block'}),
                dcc.Dropdown(id='network-layout-dropdown',
                             options=[{'label': x, 'value': x} for x in network_layouts],
                             value=network_layouts[0],
                             style={'width': '80%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                html.Div(id='network-graph')]),
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


def append_new_time_series(current_graphs, length, name, id, data, xlab, ylab):
    if current_graphs is None:
        i = 1
        current_graphs = []
    else:
        i = int((len(current_graphs) + 2)/3)

    name = name + str(i)  # id number corresponding to index in the list of graphs

    graph = dcc.Graph(
        id=id + '-graph-{}'.format(i),
        figure={
            'data': data,
            'layout': go.Layout(
                title={'text': name},
                xaxis={'title': xlab,
                       'rangeslider': {'visible': True},
                       'tickmode': 'linear',
                       'dtick' : int(length/50)},
                yaxis={'title': ylab},
                clickmode='event+select',
                dragmode='select'
            )}
    )

    current_graphs.append(graph)
    return current_graphs


def append_new_dynamic_graph(current_graphs, name, data, xlab):
    if current_graphs is None:
        i = 1
        current_graphs = []
    else:
        i = int((len(current_graphs) + 3)/3)

    name = name + str(i)  # id number corresponding to index in the list of graphs

    length = len(data)

    graph = dcc.Graph(
        id="dynamic-graph-{}".format(i),
        figure={
            'data': [],
            'layout': go.Layout(
                title=name,
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                clickmode='event+select',
                dragmode='select')}
    )

    current_graphs.append(graph)
    return current_graphs


def extend_dynamic_graph(dynamic_graph, data):
    if len(data) > 0:
        if dynamic_graph['props']['figure']['data'] != []:
            l = dynamic_graph['props']['figure']['data'][-1]['x'][0]
        else:
            l = 0

        for i, iteration in enumerate(data, 1):
            yellow = ((l+i)*10) % 255
            for node in iteration.keys():
                if iteration[node] is not None:
                    edge_trace = go.Scatter(
                        x=[l + (i*0.5), l + (i*0.5) + 1],
                        y=[iteration[node], int(node)],
                        line=dict(width=0.5, color=('rgb(255,' + str(yellow) + ',0)')),
                        hoverinfo='none',
                        mode='lines')
                    dynamic_graph['props']['figure']['data'].append(edge_trace)

    return dynamic_graph


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

        return childrenUplMess, datasets, options, options[0]['value']

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
    if datasets is None:
        raise PreventUpdate

    if 'display' in style.keys() and style['display'] == 'none':
        return []
    elif i is None:
        return []
    else:
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


####################################
# VISUAL ANALYTICS PANEL CALLBACKS #
####################################
@app.callback(Output('dijkstra-run-button', 'n_clicks'),
              [Input('dijkstra-info', 'data')])
def end_dijkstra(data):
    if data in (None, []):
        raise PreventUpdate
    elif data[0][0][0] is None:
        return 0
    else:
        return 2


@app.callback([Output('dijkstra-info', 'data'),
               Output('data-info', 'data'),
               Output('dynamic-graph-info', 'data'),
               Output('stored-alg-output', 'data'),
               Output('graph-info', 'data')],
              [Input('dijkstra-run-button', 'n_clicks'),
               Input('data-info', 'modified_timestamp')],
              [State('datasets', 'data'),
               State('dijkstra-start-dropdown', 'value'),
               State('dijkstra-weight-dropdown', 'value'),
               State('dataset-dropdown', 'value'),
               State('dijkstra-weight-radio', 'value'),
               State('dijkstra-info', 'data'),
               State('graph-info', 'data'),
               State('dynamic-graph-info', 'data'),
               State('stored-alg-output', 'data')])
def dijkstra(n_clicks, time_stamp, datasets, start, weight, i, use_weight_column, prev_data, iterations,
             dynamic_graph_data, data):
    if n_clicks > 0 and i not in ("", None):
        if n_clicks == 1:
            df = getDataFrame(datasets, i)

            if use_weight_column == 'no':
                df['weight'] = 1  # list of ones
                weight = 'weight'

            G = createDiGraph(df, weight)

            if data is not None:
                l = len(data)
            else:
                l = 0

            data["Dijkstra run {}".format(l+1)] = {'Run': l+1,
                                                't_added': time.time(),
                                                'dataset_number': i,
                                                'start': start,
                                                'weight': weight,
                                                'use_weight_column': use_weight_column,
                                                'iterations': list()}

            return init_dijkstra(G, start), [], [], data, []
        else:
            run = list(data.keys())[-1]

            Q, dist, prev, neighs = prev_data
            if iterations is None:
                iterations = []
            output = iter_dijkstra(Q, dist, prev, neighs, iterations, dynamic_graph_data)

            data[run]['iterations'].extend(output[2])

            return output[0], data, output[3], data, output[1]
    else:
        return [], [], dynamic_graph_data, data, iterations


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


def iter_dijkstra(Q, dist, prev, neighs, iterations, dynamic_graph_data):
    dynamic_graph_data = []
    iter_data = list()
    for i in range(0, 30):
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
                        'v': neighs_u[0][0],
                        'neighs_u': column(neighs_u,0),
                        'dist': dist.copy(),
                        'prev': prev.copy()})

            dynamic_graph_data.append(prev.copy())

    alg_output = [[Q, clean_dict(dist), clean_dict(prev), clean_dict(neighs)], iterations,
                  iter_data, dynamic_graph_data]

    return alg_output


@app.callback(Output('saved-vis-graphs', 'children'),
              [Input('graph-info', 'modified_timestamp')],
              [State('saved-vis-graphs', 'children'),
               State('graph-info', 'data'),
               State('dynamic-graph-info', 'data')])
def add_dijkstra_graphs(t, current_graphs, graph_data, dynamic_graph_data):
    if graph_data is not None:
        if graph_data == []:
            current_graphs = append_new_dynamic_graph(current_graphs,
                                                      name='Alg: Dijkstra | Data: {} | Type: Dynamic Graph | Run:'.format(1),
                                                      data=[],
                                                      xlab='iteration number')
            current_graphs = append_new_time_series(current_graphs, 1, name='1', id='time', data=[], xlab='', ylab='')
            current_graphs = append_new_time_series(current_graphs, 1, name='2', id='memory', data=[], xlab='', ylab='')
        else:
            current_graphs = current_graphs[:-2]

            current_graphs[-1] = extend_dynamic_graph(
                current_graphs[-1],
                data=dynamic_graph_data
            )

            time_trace = go.Scatter(x=column(graph_data, 0), y=column(graph_data, 1))
            memory_trace = go.Scatter(x=column(graph_data, 0), y=column(graph_data, 2))

            length = len(graph_data)

            current_graphs = append_new_time_series(
                current_graphs,
                length,
                name='Alg: Dijkstra | Data: {} | Type: Runtime | Run:'.format(1),
                id='time',
                data=[time_trace],
                xlab='iteration number',
                ylab='time (ms)'
            )

            current_graphs = append_new_time_series(
                current_graphs,
                length,
                name='Alg: Dijkstra | Data: {} | Type: Memory | Run:'.format(1),
                id='memory',
                data=[memory_trace],
                xlab='iteration number',
                ylab='Memory (MB)'
            )

        return current_graphs
    else:
        raise PreventUpdate


@app.callback([Output('show-graphs-dropdown', 'options'),
               Output('show-graphs-dropdown', 'value')],
              [Input('saved-vis-graphs', 'children')])
def set_show_visualizations_dropdown_options(current_dijkstra_graphs):
    current_graphs = []

    if current_dijkstra_graphs is not None:
        current_graphs.extend(current_dijkstra_graphs)

    options = [{'label': graph['props']['figure']['layout']['title']['text'], 'value': graph['props']['id']} for graph in current_graphs]
    value = [options[-3]["value"], options[-2]["value"], options[-1]["value"]] if len(options) > 2 else []

    return options, value


@app.callback(Output('shown-vis-graphs', 'children'),
              [Input('show-graphs-dropdown', 'value')],
              [State('saved-vis-graphs', 'children')])
def hide_visualizations(selected_graph_ids, saved_dijkstra_graphs):
    result = []
    saved_graphs = []

    if saved_dijkstra_graphs is not None:
        saved_graphs.extend(saved_dijkstra_graphs)

    for graph in saved_graphs:
        if graph['props']['id'] in selected_graph_ids:
            result.append(graph.copy())
    return result


@app.callback(Output('selected-range', 'data'),
              [Input('dynamic-graph-1', 'relayoutData'),
               Input('memory-graph-1', 'relayoutData'),
               Input('time-graph-1', 'relayoutData')])
def update_range(selection0, selection1, selection2):
    print(selection0)
    print(selection1)
    print(selection2)
    return []



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
        edges = [{'data': {'source': row['source'],
                           'target': row['target']}} for _, row in df[['source', 'target']].iterrows()]
        elements.extend(edges)

        return {}, [html.H3(df_name), cyto.Cytoscape(
            id='cytoscape-layout-1',
            elements=elements,
            style={'width': '100%', 'height': '350px'},
            layout={
                'name': layout,
                'animate': True
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


@app.callback(Output('cytoscape-network-animation', 'layout'),
              [Input('animation-network-layout-dropdown', 'value')],
              [State('network-animation', 'children')])
def set_animation_network_layout(layout, cur_animation):
    if cur_animation is None:
        raise PreventUpdate

    return {'name': layout, 'animate': True}


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

    return cyto.Cytoscape(
        id='cytoscape-network-animation',
        elements=elements,
        layout={'name': layout, 'animate': True},
        style={'width': 'auto', 'height': '600px'},
        stylesheet=[
            {'selector': 'edge',
             'style': {
                 'content': 'data(label)'}
             },
            {'selector': 'node',
             'style': {
                 'content': 'data(label)'}
             }
        ]
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
               Input('animation-interval', 'n_intervals')],
              [State('iteration-range-slider', 'value')])
def set_iteration_range_slider_value(slider_min, slider_max, stop_clicks, run_clicks, reset_clicks, n_intervals, cur_iter_range):
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

    COL_VISITED = '#2980B9'
    COL_UNVISITED = '#95A5A6'
    COL_CURRENT_NODE = '#D35400'
    COL_CONSIDERING = '#DC7633'

    i = int(iteration_range[1])  # iteration_range = (min, value, max)
    iteration = run_data[run_name]['iterations'][i]  # iteration data: dictionary containing Q, u, neighs_u, dist, prev
    df = getDataFrame(datasets, run_data[run_name]['dataset_number'])

    weight = run_data[run_name]['weight']

    use_weight_column = run_data[run_name]['use_weight_column']
    if use_weight_column == 'no':
        df['weight'] = 1  # list of ones
        weight = 'weight'

    nodes_visited = [str(node) for node, dist in iteration["dist"].items() if dist != 'inf']
    nodes_unvisited = [str(node) for node, dist in iteration["dist"].items() if dist == 'inf']
    # assumption only one entry of edge (x,y) in a dataset
    edges_visited = [{'id': "{}-{}".format(int(src), int(tg)),
                      'weight': df[(df['source'] == int(src)) & (df['target'] == int(tg))][weight].iloc[0]
                      } for tg, src in iteration['prev'].items() if src is not None]
    edges_unvisited = ["{}-{}".format(src, tg) for tg, src in iteration['prev'].items() if src is None]

    stylesheet = []
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
             'content': edge['weight'],
             'line-color': COL_VISITED}
         } for edge in edges_visited
    ])
    stylesheet.extend([  # unvisited edges
        {'selector': '#{}'.format(edge_id),
         'style': {
             'content': 'inf',
             'line-color': COL_UNVISITED}
         } for edge_id in edges_unvisited
    ])
    start = run_data[run_name]['start']
    stylesheet.append({
        'selector': '#{}'.format(start),
        'style': {
            'shape': 'rectangle'}})
    stylesheet.append({
        'selector': '#{}'.format(iteration['u']),
        'style': {
            'background-color': COL_CURRENT_NODE}})
    stylesheet.append({  # current node u
        'selector': '#{}'.format(iteration['v']),
        'style': {
            'background-color': COL_CONSIDERING}
    })
    stylesheet.append({  # edges from u to neighbours that are still in Q
        'selector': '#{}-{}'.format(iteration['u'], iteration['v']),
        'style': {
            'line-color': COL_CONSIDERING}
    })
    return stylesheet


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
    app.run_server(debug=True, port=7000)  # Might want to switch to processes=4
