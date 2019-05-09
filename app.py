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

import networkx as nx
from algorithms import Dijkstra

app = dash.Dash(__name__)

####################
# GLOBAL VARIABLES #
####################
N = 1000
M = 100
df1 = pd.DataFrame({'source': [random.randint(1, M) for _ in range(N)],
                    'target': [random.randint(1, M) for _ in range(N)],
                    'weight': [random.lognormvariate(mu=0, sigma=0.5) for _ in range(N)]})
G1 = nx.DiGraph(df1)
df2 = pd.DataFrame({'source': [random.randint(1, M) for _ in range(N)],
                    'target': [random.randint(1, M) for _ in range(N)],
                    'weight': [random.lognormvariate(mu=0, sigma=0.5) for _ in range(N)]})
G2 = nx.DiGraph(df2)
datasets = {'df1': df1, 'df2': df2}

algorithms = {
    'Shortest path': ['Dijkstra', 'Bellman-Ford', 'Floyd-Warshall'],
    'Minimal spanning tree': ['Kruskal', 'Prim'],
    'Matching': ['Ford-Fulkerson']
}

network_layouts = ['breadthfirst', 'circle', 'concentric', 'cose', 'grid', 'random']

app.config['suppress_callback_exceptions'] = True

##############
# APP LAYOUT #
##############
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
                                ]
                        )],
            # Do allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='upload-message'),
        dcc.Store(id='dataset1', storage_type='memory'),
        dcc.Store(id='dataset2', storage_type='memory'),
        dcc.Store(id='dataset3', storage_type='memory'),
        dcc.Store(id='dataset4', storage_type='memory'),
        dcc.Store(id='dataset5', storage_type='memory'),

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
        html.H1('Visual analytics panel', style={'display': 'inline-block'}),
        dcc.Dropdown(id='show-graphs-dropdown', multi=True,
                     style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'middle', 'textAlign': 'right'}),
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
            html.Div(id='network-graph')])
    ]),

    # HIDDEN DIVS
    html.Div(id='saved-vis-graphs', style={'display': 'none'}),
])


# FUNCTIONS
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

#########################
# INPUT PANEL CALLBACKS #
#########################
# Callback functions; functions that execute when something is changed
@app.callback([Output('upload-message', 'children'),
               Output('dataset1', 'data'),
               Output('dataset2', 'data'),
               Output('dataset3', 'data'),
               Output('dataset4', 'data'),
               Output('dataset5', 'data')],
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

        return childrenUplMess, datasets[0], datasets[1], datasets[2], datasets[3], datasets[4]

    try:
        return html.Div(['No dataset is uploaded.']), {}, {}, {}, {}, {}
    except Exception as e:
        print(e)

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

@app.callback(Output('saved-vis-graphs', 'children'),
              [Input('dijkstra-run-button', 'n_clicks')],
              [State('dijkstra-start-dropdown', 'value'), State('dijkstra-weight-dropdown', 'value'),
               State('dataset-dropdown', 'value'), State('saved-vis-graphs', 'children'),
               State('dijkstra-weight-radio', 'value')])
def run_dijkstra(n_clicks, start, weight, df_name, current_graphs, use_weight_column):
    if n_clicks > 0:
        df = datasets[df_name]

        if use_weight_column == 'no':
            df['weight'] = 1  # list of ones
            weight = 'weight'

        G = nx.DiGraph(df)
        dijkstra = Dijkstra(G, start, weight).dijkstra()  # Dijkstra's algorithm as generator
        time = []
        memory_use = []

        for t, memory, Q, u, neighs_u, dist, prev in dijkstra:
            time.append(t)
            memory_use.append(memory/1000000)  # in megabytes
            result = dist, prev

        current_graphs = append_new_graph(
            current_graphs,
            name='Alg:dijkstra | Data:{} | Run:'.format(df_name),
            data=[{'x': [i for i in range(len(time))], 'y': time, 'type': 'bar', 'name': 'SF'}],
            xlab='iteration number',
            ylab='time (s)'
        )
        current_graphs = append_new_graph(
            current_graphs,
            name='Alg:dijkstra | Data:{} | Run:'.format(df_name),
            data=[{'x': [i for i in range(len(memory_use))], 'y': memory_use, 'type': 'bar', 'name': 'SF'}],
            xlab='iteration number',
            ylab='memory (MB)'
        )
        return current_graphs


@app.callback(Output('show-graphs-dropdown', 'options'),
              [Input('saved-vis-graphs', 'children')])
def set_show_visualizations_dropdown_options(current_graphs):
    if current_graphs is None:
        return []
    else:
        return [{'label': graph['props']['id'], 'value': graph['props']['id']} for graph in current_graphs]


@app.callback(Output('show-graphs-dropdown', 'value'),
              [Input('show-graphs-dropdown', 'options')],
              [State('show-graphs-dropdown', 'value')])
def set_show_visualizations_dropdown_value(options, current_values):
    if len(options) > 0:
        if current_values is None:  # no value set
            return list([options[-1]['value']])  # set last added option as value
        else:  # at least one value present
            current_values.append(options[-1]['value'])
            return current_values


@app.callback(Output('shown-vis-graphs', 'children'),
              [Input('show-graphs-dropdown', 'value')],
              [State('saved-vis-graphs', 'children')])
def hide_visualizations(selected_graph_ids, saved_graphs):
    result = []
    if saved_graphs is None:
        return result
    else:
        for graph in saved_graphs:
            if graph['props']['id'] in selected_graph_ids:
                result.append(graph.copy())
        return result


##########################
# OUTPUT PANEL CALLBACKS #
##########################
@app.callback([Output('network', 'style'), Output('network-graph', 'children')],
              [Input('draw-network-radio', 'value'), Input('network-layout-dropdown', 'value')],
              [State('dataset-dropdown', 'value')])
def show_network(draw_network, layout, df_name):
    if draw_network == 'yes':
        df = datasets[df_name]
        set_of_nodes = set(df['source']) | set(df['target'])  # union of the sets of source and target nodes
        nodes = [{'data': {'id': x, 'label': x}} for x in set_of_nodes]
        edges = [{'data': {'source': row['source'], 'target': row['target']}} for _, row in df[['source', 'target']].iterrows()]
        elements = nodes + edges

        return {}, [html.H3(df_name), cyto.Cytoscape(
            id='cytoscape-layout-1',
            elements=elements,
            style={'width': '100%', 'height': '350px'},
            layout={
                'name': layout
            })]
    else:
        return {'display': 'none'}, []


#########################
# Testing with strings! #
#########################
@app.callback(Output('show-div-content', 'children'),
              [Input('dataset-dropdown', 'value')])
def show_div_content(value):
    return "{} has size {} MB".format(value, str(round(sys.getsizeof(df1)/1000000, 2)))


if __name__ == '__main__':
    app.run_server(debug=True)