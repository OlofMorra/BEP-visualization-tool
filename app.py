# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd
import base64
import datetime
import io

import networkx as nx

app = dash.Dash(__name__)

df = None

app.layout = html.Div(id='main-body', children=[
    html.Div(className='input-panel', children=[
        html.H1('Input panel'),
        html.P('Select file type'),
        dcc.Dropdown(id="dropdown-filetype",
            options=[
                {'label': 'CSV', 'value': 'csv'},
                {'label': 'XLS', 'value': 'xls'}
            ],
            value='csv'
        ),
        html.P('Select split character'),
        dcc.Dropdown(id="dropdown-splittype",
            options=[
                {'label': 'Space', 'value': ' '},
                {'label': ',', 'value': ','},
                {'label': "Tab", 'value': '\t'}
            ],
            value=' '
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
        html.Div(id='head-data-upload')]
    ),
    html.Div(className='vis-panel',
        children=[]
    ),
    html.Div(className='output-panel',
             children=['Right panel']
    )
])


# Callback functions; functions that execute when something is changed
@app.callback(Output('head-data-upload', 'children'),
              [Input('upload-field', 'contents'),
               Input('dropdown-splittype', 'value'),
               Input('dropdown-filetype', 'value')],
              )
def load_data(content, split, filetype):
    if (content == None):
        return html.Div([
            'Select a data file.'
        ])
    elif (split == None):
        return html.Div([
            'Select a split character.'
        ])
    elif (filetype == None):
        return html.Div([
            'Select a filetype.'
        ])
    else:
        return html.Div([
            'Loaded' + content['name']
        ])

    if filetype == 'csv':
        df = pd.read_csv(content)
    elif filetype == 'xls':
        df = pd.read_excel(content)

if __name__ == '__main__':
    app.run_server(debug=True)
