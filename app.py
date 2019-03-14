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

# Functions
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

# Callback functions; functions that execute when something is changed
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

if __name__ == '__main__':
    app.run_server(debug=True)
