import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc

app = dash.Dash()

app.layout = html.Div([
html.Button(id='button', n_clicks=0, children='Add graph'),
html.Div(id='container'),
html.Div(dcc.Graph(id='empty', figure={'data': []}), style={'display': 'none'}),
])


app.config['suppress_callback_exceptions']=True


for i in range(10):
    @app.callback(Output('graph_{}'.format(i), 'children'), [
    Input('dropdown_{}'.format(i), 'value'),
    ])
    def graph_update(something):
        return html.H1(something)


@app.callback(Output('container', 'children'), [Input('button', 'n_clicks')])
def display_graphs(n_clicks):
    graphs = []
    for i in range(n_clicks):
        graphs.append(html.Div(children=[
            html.H1("graph_{}".format(i)),
            dcc.Dropdown(
                value=['a'],
                options=[{'label': i, 'value': i} for i in ['a', 'b', 'c', 'd']],
                multi=True,
                id='dropdown_{}'.format(i)
            ),
            html.Div(id="graph_{}".format(i)),
        ]))
    return html.Div(graphs)


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')