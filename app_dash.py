from dash import Dash, dcc, Output, Input, html, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px 
import pandas as pd

app = Dash(__name__,external_stylesheets=[dbc.themes.COSMO])

#------------Data-----------
wlp_ground_truth_df = pd.read_csv('data/wlp.csv')
wlp_ground_truth_df = wlp_ground_truth_df.drop(['Unnamed: 0'],axis=1)

twas_df = pd.read_csv('data/twas.csv')
twas_df = twas_df.drop(['Unnamed: 0'],axis=1)


#------------Components-----------
fileDropdown = dcc.Dropdown(
   options=[
       {'label': 'WLP', 'value': 'wlp_ground_truth_df'},
       {'label': 'TWAS', 'value': 'twas_df'},
   ],
   value = 'wlp_ground_truth_df',
   id = 'inputFile-dropdown'
)
searchDropdown = dcc.Dropdown(id='search-dropdown')

data_table_1 =  dash_table.DataTable(data = None, columns = None, style_cell={'whiteSpace':'normal','height':'auto'})
data_table_2 =  dash_table.DataTable(data = None, columns = None,style_cell={'whiteSpace':'normal','height':'auto'})

searchInput = dcc.Input(id="search-input", type="text",placeholder=None)

grapgh_1 = dcc.Graph(figure= px.bar(x=None, y=None))
graph_2 = dcc.Graph(figure = px.bar(x=None, y=None))


#------------Layout----------------
app.layout = html.Div(children = [
     html.Div( #div for displaying error and warning headers
            dbc.Row([html.Div(
                        children = [
                            dbc.Col(html.Div([html.H3(children = ["Errors"],style={'textAlign':'center'})]), width=6,id='E_header'),
                            dbc.Col(html.Div([html.H3(children = ["Warnings"],style={'textAlign':'center'})]), width=6,id='W_header')
                        ], style={'display': 'flex'})
                    ])
    ),
    html.Div(
            dbc.Row([html.Div( #div for display error and warning counts
                        children = [
                            dbc.Col(html.Div([html.H1(None,style={'textAlign':'center'}, id='ErrorCount')]), width=6,id='E_count'),
                            dbc.Col(html.Div([html.H1(None,style={'textAlign':'center'}, id='WarningCount')]), width=6,id='W_count')
                        ], style={'display': 'flex'})
                    ])
    ),
    html.Div(children="Select File",style={"font-weight": "bold",'textAlign':'center'}),
    html.Div(children=fileDropdown,style={"width": "35%",'margin': 'auto'}),
    html.Div(children="Select Column",style={"font-weight": "bold"}),
    html.Div(children=[searchDropdown, searchInput],style={"display":'inline'}),
    html.Div(children=[None], id='temp-output'),
    html.Div(dbc.Row([
            dbc.Col(children = [data_table_1, html.Br(),data_table_2], style={'margin':'10px'})
        ])),
    grapgh_1, graph_2

])


# ------------Callbacks---------------
@app.callback(
    Output(grapgh_1, component_property="figure"),
    Output(graph_2, component_property="figure"),
    Input('inputFile-dropdown', component_property='value'), 
    Input(searchDropdown, component_property='value'),
    Input(searchInput, component_property='value')  
)
def update_graphs(fileName, colName, colValue): #update graphs based on search results
    if fileName == 'wlp_ground_truth_df':
        df = wlp_ground_truth_df
    elif fileName == 'twas_df':
        df = twas_df
    
    if colName!=None and colValue!=None and len(colValue)!=0:
        df = df[df[colName]==colValue]

    if df.empty == False: #if the data frame is not empty (i.e the filtering did not produce any results)
        if fileName == 'wlp_ground_truth_df':
            eventIDCounts = df.groupby(['EventID'])['EventID'].count()
            eventID_Fig = px.bar(x=eventIDCounts.index.tolist(), y=eventIDCounts.tolist(),labels={
                        'x': "EventID",
                        'y': "Number of Occurences"
                    },title="Distribution of Event Types")

            logIDCounts = df.groupby(['LogID'])['LogID'].count()
            logID_Fig = px.bar(x=logIDCounts.index.tolist(), y=logIDCounts.tolist(),labels={
                        'x': "LogID",
                        'y': "Number of Occurences"
                    },title="Distribution of Log Types")
            return eventID_Fig,logID_Fig
        else:
            classCounts = df.groupby(['class'])['class'].count()
            classFig = px.bar(x=classCounts.index.tolist(), y=classCounts.tolist(),labels={
                        'x': "Class",
                        'y': "Number of Occurences"
                    },title="Distribution of Class Types")

            sign = df.groupby(['sign'])['sign'].count()
            sign_fig = px.bar(x=sign.index.tolist(), y=sign.tolist(),labels={
                        'x': "Sign",
                        'y': "Number of Occurences"
                    },title="Distribution of Signs")
            return classFig,sign_fig
    else:
        return px.bar(x=None, y=None), px.bar(x=None, y=None)
    
        

@app.callback(
    Output(data_table_1, component_property='data'),
    Output(data_table_1, component_property='columns'),
    Output(data_table_2, component_property='data'),
    Output(data_table_2, component_property='columns'),
    Output('search-dropdown', component_property = 'options'),
    Output('ErrorCount', component_property = 'children'),
    Output('WarningCount', component_property = 'children'),
    Input('inputFile-dropdown', component_property='value'),
    Input(searchDropdown, component_property='value'),
    Input(searchInput, component_property='value')   
)
def update_data_view(fileName, colName, colValue): #updates view of the data based on selected file and filtering criteria
    if fileName == 'wlp_ground_truth_df':
        df = wlp_ground_truth_df
    elif fileName == 'twas_df':
        df = twas_df

    colOptions = []
    for c in df.columns.to_list():
        colOptions.append({'label':c,'value':c}) #update the options in search dropdown based on columns of data

    if colName!=None and colValue!=None and len(colValue)!=0:
        df = df[df[colName]==colValue] #filter the data based on search values

    #split dataframe in two
    df_1 = df.iloc[:,:-4] #everything until the last 4 rows
    df_2 = df.iloc[:,-4:] #last 4 rows

    data1 =  df_1.to_dict('records')[:5] #displays the first 5 rows only...can adjust
    columns1 = [{"name": i, "id": i} for i in df_1.columns]
    data2 =  df_2.to_dict('records')[:5]
    columns2 = [{"name": i, "id": i} for i in df_2.columns]

    if fileName == 'wlp_ground_truth_df':
        typeCount = df.groupby(['LogID'])['LogID'].count()
    else:
        typeCount = df.groupby(['class'])['class'].count()
    errorCount = 0
    warningCount = 0
    if 'E' in typeCount:
        errorCount = typeCount['E']
    if 'W' in typeCount:
        warningCount = typeCount['W']

    return data1, columns1, data2, columns2,colOptions,errorCount,warningCount


if __name__ == '__main__':
    app.run_server(debug=True)