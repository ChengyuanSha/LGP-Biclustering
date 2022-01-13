import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import plotly.express as px
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from _processing_funcs import ResultProcessing
import dash_cytoscape as cyto

def convert_program_str_repr(model, names):
    # convert the raw string to user friendly string
    s = ''
    original_str = model.bestEffProgStr_.splitlines()
    i = 0
    indentation = False
    indentation_level = 1
    connecting_list = []
    s = s + "The default value of r0: " + str(round(model.register_[model.numberOfInput], 2)) + "\nModels:\n"
    while i < len(original_str):
        current_string = original_str[i]
        vars_in_line = re.findall(r'r\d+', current_string)
        # reformat structure
        if 'if' not in current_string:
            extract = [x.strip() for x in current_string.split(',')]
            current_string = extract[0][:3] + ' '+ extract[1] + ' = ' +  extract[2] + ' ' + extract[0][-1] + ' ' + extract[3][:-1]
        # substitute variable index
        for var in vars_in_line:
            var = var[1:]
            if var not in connecting_list: # not a connecting calculation variable
                if int(var) < model.numberOfVariable and int(var) != 0: # var is calculation variable
                    if (i+1 < len(original_str)) and var in ( [i[1:] for i in re.findall(r'r\d+', original_str[i+1])] ):
                        # a calculation variable connecting the two lines in a program
                        connecting_list.append(var)
                    else: # calculation variable is a constant
                        current_string = re.sub('r' + re.escape(var), str(round(model.register_[int(var)], 2)),
                                            current_string)
                elif int(var) >= model.numberOfVariable and int(var) != 0: # features
                    name_index = int(var) - model.numberOfVariable
                    current_string = re.sub('r' + re.escape(var), str(names[name_index]), current_string)
        # take care of indentation
        if indentation:
            current_string = current_string[:3] + indentation_level*'  ' + 'then ' + current_string[3:]
        if 'if' in current_string:
            indentation = True
            indentation_level += 1
        else:
            indentation = False
        s += current_string + '\n'
        i += 1
    s += 'Output register r[0] will then go through sigmoid transformation S \nif S(r[0]) is less or equal ' \
         'than 0.5:\n  this sample will be classified by this model as class 0, i.e. diseased. \nelse:\n' \
         '  class 1, i.e. healthy'
    return s




# draw Number of effective instructions VS num_of_eff_features graph
def eff_vs_numOfEffFeature_graph(program_length_list, num_of_eff_features, accuracy):
    d = {'Number of effective instructions': program_length_list, 'Number of effective feature': num_of_eff_features,
         'accuracy': accuracy}
    df_h = pd.DataFrame(d)
    temp = df_h.groupby(['Number of effective instructions', 'Number of effective feature']).size().reset_index(name='count')
    df_h = df_h.merge(temp,  how='left')
    # draw complete
    plt.scatter(x=df_h["Number of effective instructions"], y=df_h["Number of effective feature"], c=df_h['accuracy'].apply(lambda x: x*100),
                s=df_h['count'], cmap='Reds')
    plt.ylim((0, 10))
    plt.xlim((0, 10))
    plt.xlabel("Number of effective instructions", fontsize=12)
    plt.ylabel("Number of effective features", fontsize=12)
    plt.colorbar()
    plt.savefig('program_lengh_vs_feature.jpeg', dpi=300, format='JPEG')



def create_network(result_data, top_percentage, names, edge_threshold=None, num_of_models=None):
    top_percentage = top_percentage * 0.01
    df, node_size_dic = result_data.get_network_data(names, top_percentage, num_of_models, edge_threshold)
    # error catching, when no data available
    if df.empty:
        return html.Div(
            html.Div(
                dcc.Markdown(
                    '''
                    ##### No network graph in given selection, try to decrease testing accuracy filter.
                    '''),
                className='pretty_container eleven columns',
            ),
            className='container-display',
        )
    nodes = [
        {
            'data': {'id': node, 'label': node, 'size': node_size_dic[node]},
            'position': {'x': np.random.randint(0, 100), 'y': np.random.randint(0, 100)},
        }
        for node in np.unique(df[['f1', 'f2']].values)
    ]
    edges = [
        {'data': {'source': df['f1'][index], 'target': df['f2'][index], 'weight': df['weight'][index]}}
        for index, row in df.iterrows()
    ]
    elements = nodes + edges
    return html.Div(
        html.Div([
            cyto.Cytoscape(
                id='cytoscape-layout-1',
                elements=elements,
                responsive=True,
                style={'width': '100%', 'height': '700px'},
                layout={
                    'name': 'cola',
                    'nodeRepulsion': 40000,
                    'nodeSpacing': 35,
                },
                zoomingEnabled=False,
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {
                            "width": "mapData(size, 0, 100, 20, 60)",
                            "height": "mapData(size, 0, 100, 20, 60)",
                            "content": "data(label)",
                            "font-size": "12px",
                            "text-valign": "center",
                            "text-halign": "center",
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            "opacity": "0.5",
                            "width": "mapData(weight, 0, 20, 1, 8)",
                            "overlay-padding": "3px",
                            "content": "data(weight)",
                            "font-size": "10px",
                            "text-valign": "center",
                            "text-halign": "center",
                        }
                    },
                ],
            )  # end cytoscape
        ],
            className='pretty_container eleven columns',
        ),
        className='container-display',
    )



def create_diff_coe_network(result_data, top_percentage, names, diff_coe, corr_df_control_matrix, corr_df_case_matrix):
    top_percentage = top_percentage * 0.01
    df, node_size_dic = result_data.get_network_data(names, top_percentage)
    # error catching, when no data available
    if df.empty:
        return html.Div(
            html.Div(
                dcc.Markdown(
                    '''
                    ##### No network graph in given selection, try to decrease testing accuracy filter.
                    '''),
                className='pretty_container eleven columns',
            ),
            className='container-display',
        )
    nodes = [
        {
            'data': {'id': node, 'label': node, 'size': node_size_dic[node]},
            'position': {'x': np.random.randint(0, 100), 'y': np.random.randint(0, 100)},
        }
        for node in np.unique(df[['f1', 'f2']].values)
    ]
    edges = [
            {
            'data': {
                  'source': df['f1'][index],
                  'target': df['f2'][index],
                  'weight': 'ctrl:' + str(corr_df_control_matrix[df['f1'][index]] [df['f2'][index]]) + " case:" + str(corr_df_case_matrix [df['f1'][index]] [df['f2'][index]]) + ' diff:' + str(diff_coe [df['f1'][index]] [df['f2'][index]])
                     },
                  'classes': 'red' if (diff_coe [df['f1'][index]] [df['f2'][index]] > 0) else 'blue'
            }
            for index, row in df.iterrows()
    ]
    elements = nodes + edges
    return html.Div(
        html.Div([
            cyto.Cytoscape(
                id='cytoscape-layout-1',
                elements=elements,
                responsive=True,
                style={'width': '100%', 'height': '700px'},
                layout={
                    'name': 'cola',
                    'nodeRepulsion': 40000,
                    'nodeSpacing': 35,
                },
                zoomingEnabled=False,
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {
                            "width": "mapData(size, 0, 100, 20, 60)",
                            "height": "mapData(size, 0, 100, 20, 60)",
                            "content": "data(label)",
                            "font-size": "12px",
                            "text-valign": "center",
                            "text-halign": "center",
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            "opacity": "0.5",
                            #"width": "mapData(weight, 0, 20, 1, 8)",
                            "overlay-padding": "3px",
                            "content": "data(weight)",
                            "font-size": "10px",
                            "text-valign": "center",
                            "text-halign": "center",
                        }
                    },
                    # Class selectors
                    {
                        'selector': '.red',
                        'style': {
                            'background-color': 'red',
                            'line-color': 'red'
                        }
                    },
                    {
                        'selector': '.blue',
                        'style': {
                            'background-color': 'blue',
                            'line-color': 'blue'
                        }
                    },
                ],
            )  # end cytoscape
        ],
            className='pretty_container eleven columns',
        ),
        className='container-display',
    )