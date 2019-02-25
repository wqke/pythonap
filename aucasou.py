import dash
from dash.dependencies import Input, Output,State
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import os
import root_pandas
import pandas as pd
import hepvector
from hepvector.numpyvector import Vector3D,LorentzVector

from matplotlib import pyplot as plt
import numpy as np
from numpy import cos,sin,tan,sqrt,absolute,real,conjugate,imag,abs,max,min

import hepvector
from hepvector.numpyvector import Vector3D,LorentzVector

import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import Data, Layout, Figure
from plotly.graph_objs import Scatter




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server



dft=root_pandas.read_root('model_tree.root',key='DecayTree')

df=dft.head(100)


B=LorentzVector(df['B_PX_TRUE'],df['B_PY_TRUE'],df['B_PZ_TRUE'],df['B_E_TRUE'])


Dst=LorentzVector(df['Dst_PX_TRUE'],df['Dst_PY_TRUE'],df['Dst_PZ_TRUE'],df['Dst_E_TRUE'])
tau=LorentzVector(df['Tau_PX_TRUE'],df['Tau_PY_TRUE'],df['Tau_PZ_TRUE'],df['Tau_E_TRUE'])
D0=LorentzVector(df['D0_PX_TRUE'],df['D0_PY_TRUE'],df['D0_PZ_TRUE'],df['D0_E_TRUE'])
nuB=LorentzVector(df['B_nu_PX_TRUE'],df['B_nu_PY_TRUE'],df['B_nu_PZ_TRUE'],df['B_nu_E_TRUE'])
K=LorentzVector(df['D0_K_PX_TRUE'],df['D0_K_PY_TRUE'],df['D0_K_PZ_TRUE'],df['D0_K_E_TRUE'])
piDst=LorentzVector(df['Dst_Pi_PX_TRUE'],df['Dst_Pi_PY_TRUE'],df['Dst_Pi_PZ_TRUE'],df['Dst_Pi_E_TRUE'])
piK=LorentzVector(df['D0_Pi_PX_TRUE'],df['D0_Pi_PY_TRUE'],df['D0_Pi_PZ_TRUE'],df['D0_Pi_E_TRUE'])
pitau1=LorentzVector(df['Tau_Pi1_PX_TRUE'],df['Tau_Pi1_PY_TRUE'],df['Tau_Pi1_PZ_TRUE'],df['Tau_Pi1_E_TRUE'])
pitau2=LorentzVector(df['Tau_Pi2_PX_TRUE'],df['Tau_Pi2_PY_TRUE'],df['Tau_Pi2_PZ_TRUE'],df['Tau_Pi2_E_TRUE'])
pitau3=LorentzVector(df['Tau_Pi3_PX_TRUE'],df['Tau_Pi3_PY_TRUE'],df['Tau_Pi3_PZ_TRUE'],df['Tau_Pi3_E_TRUE'])
nutau=LorentzVector(df['Tau_nu_PX_TRUE'],df['Tau_nu_PY_TRUE'],df['Tau_nu_PZ_TRUE'],df['Tau_nu_E_TRUE'])




def change_frame(COM):
        newB=B.boost(-COM.boostp3)
        newDst=Dst.boost(-COM.boostp3)
        newtau=tau.boost(-COM.boostp3)
        newD0=D0.boost(-COM.boostp3)
        newnuB=nuB.boost(-COM.boostp3)
        newK=K.boost(-COM.boostp3)
        newpiDst=piDst.boost(-COM.boostp3)
        newpitau1=pitau1.boost(-COM.boostp3)
        newpitau2=pitau2.boost(-COM.boostp3)
        newpitau3=pitau3.boost(-COM.boostp3)
        newnutau=nutau.boost(-COM.boostp3)
        newpiK=piK.boost(-COM.boostp3)
        res=[newB,newDst,newtau,newD0,newnuB,newK,newpiDst,newpitau1,newpitau2,newpitau3,newnutau,newpiK]
        return res




nouvtau=tau.boost(-(tau+nuB).boostp3)
nouvnu=nuB.boost(-(tau+nuB).boostp3)
nouvpi=piDst.boost(-(piDst+D0).boostp3)
nouvD0=D0.boost(-(piDst+D0).boostp3)
nouvDst=D0.boost(-B.boostp3)


unittau=(nouvtau.p3).unit
unitnu=(nouvnu.p3).unit
unitDst=(nouvDst.p3).unit
unitD0=(nouvD0.p3).unit

normal1=unittau.cross(unitDst)
normal2=unitDst.cross(unitD0)
coski=normal1.dot(normal2)
costhetast=unitD0.dot(unitDst)
costhetal=unitDst.dot(unittau)

chi=np.arccos(coski)
thetast=np.arccos(costhetast)
thetal=np.arccos(costhetal)




trace_phase=go.Scatter3d(
            x=thetast,
            y=thetal,
            z=chi,
            mode='markers',
            marker=dict(
                size=5,
                color=chi,
                colorscale='Viridis',
                opacity=0.8
            )
)

app.layout = html.Div(children=[
        html.Div([
                html.Div([
                        html.Div([
                                html.H2('B decay visualisation',
                                style={
                                'position': 'relative',
                                'top': '0px',
                                'left': '10px',
                                'font-family': 'Dosis',
                                'display': 'inline',
                                'font-size':'5.0rem',
                                'color': '#4D637F'
                                 }),
                                html.Br(),
                                html.P('Choose ranges :',
                                style={
                                'font-family': 'Dosis',
                                'display': 'inline',
                                'font-size': '3rem',
                                'color': '#4D637F'
                                 }),
                         ]),
                        html.Br(),
                        html.P('theta* :',
                                style={
                                        'display':'inline-block',
                                        'verticalAlign': 'top',
                                        'marginRight': '10px'
                                }
                         ),
                        html.Div([
                                dcc.RangeSlider(
                                        id='choose-thetast',
                                        min=0,
                                        max=np.pi,
                                        value=[0,np.pi], step=0.1
                                ),
                                html.Div(id='output-range-thetast')
                         ],style={'width':300,'display':'inline-block','marginBottom':10}),
                        html.Br(),
                        html.P('theta_l :',
                                        style={
                                                'display':'inline-block',
                                                'verticalAlign': 'top',
                                                'marginRight': '10px'
                                        }
                        ),
                        html.Div([
                                dcc.RangeSlider(
                                        id='choose-thetal',
                                        min=0, max=np.pi, value=[0,np.pi], step=0.1,
                                ),
                                html.Div(id='output-range-thetal')
                        ], style={'width':300, 'display':'inline-block', 'marginBottom':10}),

                         html.Br(),
                         html.P('chi :',
                                        style={
                                                'display':'inline-block',
                                                'verticalAlign': 'top',
                                                'marginRight': '10px'
                                        }
                         ),
                         html.Div([
                                dcc.RangeSlider(
                                        id='choose-chi',
                                        min=0, max=np.pi, value=[0,np.pi], step=0.1,
                                ),
                                html.Div(id='output-range-chi')
                        ], style={'width':300, 'display':'inline-block', 'marginBottom':10}),

                ], style={'margin':20} ),

                html.P('Phase space of selected ranges',
                        style={
                    'font-family': 'Dosis',
                    'display': 'inline',
                    'font-size': '3.0rem',
                    'color': '#4D637F'
                }
                ),

                dcc.Graph(id = 'phase-space')#,clickData={'pointNumber':[]})
        ], className='six columns', style={'margin':0}),

        html.Div([
                dcc.RadioItems(
                    options=[
                        {'label':'3D one event','value':'3D'},
                        {'label':'XY projection','value':'XY'},
                        {'label':'YZ projection','value':'YZ'},
                        {'label':'ZX projection','value':'ZX'}
                    ],
                    value='3D',
                    id='which-D'
                ),
                html.Br(),
                html.P('Select frame:',
                 style={
                    'font-family': 'Dosis',
                    'display': 'inline',
                    'font-size': '3rem',
                    'color': '#4D637F'
                }),

                dcc.Dropdown(
                    options=[{'label': 'Lab', 'value': 'lab'},
                                        {'label': 'COM of B', 'value': 'B'},
                                        {'label': 'COM of D*', 'value': 'D'},
                                        {'label': 'COM of tau', 'value': 'tau'},
                                        {'label': 'COM of (B-D*)', 'value': 'B-D'}],
                        value='lab',
                        id='dropdown-frame'
                ),
                dcc.Graph(id='frame-graph')
               
                ], className='six columns', style={'margin':0}),

])

#--------------------------------
#                                |
#Callback for the range sliders  |
#                                |
#--------------------------------

@app.callback(
    dash.dependencies.Output('output-range-thetast', 'children'),
    [dash.dependencies.Input('choose-thetast', 'value')])

def update_output1(value):
        return 'You have selected "{}"'.format(value)



@app.callback(
    dash.dependencies.Output('output-range-thetal', 'children'),
    [dash.dependencies.Input('choose-thetal', 'value')])
def update_output2(value):
        return 'You have selected "{}"'.format(value)


@app.callback(
    dash.dependencies.Output('output-range-chi', 'children'),
    [dash.dependencies.Input('choose-chi', 'value')])
def update_output3(value):
        return 'You have selected "{}"'.format(value)

@app.callback(
        dash.dependencies.Output('phase-space', 'figure'),
        [Input('choose-thetast', 'value'),
        Input('choose-thetal', 'value'),
        Input('choose-chi', 'value')])
def plot_phase_space(rangest,rangel,rangechi):
        return {'data': [trace_phase], 'layout': go.Layout(hovermode = 'closest',
                                        clickmode='event+select',
                                        scene={'xaxis':{'title':'thetast','range':rangest},'yaxis':{'title':'thetal','range':rangel},'zaxis':{'title':'chi','range':rangechi}},
                                        )

        }

#------------------------
#                        |
#Callback for clickData  |
#                        |
#------------------------

@app.callback(
Output('frame-graph', 'figure'),
[Input('phase-space', 'clickData')])
def drawevent(selection):
        if selection is None:
                return {}
        else:
                i=selection['points'][0]['pointNumber']
                PV_X,PV_Y,PV_Z=(df['B_Ori_z_TRUE'][i],df['B_Ori_x_TRUE'][i],df['B_Ori_y_TRUE'][i])
                B_X,B_Y,B_Z=(df['B_End_z_TRUE'][i],df['B_End_x_TRUE'][i],df['B_End_y_TRUE'][i])


                traceB=go.Scatter3d(
                        x=[PV_X,B_X],
                        y=[PV_Y,B_Y],
                        z=[PV_Z,B_Z],
                        mode='lines+markers+text',
                        marker=dict(
                                size=5,
                                color= "rgb(5,200,5)",
                                opacity=0.8),
                        text=['PV', ''],
                        textposition='top left',
                        line = dict(
                                color = ('rgb(0, 0, 255)'),
                                width = 3)
                        )

                return {'data': [traceB], 'layout': go.Layout(hovermode = 'closest',
                                        scene={'xaxis':{'title':''},'yaxis':{'title':'thetal'},'zaxis':{'title':'chi'}},
                                        )

                                }






if __name__ == '__main__':
    app.run_server(debug=True)

                                                                                                
