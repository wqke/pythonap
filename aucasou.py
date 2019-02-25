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
                Dst_X,Dst_Y,Dst_Z=(df['Dst_End_z_TRUE'][i],df['Dst_End_x_TRUE'][i],df['Dst_End_y_TRUE'][i])
                tau_X,tau_Y,tau_Z=[df['Tau_End_z_TRUE'][i],df['Tau_End_x_TRUE'][i],df['Tau_End_y_TRUE'][i]]
                D0_X,D0_Y,D0_Z=[df['D0_End_z_TRUE'][i],df['D0_End_x_TRUE'][i],df['D0_End_y_TRUE'][i]]
                nu_X=df['B_nu_PZ_TRUE'][i]*10000/df['B_nu_P_TRUE'][i]+B_X
                nu_Y=df['B_nu_PX_TRUE'][i]*10000/df['B_nu_P_TRUE'][i]+B_Y
                nu_Z=df['B_nu_PY_TRUE'][i]*10000/df['B_nu_P_TRUE'][i]+B_Z
                K_X=df['D0_K_PZ_TRUE'][i]*10000/df['D0_K_P_TRUE'][i]+D0_X
                K_Y=df['D0_K_PX_TRUE'][i]*10000/df['D0_K_P_TRUE'][i]+D0_Y
                K_Z=df['D0_K_PY_TRUE'][i]*10000/df['D0_K_P_TRUE'][i]+D0_Z
                piK_X=df['D0_Pi_PZ_TRUE'][i]*10000/df['D0_Pi_P_TRUE'][i]+D0_X
                piK_Y=df['D0_Pi_PX_TRUE'][i]*10000/df['D0_Pi_P_TRUE'][i]+D0_Y
                piK_Z=df['D0_Pi_PY_TRUE'][i]*10000/df['D0_Pi_P_TRUE'][i]+D0_Z
                piDst_X=df['Dst_Pi_PZ_TRUE'][i]*10000/df['Dst_Pi_P_TRUE'][i]+Dst_X
                piDst_Y=df['Dst_Pi_PX_TRUE'][i]*10000/df['Dst_Pi_P_TRUE'][i]+Dst_Y
                piDst_Z=df['Dst_Pi_PY_TRUE'][i]*10000/df['Dst_Pi_P_TRUE'][i]+Dst_Z

                pitau1_X=df['Tau_Pi1_PZ_TRUE'][i]*10000/df['Tau_Pi1_P_TRUE'][i]+tau_X
                pitau1_Y=df['Tau_Pi1_PX_TRUE'][i]*10000/df['Tau_Pi1_P_TRUE'][i]+tau_Y
                pitau1_Z=df['Tau_Pi1_PY_TRUE'][i]*10000/df['Tau_Pi1_P_TRUE'][i]+tau_Z

                pitau2_X=df['Tau_Pi2_PZ_TRUE'][i]*10000/df['Tau_Pi2_P_TRUE'][i]+tau_X
                pitau2_Y=df['Tau_Pi2_PX_TRUE'][i]*10000/df['Tau_Pi2_P_TRUE'][i]+tau_Y
                pitau2_Z=df['Tau_Pi2_PY_TRUE'][i]*10000/df['Tau_Pi2_P_TRUE'][i]+tau_Z
                pitau3_X=df['Tau_Pi3_PZ_TRUE'][i]*10000/df['Tau_Pi3_P_TRUE'][i]+tau_X
                pitau3_Y=df['Tau_Pi3_PX_TRUE'][i]*10000/df['Tau_Pi3_P_TRUE'][i]+tau_Y
                pitau3_Z=df['Tau_Pi3_PY_TRUE'][i]*10000/df['Tau_Pi3_P_TRUE'][i]+tau_Z
                nutau_X=df['Tau_nu_PZ_TRUE'][i]*10000/df['Tau_nu_P_TRUE'][i]+tau_X
                nutau_Y=df['Tau_nu_PX_TRUE'][i]*10000/df['Tau_nu_P_TRUE'][i]+tau_Y
                nutau_Z=df['Tau_nu_PY_TRUE'][i]*10000/df['Tau_nu_P_TRUE'][i]+tau_Z

                [traceB,traceDst,tracetau,traceD0,tracenuB,traceK]=[go.Scatter3d(x=[PV_X,B_X],y=[PV_Y,B_Y],z=[PV_Z,B_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['PV', ''],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3)),go.Scatter3d(x=[B_X,Dst_X],y=[B_Y,Dst_Y],z=[B_Z,Dst_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)",opacity=0.8),text=['', 'D*'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3)),go.Scatter3d(x=[B_X,tau_X],y=[B_Y,tau_Y],z=[B_Z,tau_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'tau'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3)),go.Scatter3d(x=[Dst_X,D0_X],y=[Dst_Y,D0_Y],z=[Dst_Z,D0_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'D0'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3)),go.Scatter3d(x=[B_X,nu_X],y=[B_Y,nu_Y],z=[B_Z,nu_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)",opacity=0.8),text=['', 'nuB'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3)),go.Scatter3d(x=[D0_X,K_X],y=[D0_Y,K_Y],z=[D0_Z,K_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'K'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))]

        
                tracepiK=go.Scatter3d(x=[D0_X,piK_X],y=[D0_Y,piK_Y],z=[D0_Z,piK_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))

                tracepiDst=go.Scatter3d(x=[Dst_X,piDst_X],y=[Dst_Y,piDst_Y],z=[Dst_Z,piDst_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))

                tracepitau1=go.Scatter3d(x=[tau_X,pitau1_X],y=[tau_Y,pitau1_Y],z=[tau_Z,pitau1_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))

                tracepitau2=go.Scatter3d(x=[tau_X,pitau2_X],y=[tau_Y,pitau2_Y],z=[tau_Z,pitau2_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))


                tracepitau3=go.Scatter3d(x=[tau_X,pitau3_X],y=[tau_Y,pitau3_Y],z=[tau_Z,pitau3_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'pi'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))


                tracenutau=go.Scatter3d(x=[tau_X,nutau_X],y=[tau_Y,nutau_Y],z=[tau_Z,nutau_Z],mode='lines+markers+text',marker=dict(size=5,color= "rgb(5,200,5)", opacity=0.8),text=['', 'nu'],textposition='top left',line = dict(color = ('rgb(0, 0, 255)'),width = 3))



                xrange=max(abs([PV_X,B_X,Dst_X,D0_X,tau_X,K_X,nutau_X,nu_X,pitau1_X,
                                pitau2_X,pitau3_X,piDst_X,piK_X]))
                yrange=max(abs([PV_Y,B_Y,Dst_Y,D0_Y,tau_Y,K_Y,nutau_Y,nu_Y,pitau1_Y,
                         pitau2_Y,pitau3_Y,piDst_Y,piK_Y]))

                zrange=max(abs([PV_Z,B_Z,Dst_Z,D0_Z,tau_Z,K_Z,nutau_Z,nu_Z,pitau1_Z,
                                pitau2_Z,pitau3_Z,piDst_Z,piK_Z]))



                layout_event = go.Layout(
                showlegend=False,

                width=400,
                height=450,
                autosize=False,
                margin=dict(t=0, b=0, l=0, r=0),
                scene=dict(
                        xaxis=dict(
                        range=[-abs(xrange), abs(xrange)],
                        title='Z direction [mm]',
                        titlefont=dict(
                        family='Arial, sans-serif',
                        size=18,
                        color='black'
                        ),
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=4,

                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                        ),
                        yaxis=dict(
                        range=[-abs(yrange), abs(yrange)],
                        title='X direction [mm]',
                        titlefont=dict(
                        family='Arial, sans-serif',
                        size=18,
                        color='black'
                        ),
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=4,

                        showbackground=True,
                                                                                                 
                        backgroundcolor='rgb(230, 230, 230)'
                        ),
                        zaxis=dict(
                        range=[-abs(zrange), abs(zrange)],
                        title='Y direction [mm]',
                        titlefont=dict(
                        family='Arial, sans-serif',
                        size=18,
                        color='black'
                        ),
                        gridcolor='#bdbdbd',
                        gridwidth=2,
                        zerolinecolor='#969696',
                        zerolinewidth=4,
                        linecolor='#636363',
                        linewidth=4,

                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                        ),
                        aspectratio = dict(x=1, y=1, z=0.7),
                        aspectmode = 'manual'
                        )
                )








                return {'data': [traceB,traceDst,tracetau,traceD0,tracenuB,traceK], 'layout':layout_event

                                }








if __name__ == '__main__':
    app.run_server(debug=True)

                                                                                                
