# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:14:41 2019

@author: ke
"""



import dash
from dash.dependencies import Input, Output
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


app = dash.Dash(__name__)
server = app.server


df=root_pandas.read_root('model_tree.root',key='DecayTree')


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



newB=B.boost(-B.boostp3)
newDst=Dst.boost(-B.boostp3)
newtau=tau.boost(-B.boostp3)
newD0=D0.boost(-B.boostp3)
newnuB=nuB.boost(-B.boostp3)
newK=K.boost(-B.boostp3)
newpiDst=piDst.boost(-B.boostp3)
newpitau1=pitau1.boost(-B.boostp3)
newpitau2=pitau2.boost(-B.boostp3)
newpitau3=pitau3.boost(-B.boostp3)
newnutau=nutau.boost(-B.boostp3)
newpiK=piK.boost(-B.boostp3)


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

############################################################################

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
    

layout_phase=go.Layout(
    showlegend=False,
    width=800,
    height=900,
    autosize=False,
    margin=dict(t=0, b=0, l=0, r=0),
    scene=dict(
        xaxis=dict(
            title='$\\theta *$',
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
            title='$\\theta_l$',
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
            title='$\\chi$',
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
        
trace_hist=go.Histogram(x=costhetast)
layout_hist = go.Layout(title='costheta* distribution')



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
                    'font-size': '6.0rem',
                    'color': '#4D637F'
                }),
				html.P('Choose ranges :'),
                ]),
			html.Br(),
            
            """RangeSlider for choosing thetast"""            
		    html.P('theta_st :',
					style={
						'display':'inline-block',
						'verticalAlign': 'top',
						'marginRight': '10px'
					}
		    ),
		    html.Div([
					dcc.RangeSlider(
						id='choose-thetast',
						min=0, max=np.pi, value=[0,np.pi], step=0.1,
					),
			html.Div(id='output-range-thetast')
		    ], style={'width':300, 'display':'inline-block', 'marginBottom':10}),


		    """RangeSlider for choosing thetal"""
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

		    """RangeSlider for choosing chi"""
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
						min=0, max=np.pi,value=[0,np.pi], step=0.1,
					),
			html.Div(id='output-range-chi')
		    ], style={'width':300, 'display':'inline-block', 'marginBottom':10}),

		], style={'margin':20} ),

		html.P('Phase space of selected ranges',
				style = {'fontWeight':600}
		),

		dcc.Graph(
				    id = 'phase-space',
			figure = dict(
				data=[trace_phase],
				layout = layout_phase
				)
		),

		html.Div([
				html.P('Maybe I should put some text here ?'
				)
		], style={'margin':20})     #maybe I should change the style
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
                )
		html.Br(),
		html.P('Select frame:', style={'display': 'inline-block'}),
		
		dcc.Dropdown(
		    options=[{'label': 'Lab', 'value': 'lab'},
					{'label': 'COM of B', 'value': 'B'},
					{'label': 'COM of D*', 'value': 'D'},
			     		{'label': 'COM of tau', 'value': 'tau'},
					{'label': 'COM of (B-D*)', 'value': 'B-D'}],
			value='lab',
			id='dropdown-frame'
		),
		dcc.Graph(
			id = 'selected-frame',
			figure = dict(
				data = [trace],
				layout = layout    #?? How to change layout for different graphs
				
			),
		)
	], className='six columns', style={'margin':0}),

])
                
#--------------------------------
#                                |
#Callback for the range sliders  |
#                                |
#--------------------------------
 
#Show the chosen ranges
@app.callback(
    dash.dependencies.Output('output-range-thetast', 'children'),
    [dash.dependencies.Input('choose-thetast', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)
                
@app.callback(
    dash.dependencies.Output('output-range-thetal', 'children'),
    [dash.dependencies.Input('choose-thetal', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)

            
@app.callback(
    dash.dependencies.Output('output-range-chi', 'children'),
    [dash.dependencies.Input('choose-chi', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)
                
#Show the corresponding graph
@app.callback(
	dash.dependencies.Output('phase-space', 'figure'),
	[Input('years-slider', 'value'),
		Input('opacity-slider', 'value'),
		Input('colorscale-picker', 'colorscale'),
		Input('hide-map-legend', 'values')],

	
	

                
#--------------------------------
#Callback for the RadioItems     |
#    and the DropdownMenu        |
#--------------------------------
 


#--------------------------------
#Callback for the RadioItems     |
#    and the DropdownMenu        |
#--------------------------------
 
	
	
	
	

  
if __name__ == '__main__':
    app.run_server(debug=True)





