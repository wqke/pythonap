# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:14:41 2019

@author: ke
"""


import json
from textwrap import dedent as d
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
nouvtau=tau.boost(-(tau+nuB).boostp3)
nouvnu=nuB.boost(-(tau+nuB).boostp3)
nouvpi=piDst.boost(-(piDst+D0).boostp3)
nouvD0=D0.boost(-(piDst+D0).boostp3)
nouvDst=D0.boost(-B.boostp3)
unittau=(nouvtau.p3).unit
unitnu=(nouvnu.p3).unit
unitDst=(nouvDst.p3).unit
unitD0=(nouvD0.p3).unit
nnewtau=tau.boost(-B.boostp3)
nnewD0=D0.boost(-B.boostp3)
unitau=nnewtau.unit
uniD0=nnewD0.unit
nnormal1=unitDst.cross(uniD0)
normal1=nnormal1.unit
nnormal2=unitDst.cross(unitau)
normal2=nnormal2.unit
pparallel=normal1.cross(unitDst)
parallel=pparallel.unit

co = normal1.dot(normal2)
si = parallel.dot(normal2)




chi = np.arctan2(si,co)



costhetast=unitD0.dot(unitDst)
costhetal=unitDst.dot(unittau)
############################################################################
def boost_frame(vect):
    newB=B.boost(-vect.boostp3)
    newDst=Dst.boost(-vect.boostp3)
    newtau=tau.boost(-vect.boostp3)
    newD0=D0.boost(-vect.boostp3)
    newnuB=nuB.boost(-vect.boostp3)
    newK=K.boost(-vect.boostp3)
    newpiDst=piDst.boost(-vect.boostp3)
    newpitau1=pitau1.boost(-vect.boostp3)
    newpitau2=pitau2.boost(-vect.boostp3)
    newpitau3=pitau3.boost(-vect.boostp3)
    newnutau=nutau.boost(-vect.boostp3)
    newpiK=piK.boost(-vect.boostp3)
    liste=[newB,newDst,newtau,newD0,newnuB,newK,newpiDst,newpitau1,newpitau2,newpitau3,newnutau,newpiK]
    return liste

############################################################################

#The histograms

trace_hist=go.Histogram(x=costhetast)
layout_hist = go.Layout(title='costheta* distribution')


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
	




###################
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

		dcc.Graph(id = 'phase-space',animate=True),

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
		dcc.Graph(id = 'selected-frame'),
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
def update_output1(value):
    return 'You have selected "{}"'.format(value)
                
@app.callback(
    dash.dependencies.Output('output-range-thetal', 'children'),
    [dash.dependencies.Input('choose-thetal', 'value')])
def update_output2(value):
    return 'You have selected "{}"'.format(value)

            
@app.callback(
    Output('output-range-chi', 'children'),
    [Input('choose-chi', 'value')])
def update_output3(value):
    return 'You have selected "{}"'.format(value)
                
	
#Show the corresponding graph
@app.callback(
	dash.dependencies.Output('phase-space', 'figure'),
	[Input('choose-thetast', 'value'),
		Input('choose-thetal', 'value'),
		Input('choose-chi', 'value')])
def plot_phase_space(rangest,rangel,rangechi):
	layout_phase=go.Layout(
	    showlegend=False,
	    width=800,
	    height=900,
	    autosize=False,
	    margin=dict(t=0, b=0, l=0, r=0),
	    scene=dict(
		xaxis=dict(
		    range=rangest,
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
		    range=ranglel,
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
	            range=rangechi,
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
	return {'data': [trace_phase], 'layout': layout_phase}




#---------------------------
#                           |
#Callback for clickData     |
#                           |
#---------------------------
	
@app.callback(
        Output('basic_graph', 'figure'),
        [Input('phase-space', 'clickData')])
def plot_basin(selection):
    if selection is None:
        return {}
    else:
                
#--------------------------------
#Callback for the RadioItems     |
#    and the DropdownMenu        |
#	and the click event	 |
#--------------------------------
 
@app.callback(
    Output('selected-frame', 'figure'),
    [Input('phase-space', 'clickData')]#,    #click event
   #  Input('dropdown-frame', 'value'),
   # Input('which-D', 'value')])
def draw_frame(selection):#,dropvalue,dimvalue):
	if selection is None:
        	return {}
	else:			#comment voir la sortie de clickData
		PV_X,PV_Y,PV_Z=[df['B_Ori_z_TRUE'][0],df['B_Ori_x_TRUE'][0],df['B_Ori_y_TRUE'][0]]
		B_X,B_Y,B_Z=[df['B_End_z_TRUE'][0],df['B_End_x_TRUE'][0],df['B_End_y_TRUE'][0]]
		traceB = go.Scatter3d(x=[PV_X,B_X],y=[PV_Y,B_Y],z=[PV_Z,B_Z],
		     mode='lines+markers+text',
		     marker = dict( size = 5,color = "rgb(5,200,5)"),
		     text=['PV', ''],
		     textposition='top left',
		     line = dict(
		       color = ('rgb(0, 0, 255)'),
			width = 3
		    ))
		Dst_X,Dst_Y,Dst_Z=[dh['Dst_End_z_TRUE'][0],dh['Dst_End_x_TRUE'][0],dh['Dst_End_y_TRUE'][0]]
		traceDst = go.Scatter3d(x=[B_X,Dst_X],y=[B_Y,Dst_Y],z=[B_Z,Dst_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'D*'],
		      textposition='bottom right',
		      line = dict(
		       color = ('rgb(0, 0,123)'),
			width = 3
		    ))
		tau_X,tau_Y,tau_Z=[dh['Tau_End_z_TRUE'][0],dh['Tau_End_x_TRUE'][0],dh['Tau_End_y_TRUE'][0]]
		tracetau = go.Scatter3d(x=[B_X,tau_X],y=[B_Y,tau_Y],z=[B_Z,tau_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'tau'],
		      textposition='top center',
		      line = dict(
			color = ('rgb(0, 0, 74)'),
			width = 3
		    ))
		D0_X,D0_Y,D0_Z=[dh['D0_End_z_TRUE'][0],dh['D0_End_x_TRUE'][0],dh['D0_End_y_TRUE'][0]]
		traceD0 = go.Scatter3d(x=[Dst_X,D0_X],y=[Dst_Y,D0_Y],z=[Dst_Z,D0_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'D0'],
		      textposition='top center',
		      line = dict(
			color = ('rgb(0, 0, 123)'),
			width = 3
		    ))
		nu_X=dh['B_nu_PZ_TRUE'][0]*10000/dh['B_nu_P_TRUE'][0]+B_X
		nu_Y=dh['B_nu_PX_TRUE'][0]*10000/dh['B_nu_P_TRUE'][0]+B_Y
		nu_Z=dh['B_nu_PY_TRUE'][0]*10000/dh['B_nu_P_TRUE'][0]+B_Z
		tracenuB=go.Scatter3d(x=[B_X,nu_X],y=[B_Y,nu_Y],z=[B_Z,nu_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'nu'],
		      textposition='top center',
		      line = dict(
			color = ('rgb(0, 0, 123)'),
			width = 3,
			dash='dash'
		    ))
		K_X=dh['D0_K_PZ_TRUE'][0]*10000/dh['D0_K_P_TRUE'][0]+D0_X
		K_Y=dh['D0_K_PX_TRUE'][0]*10000/dh['D0_K_P_TRUE'][0]+D0_Y
		K_Z=dh['D0_K_PY_TRUE'][0]*10000/dh['D0_K_P_TRUE'][0]+D0_Z
		traceK=go.Scatter3d(x=[D0_X,K_X],y=[D0_Y,K_Y],z=[D0_Z,K_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'K'],
		      textposition='top center',
		      line = dict(
			color = ('rgb(0, 0,11)'),
			width = 3,
		    ))
		piK_X=dh['D0_Pi_PZ_TRUE'][0]*10000/dh['D0_Pi_P_TRUE'][0]+D0_X
		piK_Y=dh['D0_Pi_PX_TRUE'][0]*10000/dh['D0_Pi_P_TRUE'][0]+D0_Y
		piK_Z=dh['D0_Pi_PY_TRUE'][0]*10000/dh['D0_Pi_P_TRUE'][0]+D0_Z
		tracepiK=go.Scatter3d(x=[D0_X,piK_X],y=[D0_Y,piK_Y],z=[D0_Z,piK_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'pi'],
		      textposition='top center',
		      line = dict(
			color = ('rgb(0, 0, 123)'),
			width = 3,
		    ))
		piDst_X=dh['Dst_Pi_PZ_TRUE'][0]*10000/dh['Dst_Pi_P_TRUE'][0]+Dst_X
		piDst_Y=dh['Dst_Pi_PX_TRUE'][0]*10000/dh['Dst_Pi_P_TRUE'][0]+Dst_Y
		piDst_Z=dh['Dst_Pi_PY_TRUE'][0]*10000/dh['Dst_Pi_P_TRUE'][0]+Dst_Z
		tracepiDst=go.Scatter3d(x=[Dst_X,piDst_X],y=[Dst_Y,piDst_Y],z=[Dst_Z,piDst_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'pi'],
		      textposition='top center',
		      line = dict(
		       color = ('rgb(0, 0, 8)'),
			width = 3,
		    ))
		pitau1_X=dh['Tau_Pi1_PZ_TRUE'][0]*10000/dh['Tau_Pi1_P_TRUE'][0]+tau_X
		pitau1_Y=dh['Tau_Pi1_PX_TRUE'][0]*10000/dh['Tau_Pi1_P_TRUE'][0]+tau_Y
		pitau1_Z=dh['Tau_Pi1_PY_TRUE'][0]*10000/dh['Tau_Pi1_P_TRUE'][0]+tau_Z
		tracepitau1=go.Scatter3d(x=[tau_X,pitau1_X],y=[tau_Y,pitau1_Y],z=[tau_Z,pitau1_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'pi'],
		      textposition='top center',
		      line = dict(
			color = ('rgb(0, 0, 7)'),
			width = 3,
		    ))
		pitau2_X=dh['Tau_Pi2_PZ_TRUE'][0]*10000/dh['Tau_Pi2_P_TRUE'][0]+tau_X
		pitau2_Y=dh['Tau_Pi2_PX_TRUE'][0]*10000/dh['Tau_Pi2_P_TRUE'][0]+tau_Y
		pitau2_Z=dh['Tau_Pi2_PY_TRUE'][0]*10000/dh['Tau_Pi2_P_TRUE'][0]+tau_Z
		tracepitau2=go.Scatter3d(x=[tau_X,pitau2_X],y=[tau_Y,pitau2_Y],z=[tau_Z,pitau2_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'pi'],
		      textposition='top center',
		      line = dict(
			   color = ('rgb(0, 0, 31)'),
			width = 3,
		    ))


		pitau3_X=dh['Tau_Pi3_PZ_TRUE'][0]*10000/dh['Tau_Pi3_P_TRUE'][0]+tau_X
		pitau3_Y=dh['Tau_Pi3_PX_TRUE'][0]*10000/dh['Tau_Pi3_P_TRUE'][0]+tau_Y
		pitau3_Z=dh['Tau_Pi3_PY_TRUE'][0]*10000/dh['Tau_Pi3_P_TRUE'][0]+tau_Z
		tracepitau3=go.Scatter3d(x=[tau_X,pitau3_X],y=[tau_Y,pitau3_Y],z=[tau_Z,pitau3_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'pi'],
		      textposition='top center',
		      line = dict(
		      color = ('rgb(0, 0, 31)'),
			width = 3,
		    ))
		nutau_X=dh['Tau_nu_PZ_TRUE'][0]*10000/dh['Tau_nu_P_TRUE'][0]+tau_X
		nutau_Y=dh['Tau_nu_PX_TRUE'][0]*10000/dh['Tau_nu_P_TRUE'][0]+tau_Y
		nutau_Z=dh['Tau_nu_PY_TRUE'][0]*10000/dh['Tau_nu_P_TRUE'][0]+tau_Z
		tracenutau=go.Scatter3d(x=[tau_X,nutau_X],y=[tau_Y,nutau_Y],z=[tau_Z,nutau_Z],
		      mode='lines+markers+text',
		      marker = dict( size = 5,color = "rgb(5,200,5)"),
		      text=['', 'nu'],
		      textposition='top center',
		      line = dict(
			color = ('rgb(0, 0,3.47)'),
			width = 3,
			dash='dash'
		    ))	
		data_one=[traceB,traceDst,tracetau,traceD0,tracenuB,traceK,tracepiDst,tracepitau1,tracepitau2,tracepitau3,tracenutau,tracepiK]
		layout_one=go.Layout(
			    showlegend=False,
			    width=800,
			    height=900,
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

		return {'data': data_one, 'layout': layout_one}




	
	
	

  
if __name__ == '__main__':
    app.run_server(debug=True)




