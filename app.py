# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import flask
from flask_cors import CORS
import os
import root_pandas
import pandas as pd
import hepvector
from hepvector.numpyvector import Vector3D,LorentzVector
import plotly
import plotly.graph_objs as go


df=root_pandas.read_root('model_tree.root',key='DecayTree')         #Read the data ->download root in git or copy root in putty



########

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


unittau=(nouvtau.p3).unit
unitnu=(nouvnu.p3).unit
unitDst=(newDst.p3).unit
unitD0=(nouvD0.p3).unit

normal1=unittau.cross(unitDst)
normal2=unitDst.cross(unitD0)
coski=normal1.dot(normal2)
costhetast=unitD0.dot(unitDst)
costhetal=unitDst.dot(unittau)

chi=np.arccos(coski)
thetast=np.arccos(costhetast)
thetal=np.arccos(costhetal)

######

x, y, z = thetast,thetal,chi

#The Phase Space scatter plot
trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=z,    # modify              
        colorscale='Viridis',  
        opacity=0.8
    )
)


#######


#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('visual-decay')                                                 #The name of the app
server = app.server



app.layout = html.Div([
html.Div(children=[
    html.H1(children='Hello Dash'),
    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='PhaseSpace',
        figure={
            'data': [
                trace1
            ],
            'layout': {
                'title': 'Phase Space'
            }
        }
    )
])

    """
html.Div([
    dcc.Input(id='my-id', value='initial value', type='text'),
    html.Div(id='my-div')
])
    

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):                                     #Basically, def(input) and return output, of the callback
    return 'You\'ve entered "{}"'.format(input_value)
"""

#if __name__ == '__main__':
#app.run_server()

if __name__ == '__main__':
    app.run_server(debug=True)


