
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:14:41 2019

@author: ke
"""
import dash
import os
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

from local import local_layout, local_callbacks



import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import root_pandas
import pandas as pd
import hepvector
from hepvector.numpyvector import Vector3D,LorentzVector

df=root_pandas.read_root('model_tree.root',key='DecayTree')

import plotly
import plotly.graph_objs as go

import plotly.plotly as py 
import plotly.tools as tls


from plotly.graph_objs import Data, Layout, Figure
from plotly.graph_objs import Scatter

import numpy as np
from numpy import cos,sin,tan,sqrt,absolute,real,conjugate,imag,abs,max,min

###This is the data###


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



histo=go.Histogram(x=costhetast)


###This is the dash code###

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']  #A modifier

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Histogram'),
    html.Div(children='''
        Dash: My first histogram 
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
              histo,
            ],
            'layout': {
                'title': 'costheta'
            }
        }
    )
])









if __name__ == '__main__':
    app.run_server(debug=True)

