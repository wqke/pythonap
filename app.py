

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import root_pandas
import pandas as pd
import hepvector
from hepvector.numpyvector import Vector3D,LorentzVector
dft=root_pandas.read_root('~/home/ke/model_tree.root',key='DecayTree')
df=dft.head(10)

import plotly
import plotly.graph_objs as go

import plotly.plotly as py 
import plotly.tools as tls


from plotly.graph_objs import Data, Layout, Figure
from plotly.graph_objs import Scatter

import numpy as np
from numpy import cos,sin,tan,sqrt,absolute,real,conjugate,imag,abs,max,min




plotly.tools.set_credentials_file(username='wke', api_key='gUC6DxjFj6NsTBewEDb5')


plotly.tools.set_config_file(world_readable=True,
                             sharing='public')







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


x, y, z = thetast,thetal,chi

trace1 = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=5,color=z,colorscale='Viridis',opacity=0.8))
data = [trace1]

layout = go.Layout(showlegend=False,width=800,height=900,autosize=False,margin=dict(t=0, b=0, l=0, r=0),scene=dict(xaxis=dict(title='$\\theta *$'),gridcolor='#bdbdbd',gridwidth=2,zerolinecolor='#969696',zerolinewidth=4,linecolor='#636363',linewidth=4,showbackground=True,backgroundcolor='rgb(230, 230,230)'),yaxis=dict(title='$\\theta_l$'),gridcolor='#bdbdbd',gridwidth=2,zerolinecolor='#969696',zerolinewidth=4,linecolor='#636363',linewidth=4,showbackground=True,backgroundcolor='rgb(230, 230, 230)'),zaxis=dict(title='$\\chi$' ),        gridcolor='#bdbdbd',        gridwidth=2,        zerolinecolor='#969696',        zerolinewidth=4,        linecolor='#636363',        linewidth=4,            showbackground=True,            backgroundcolor='rgb(230, 230,230)'        ),        aspectratio = dict(x=1, y=1, z=0.7),aspectmode = 'manual'))

    
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='3d-scatter-colorscale')


