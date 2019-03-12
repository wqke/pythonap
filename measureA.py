import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import os
import root_pandas
import pandas as pd
import hepvector
from hepvector.numpyvector import Vector3D,LorentzVector
import numpy as np
from numpy import cos,sin,tan,sqrt,absolute,real,conjugate,imag,abs,max,min
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


dft=root_pandas.read_root('model_tree.root',key='DecayTree')

dft['W_PX_TRUE']=dft['B_PX_TRUE']-dft['Dst_PX_TRUE']
dft['W_PY_TRUE']=dft['B_PY_TRUE']-dft['Dst_PY_TRUE']
dft['W_PZ_TRUE']=dft['B_PZ_TRUE']-dft['Dst_PZ_TRUE']
dft['W_E_TRUE']=dft['B_E_TRUE']-dft['Dst_E_TRUE']
df=dft

B=LorentzVector(df['B_PX_TRUE'],df['B_PY_TRUE'],df['B_PZ_TRUE'],df['B_E_TRUE'])
W=LorentzVector(df['W_PX_TRUE'],df['W_PY_TRUE'],df['W_PZ_TRUE'],df['W_E_TRUE'])
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
q2=(B-Dst).mag2

###
#1D fit
x=costhetast
bin_heights, bin_borders, _=plt.hist(x,density=True,bins=bins)
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
#def func1 ...
popt, _ = curve_fit(func1, bin_centers, bin_heights)
###

chi= chi[~np.isnan(chi)]
q2=q2[:len(chi)]

#3D histogram
q2_heights, q2_borders, _=plt.hist(q2,density=True,bins=np.linspace(min(q2),max(q2),100))
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = chi,q2
hist, xedges, yedges = np.histogram2d(x, y, bins=100, range=[[-np.pi, np.pi], [min(q2), max(q2)]],normed=True)
plt.close()
x_centers = xedges[:-1] + np.diff(xedges) / 2
y_centers = yedges[:-1] + np.diff(yedges) / 2

xv,yv=np.meshgrid(x_centers, y_centers)

fig = plt.figure(figsize=plt.figaspect(0.35))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xv, yv, np.transpose(hist), cmap=cm.coolwarm)
ax.set_xlabel ('chi')
ax.set_ylabel ('q2')
plt.close()

def func1(x,A):
        res=1+A*cos(2*x)
        return res
    
Alist=[]
    
q2list=y_centers
    
for i in range(len(q2list)):
        coord=2*np.pi*hist[:-1,i]/q2_heights[i]
        popt, _ = curve_fit(func1, x_centers[:-1],coord)
        Alist.append(popt[0])
        
plt.plot(q2list,Alist)



""""""


chi= chi[~np.isnan(chi)]
q2=q2[:len(chi)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = chi,q2
hist, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[-np.pi, np.pi], [min(q2), max(q2)]],normed=True)

x_centers = xedges[:-1] + np.diff(xedges) / 2
y_centers = yedges[:-1] + np.diff(yedges) / 2

xv,yv=np.meshgrid(x_centers, y_centers)

fig = plt.figure(figsize=plt.figaspect(0.35))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xv, yv, np.transpose(hist), cmap=cm.coolwarm)
ax.set_xlabel ('chi')
ax.set_ylabel ('q2')

q2_heights, q2_borders, _=plt.hist(q2,density=True,bins=50)#bins=np.linspace(min(q2),max(q2),50))
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2



def func1(x,A):
        res=1+A*cos(2*x)
        return res
Alist=[]
q2list=q2_centers
for i in range(len(q2list)):
        coord=2*np.pi*hist[:,i]/q2_heights[i]
        popt, _ = curve_fit(func1, x_centers[:],coord)
        Alist.append(popt[0])
        
        
        





