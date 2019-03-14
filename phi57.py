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
from matplotlib import cm,rc


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

q2=q2[~np.isnan(chi)]
costhetal=costhetal[~np.isnan(chi)]
costhetast=costhetast[~np.isnan(chi)]
chi= chi[~np.isnan(chi)]


q2_1=list(set(q2[costhetast>0.9]) & set(q2[costhetast<=1]))
chi_1=list(set(chi[costhetast>0.9]) & set(chi[costhetast<=1]))

chi_0=list(set(chi[costhetast<0.07171]) & set(chi[costhetast>-0.07171]))
q2_0=list(set(q2[costhetast<0.07171]) & set(q2[costhetast>-0.07170]))

chi_m1=list(set(chi[costhetast<-0.8852]) & set(chi[costhetast>=-1]))
q2_m1=list(set(q2[costhetast<-0.8852]) & set(q2[costhetast>=-1]))

q2_heights, q2_borders, _=plt.hist(q2,bins=4) 
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2
plt.close()


x0, y0 = chi_0,q2_0
hist0, xedges0, yedges0 = np.histogram2d(x0, y0, bins=4)
plt.close()

x1, y1 = chi_1,q2_1
hist1, xedges1, yedges1 = np.histogram2d(x1, y1, bins=4)
plt.close()

xm1, ym1 = chi_m1,q2_m1
histm1, xedgesm1, yedgesm1 = np.histogram2d(xm1, ym1, bins=4)
plt.close()

x_centers1= xedges1[:-1] + np.diff(xedges1)/2
y_centers1 = yedges1[:-1] + np.diff(yedges1)/2
x_centers0 = xedges0[:-1] + np.diff(xedges0)/2
y_centers0 = yedges0[:-1] + np.diff(yedges0)/2
x_centersm1 = xedgesm1[:-1] + np.diff(xedgesm1)/2
y_centersm1 = yedgesm1[:-1] + np.diff(yedgesm1)/2


phi57=hist0*2-hist1-histm1

"""""""

q1=list(set(q2[chi>-0.1]) & set(q2[costhetast<0.1]) & set(q2[chi<0.1]) &set(q2[chi>-0.1]))
q22=list(set(q2[costhetast>-0.1]) & set(q2[costhetast<0.1]) & set(q2[chi<min(chi)+0.188]) &set(q2[chi>=min(chi)]))
q3=list(set(q2[costhetast>=-1]) & set(q2[costhetast<-0.9]) & set(q2[chi<0.15]) &set(q2[chi>-0.15]))
q4=list(set(q2[costhetast>=-1]) & set(q2[costhetast<-0.85]) & set(q2[chi<min(chi)+0.1998]) &set(q2[chi>=min(chi)]))

q5=list(set(q2[costhetast>0.85]) & set(q2[costhetast<=1]) & set(q2[chi<=max(chi)]) &set(q2[chi>max(chi)-0.187]))
q6=list(set(q2[costhetast>-0.09]) & set(q2[costhetast<=0.09]) & set(q2[chi<=max(chi)]) &set(q2[chi>max(chi)-0.2085]))
q7=list(set(q2[costhetast>0.8695]) & set(q2[costhetast<1]) & set(q2[chi<=max(chi)]) &set(q2[chi>max(chi)-0.2085]))
q8=list(set(q2[costhetast>0.8697]) & set(q2[costhetast<1]) & set(q2[chi<0.07]) &set(q2[chi>-0.0964]))

q9=list(set(q2[costhetast>-0.1]) & set(q2[costhetast<0.1]) & set(q2[chi<np.pi]) &set(q2[chi>np.pi-0.1926]))
q10=list(set(q2[costhetast>-1]) & set(q2[costhetast<-0.9]) & set(q2[chi<0.1]) &set(q2[chi>-0.2]))
q11=list(set(q2[costhetast>-0.1]) & set(q2[costhetast<0.1]) & set(q2[chi<0.1]) &set(q2[chi>-0.1]))
q12=list(set(q2[costhetast>-1]) & set(q2[costhetast<-0.85]) & set(q2[chi<np.pi]) &set(q2[chi>np.pi-0.2059]))


q13=list(set(q2[costhetast>-0.1]) & set(q2[costhetast<0.1]) & set(q2[chi<-np.pi+0.188]) &set(q2[chi>-np.pi]))
q14=list(set(q2[costhetast>0.89]) & set(q2[costhetast<1]) & set(q2[chi<0.1]) &set(q2[chi>-0.0977]))
q15=list(set(q2[costhetast>-0.1]) & set(q2[costhetast<0.1]) & set(q2[chi<0.1]) &set(q2[chi>-0.0999]))
q16=list(set(q2[costhetast>0.83]) & set(q2[costhetast<1]) & set(q2[chi<0.1562-np.pi]) &set(q2[chi>-np.pi]))

heights=plt.hist(q1,bins=4)[0]-plt.hist(q22,bins=4)[0]-plt.hist(q3,bins=4)[0]+plt.hist(q4,bins=4)[0]+plt.hist(q5,bins=4)[0]-plt.hist(q6,bins=4)[0]-plt.hist(q7,bins=4)[0]+plt.hist(q8,bins=4)[0]-plt.hist(q9,bins=4)[0]-plt.hist(q10,bins=4)[0]+plt.hist(q11,bins=4)[0]+plt.hist(q12,bins=4)[0]-plt.hist(q13,bins=4)[0]-plt.hist(q14,bins=4)[0]+plt.hist(q15,bins=4)[0]+plt.hist(q16,bins=4)[0] 





