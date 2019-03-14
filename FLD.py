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



q2_heights, q2_borders, _=plt.hist(q2,bins=np.linspace(min(q2),max(q2),11)) 
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2
plt.close()

""""""

def fitF1(x,a):
  res=(0.5*a*x**2)+(0.25-0.25*a)*(1-x**2)
  return res
q2list=q2_centers
Flist=[]
Ferr=[]
q2err=[]

for i in range(10):
  set1=list(set(costhetast[q2>q2_borders[i]]) & set(costhetast[q2<q2_borders[i+1]]))
  bin_heights, bin_borders, _=plt.hist(set1,bins=10,density=1/q2_heights[i])
  bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
  popt, pcov = curve_fit(fitF1, bin_centers, bin_heights)
  a=popt[0]
  Flist.append(a)
  aerr=np.sqrt(np.diag(pcov))
  Ferr.append(aerr[0])
  q2err.append((max(q2)-min(q2))/20.)
  plt.close()

  
plt.errorbar(q2list,Flist, xerr=q2err,yerr=Ferr, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
def power(x,c,d,e):
  res=c*x**2+d*x+e
  return res

sol,_=curve_fit(power, q2list, Flist, maxfev=2000)
plt.plot(np.linspace(3,12,50),power(np.linspace(3,12,50),sol[0],sol[1],sol[2]),color='r',label='parabolic fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$F_{L}^{D^*}$ ($q^2$)')
plt.title(r'$F_{L}^{D^*}$ ($q^2$)',fontsize=14, color='black')
plt.legend()
plt.ylim(-0.2,1.4)
plt.xlim(3,11)
plt.grid(linestyle='-', linewidth='0.5', color='gray')
