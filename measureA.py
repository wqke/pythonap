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

###
#1D fit
x=costhetast
bin_heights, bin_borders, _=plt.hist(x,density=True,bins=bins)
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
#def func1 ...
popt, _ = curve_fit(func1, bin_centers, bin_heights)
###


###Calculate polarisation : costhetast

def fitFD(x,FD):
        res=(2*FD*x**2+(1-FD)*(1-x**2))/4.
        return res

q2_heights, q2_borders, _=plt.hist(q2,bins=np.linspace(min(q2),max(q2),21)) #density=True,
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = costhetast,q2
hist, xedges, yedges = np.histogram2d(x, y, bins=20, range=[[-1, 1], [min(q2), max(q2)]])
plt.close()
x_centers = xedges[:-1] + np.diff(xedges)/2
y_centers = yedges[:-1] + np.diff(yedges)/2

xv,yv=np.meshgrid(x_centers, y_centers)

fig = plt.figure(figsize=plt.figaspect(0.35))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xv, yv, np.transpose(hist), cmap=cm.coolwarm)
ax.set_xlabel ('costheta*')
ax.set_ylabel ('q2')
plt.close()


FDlist=[]
FDerr=[]
q2err=[]
q2list=y_centers
    
for i in range(len(q2list)):
        coord=hist[...,i]/q2_heights[i]
        popt, pcov= curve_fit(fitFD, x_centers,coord)
        FDlist.append(popt[0])
        FDerr.append(np.sqrt(np.diag(pcov)))
        q2err.append((max(q2)-min(q2))/20.)
        
plt.errorbar(q2list,FDlist, xerr=q2err,yerr=FDerr, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)

def power(x,c,d,e):
        res=c*x**2+d*x+e
        return res
sol,_=curve_fit(power, q2list, FDlist, maxfev=2000)
plt.plot(np.linspace(3,12,50),power(np.linspace(3,12,50),sol[0],sol[1],sol[2]),color='r',label='parabolic fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$F_L^{D^*}$ ($q^2$)')
plt.title(r'$F_L^{D^*}$ curve fit',fontsize=14, color='black')
plt.legend()



#3D histogram, fit A

##No nan requirement
q2=q2[~np.isnan(chi)]
chi= chi[~np.isnan(chi)]

q2_heights, q2_borders, _=plt.hist(q2,bins=np.linspace(min(q2),max(q2),21))
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = chi,q2
hist, xedges, yedges = np.histogram2d(x, y, bins=20)#, range=[[-np.pi, np.pi], [min(q2), max(q2)]])
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
Aerr=[]    
q2err=[]
q2list=y_centers
    
for i in range(len(q2list)):
        coord=2*hist[:,i]/q2_heights[i]
        popt, pcov = curve_fit(func1, x_centers,coord)
        Alist.append(popt[0])
        Aerr.append(np.sqrt(np.diag(pcov)))
        q2err.append((max(q2)-min(q2))/20.)
        

plt.errorbar(q2list,Alist, xerr=q2err,yerr=Aerr, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)

def power(x,c,d,e):
        res=c*x**2+d*x+e
        return res
sol,_=curve_fit(power, q2list, Alist, maxfev=2000)
plt.plot(np.linspace(3,12,50),power(np.linspace(3,12,50),sol[0],sol[1],sol[2]),color='r',label='parabolic fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_C^{(1)}({q^2})$')
plt.title(r'$A_C^{(1)}({q^2})$ curve fit',fontsize=14, color='black')
plt.legend()


""""""







