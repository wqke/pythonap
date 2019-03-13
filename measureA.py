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


###Calculate A_FB
def fitAFB(x,a,b,c,d):
        res=a*x+b*x**2/2.+c*x**3/3.+d
        return res

q2_heights, q2_borders, _=plt.hist(q2,bins=np.linspace(min(q2),max(q2),21))
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2
plt.close()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = costhetal,q2
hist, xedges, yedges = np.histogram2d(x, y, bins=20, range=[[-1, 1], [min(q2), max(q2)]])
plt.close()
x_centers = xedges[:-1] + np.diff(xedges)/2
y_centers = yedges[:-1] + np.diff(yedges)/2


xv,yv=np.meshgrid(x_centers, y_centers)

fig = plt.figure(figsize=plt.figaspect(0.35))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xv, yv, np.transpose(hist), cmap=cm.coolwarm)
ax.set_xlabel ('costheta_l')
ax.set_ylabel ('q2')
plt.close()

AFBlist=[]
AFBerr=[]
q2err=[]
q2list=y_centers


for i in range(len(q2list)):
        #coord=hist[...,i]
        coord=q2_heights
        popt, pcov= curve_fit(fitAFB, x_centers,coord)
        a,b,c,d=(popt[0],popt[1],popt[2],popt[3])
        afb=b/q2_heights[i]
        AFBlist.append(afb)
        aerr,berr,cerr,derr=np.sqrt(np.diag(pcov))
        
        errz=rab*np.sqrt( 2*aerr**2/(a-c)**2+2*aerr**2/(2*a+2*c)**2)
        RABerr.append(errz)
        q2err.append((max(q2)-min(q2))/20.)
        
plt.errorbar(q2list,RABlist, xerr=q2err,yerr=RABerr, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)

def power(x,c,d,e):
        res=c*x**2+d*x+e
        return res




"""
sol,_=curve_fit(power, q2list, RABlist, maxfev=2000)

plt.plot(np.linspace(3,12,50),power(np.linspace(3,12,50),sol[0],sol[1],sol[2]),color='r',label='parabolic fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$R_{A,B}$ ($q^2$)')
plt.title(r'$R_{A,B}$ curve fit',fontsize=14, color='black')
plt.legend()
"""





###Calculate R_LT
def fitRLT(x,a,c):
        res=a+c*x**2
        return res

q2_heights, q2_borders, _=plt.hist(q2,bins=np.linspace(min(q2),max(q2),21))
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
ax.set_xlabel ('costheta_l')
ax.set_ylabel ('q2')
plt.close()

RLTlist=[]
RLTerr=[]
q2err=[]
q2list=y_centers


for i in range(len(q2list)):
        coord=hist[...,i]
        popt, pcov= curve_fit(fitRLT, x_centers,coord)
        a,c=(popt[0],popt[1])
        aerr,cerr=np.sqrt(np.diag(pcov))
        rlt=(a+c)/(2*a)
        RLTlist.append(rlt)
        
        errz=rlt*np.sqrt( 2*aerr**2/(a+c)**2+aerr**2/(a)**2)
        RLTerr.append(errz)
        q2err.append((max(q2)-min(q2))/20.)
        
plt.errorbar(q2list,RLTlist, xerr=q2err,yerr=RLTerr, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)

def power(x,c,d,e):
        res=c*x**2+d*x+e
        return res
sol,_=curve_fit(power, q2list, RLTlist, maxfev=2000)
plt.plot(np.linspace(3,12,50),power(np.linspace(3,12,50),sol[0],sol[1],sol[2]),color='r',label='parabolic fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$R_{L,T}$ ($q^2$)')
plt.title(r'$R_{L,T}$ curve fit',fontsize=14, color='black')
plt.legend()







###Calculate R_AB
def fitRAB(x,a,b,c):
        res=a+b*x+c*x**2
        return res

q2_heights, q2_borders, _=plt.hist(q2,bins=np.linspace(min(q2),max(q2),21)) #density=True,
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2
plt.close()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = costhetal,q2
hist, xedges, yedges = np.histogram2d(x, y, bins=20, range=[[-1, 1], [min(q2), max(q2)]])
plt.close()
x_centers = xedges[:-1] + np.diff(xedges)/2
y_centers = yedges[:-1] + np.diff(yedges)/2


xv,yv=np.meshgrid(x_centers, y_centers)

fig = plt.figure(figsize=plt.figaspect(0.35))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xv, yv, np.transpose(hist), cmap=cm.coolwarm)
ax.set_xlabel ('costheta_l')
ax.set_ylabel ('q2')
plt.close()

RABlist=[]
RABerr=[]
q2err=[]
q2list=y_centers


for i in range(len(q2list)):
        coord=hist[...,i]
        popt, pcov= curve_fit(fitRAB, x_centers,coord)
        a,b,c=(popt[0],popt[1],popt[2])
        aerr,berr,cerr=np.sqrt(np.diag(pcov))
        rab=(a-c)/(2*(a+c))
        RABlist.append(rab)
        
        errz=rab*np.sqrt( 2*aerr**2/(a-c)**2+2*aerr**2/(2*a+2*c)**2)
        RABerr.append(errz)
        q2err.append((max(q2)-min(q2))/20.)
        
plt.errorbar(q2list,RABlist, xerr=q2err,yerr=RABerr, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)

def power(x,c,d,e):
        res=c*x**2+d*x+e
        return res
sol,_=curve_fit(power, q2list, RABlist, maxfev=2000)
plt.plot(np.linspace(3,12,50),power(np.linspace(3,12,50),sol[0],sol[1],sol[2]),color='r',label='parabolic fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$R_{A,B}$ ($q^2$)')
plt.title(r'$R_{A,B}$ curve fit',fontsize=14, color='black')
plt.legend()



###Calculate FD

def fitFD(x,FD,c):
        res=(2*FD*x**3/3.+(1-FD)*(x-x**3/3.)+c)/4.
        return res

q2_heights, q2_borders, _=plt.hist(q2,bins=np.linspace(min(q2),max(q2),201)) 
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = costhetast,q2
hist, xedges, yedges = np.histogram2d(x, y, bins=200)
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
        coord=hist[...,i]
        popt, pcov= curve_fit(fitFD, x_centers,coord)
        FDlist.append(popt[0]/q2_heights[i])
        FDerr.append(np.sqrt(np.diag(pcov)))
        q2err.append((max(q2)-min(q2))/200.)
        
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



#Calculate Ac3

##No nan requirement
costhetast= costhetast[~np.isnan(chi)]
costhetal= costhetal[~np.isnan(chi)]
q2=q2[~np.isnan(chi)]
chi= chi[~np.isnan(chi)]

q2_heights, q2_borders, _=plt.hist(q2,bins=np.linspace(min(q2),max(q2),21)) 
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y ,z , w= chi,q2,costhetal,costhetast
hist,edges= np.histogramdd((x, y,z,w), bins=(20,20,20,20),range=((-np.pi,np.pi),(min(q2),max(q2)),(-1,1),(-1,1)))
[xedges,yedges,zedges,wedges]=edges
plt.close()

x_centers = xedges[:-1] + np.diff(xedges)/2
y_centers = yedges[:-1] + np.diff(yedges)/2
z_centers = zedges[:-1] + np.diff(zedges)/2
w_centers = wedges[:-1] + np.diff(wedges)/2
step=2/20.

newhist=np.zeros((20,20,20))

Ip=


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



#3D histogram, fit A1

##No nan requirement
q2=q2[~np.isnan(chi)]
chi= chi[~np.isnan(chi)]


def func2(x,a,cc,cs):
        res=a+cc*cos(2*x)+cs*sin(2*x)
        return res


q2_heights, q2_borders, _=plt.hist(q2,bins=np.linspace(min(q2),max(q2),201))
q2_centers = q2_borders[:-1] + np.diff(q2_borders) / 2
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = chi,q2
hist, xedges, yedges = np.histogram2d(x, y, bins=200)#, range=[[-np.pi, np.pi], [min(q2), max(q2)]])
plt.close()
x_centers = xedges[:-1] + np.diff(xedges) / 2
y_centers = yedges[:-1] + np.diff(yedges) / 2

xv,yv=np.meshgrid(x_centers, y_centers)

fig = plt.figure(figsize=plt.figaspect(0.35))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xv, yv, np.transpose(hist), cmap=cm.coolwarm)
plt.close()

    
A1list=[]
A1err=[]    
q2err=[]
q2list=y_centers
    
for i in range(len(q2list)):
        coord=hist[:,i]
        popt, pcov = curve_fit(func2, x_centers,coord)
        a,cc,cs=(popt[0],popt[1],popt[2])
        A1list.append(cs/q2_heights[i])
        aerr,ccerr,cserr=np.sqrt(np.diag(pcov))
        A1err.append(cserr/q2_heights[i])
        q2err.append((max(q2)-min(q2))/200.)
        

plt.errorbar(q2list,A1list, xerr=q2err,yerr=A1err, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)

def power(x,d,e):
        res=d*x+e
        return res
sol,_=curve_fit(power, q2list, A1list, maxfev=2000)
plt.plot(np.linspace(3,12,50),power(np.linspace(3,12,50),sol[0],sol[1]),color='r',label='linear fit')
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_9({q^2})$')
plt.ylim([-0.004,0.004])
plt.title(r'$A_9({q^2})$ curve fit',fontsize=14, color='black')
plt.legend()


""""""







