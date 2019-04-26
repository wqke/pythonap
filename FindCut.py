# Copyright 2017 CERN for the benefit of the LHCb collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import matplotlib
matplotlib.use('Agg')  #fix python2 tkinter problem

import tensorflow as tf
import numpy as np
import collections

import sys, os, math
sys.path.append("../../TensorFlowAnalysis")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU

import TensorFlowAnalysis as tfa

from ROOT import TFile, TChain, TH3F
from root_numpy import root2array, rec2array, tree2array

import matplotlib.pyplot as plt
import rootplot.root2matplotlib as r2m
from scipy.stats import norm as sci_norm 
from scipy.stats import sem as sem
import matplotlib.mlab as mlab
from root_pandas import to_root, read_root
from uncertainties import *
import pandas as pd
import random
import math


def MakeHistogram(phsp, sample, bins, weights = None, normed = False) : 
  hist = np.histogramdd(sample, bins = bins, range = phsp.Bounds(), weights = weights, normed = normed )
  return hist[0]  # Only return the histogram itself, not the bin boundaries

def MakeHistogram_1D(sample, bins, weights = None, normed = False, density = None) : 
  hist = np.histogram(sample, bins = bins, normed = normed, weights = weights, density = density)
  return hist[0]  # Only return the histogram itself, not the bin boundaries
  
def HistogramNorm(hist) : 
  return np.sum( hist )

def BinnedChi2(hist1, hist2, err) :
	return tf.reduce_sum( ((hist1 - hist2)/err)**2 )



M_B = 5.27963
M_Dst = 2.01026
q2_max = (M_B - M_Dst)**2

#select the last 20% of the q2 interval
q2_min=q2_max*0.8

print [q2_min,q2_max]

#background fractions
frac={}

#Fractions defined using Run 1 R(D*) fit
n_signal = 1296.
feed_frac = 0.11
n_feed = feed_frac*n_signal
n_ds = 6835.
n_d0 = 1.41 * 445.
n_dplus = 0.245 * n_ds
n_prompt = 424.
total_yield = n_signal + n_ds + n_dplus + n_feed + n_d0 + n_prompt

frac['signal'] = n_signal/total_yield  #floating
frac['Ds'] = n_ds/total_yield
frac['Dplus']= n_dplus/total_yield
frac['feed'] = n_feed/total_yield
frac['D0'] = n_d0/total_yield
frac['prompt'] = n_prompt/total_yield


evt={}

evt['signal'] =100000* n_signal/total_yield 
evt['Ds'] = 100000*n_ds/total_yield
evt['Dplus']= 100000*n_dplus/total_yield
evt['feed'] = 100000*n_feed/total_yield
evt['D0'] = 100000*n_d0/total_yield
evt['prompt'] = 100000*n_prompt/total_yield

bkg_names = list(frac)
bkg_files={}
for bkg in bkg_names:
  bkg_files[bkg] = "/data/lhcb/users/hill/Bd2DstTauNu_Angular/RapidSim_tuples/Merged_Bkg/%s.root" % bkg

tot=0.
tot1=0.
for bkg in bkg_names:
  df=read_root(bkg_files[bkg],key="DecayTree",columns=["q2_reco"])
  tot+=len(df)
  df=df.sample(n=int(evt[bkg]),random_state=int(evt[bkg]))
  df=df.query("q2_reco > %s and q2_reco <= %s" % (q2_min,q2_max))
  tot1+=len(df)
  print bkg,":",len(df)
  
print(tot1/tot)
  
  
  
from root_pandas import *	
def inter(lst1, lst2):
    return list(set(lst1) & set(lst2))

columns=[ 'Tau_FD_z',  'Tau_M', 'Tau_E','Tau_P', '3pi_M', '3pi_PZ', 'Tau_m12', 'Tau_m13','Tau_m23',
         'Tau_FD', 'costheta_D_reco','costheta_L_reco','q2_reco','Tau_PZ_reco','Tau_PT',
        'chi_reco', 'Tau_life_reco']

