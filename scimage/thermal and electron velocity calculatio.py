# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:13:49 2020

@author: amirc
"""

import numpy as np 
import math
import matplotlib.pyplot as plt
import sys
#from functions_of_thickness import*
from general_functions import *
#from file_functions import*
import pickle


jz_data=np.load('jz_t50point0.npz')
uiz_data=np.load('uiz_t50point0.npz')
rho_data=np.load('rho_t50point0.npz')


jz=jz_data['jz']
uiz=uiz_data['uiz']
rho=rho_data['rho']


nx = jz.shape[0]
ny = jz.shape[1]

uez=uiz-jz/rho
ratio= abs(uez)/abs(uiz)
uiz_rms=np.sum(uiz**2)/(nx*ny)

x_indices=[index[0] for index in indexes_of_local_jzmax]
y_indices=[index[1] for index in indexes_of_local_jzmax]

uez_max=uiz[x_indices,y_indices]-jz[x_indices,y_indices]/rho[x_indices,y_indices]
electron_velocity = uez_max.tolist()
thermal_velocity= np.sqrt(0.5)

p= abs(jz[x_indices,y_indices]).tolist()
l=value_of_pjz_peak


q=abs(uez_max).tolist()


plt.rcParams['font.size']=10
plt.rcParams['xtick.labelsize']=10
plt.rcParams['ytick.labelsize']=10
 
plt.hist(q,color = 'darkblue', edgecolor = 'black')
plt.legend(prop={'size': 10}, title = '$J_{thr}=j_{rms} $\n$n=30$ \n$\omega_{ci}t=50$',title_fontsize=12)
plt.xlabel('Electron velocity ',fontsize=11)
plt.ylabel('Number',fontsize=11) 
plt.show() 

 
plt.hist(l,color = 'darkblue', edgecolor = 'black')
plt.legend(prop={'size': 10}, title = '$J_{thr}=j_{rms} $\n$n=30$ \n$\omega_{ci}t=50$',title_fontsize=20)
plt.xlabel('jz peak values ',fontsize=11)
plt.ylabel('Number',fontsize=11) 
plt.show() 

