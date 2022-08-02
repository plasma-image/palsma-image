# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:41:47 2020

@author: amirc
"""

import numpy as np 
import math
import matplotlib.pyplot as plt
import sys
#from functions_of_thickness import*
from general_functions import *
#from newanalysis_for_characterization import*
#from file_functions import*
import pickle



#jz_data=np.load('jz_t50point0.npz')
#uiz_data=np.load('uiz_t50point0.npz')
#rho_data=np.load('rho_t50point0.npz')


jz=jz_data['jz']
uiz=uiz_data['uiz']
rho=rho_data['rho']


nx = jz.shape[0]
ny = jz.shape[1]


uez=uiz-jz/rho
ratio= abs(uez)/abs(uiz)
uiz_rms=np.sum(uiz**2)/(nx*ny)


x_indices=[index[0] for index in indexes_of_valid_local_maxima]
y_indices=[index[1] for index in indexes_of_valid_local_maxima]

uiz_peak=abs(uiz[x_indices,y_indices])
#ion_velocity = uiz_peak.tolist()

uez_peak=abs(uiz[x_indices,y_indices]-jz[x_indices,y_indices]/rho[x_indices,y_indices])
#electron_velocity = uez_peak.tolist()

ratio1 = uez_peak / uiz_peak

#ratio1 = [i / j for i, j in zip(electron_velocity,ion_velocity)] 

##############################################################################

ratio2 = abs(1 - 1/np.array(ave_thicknesses)**2)

# ratio2= [1-1/i**2 for i in ave_thicknesses]  
# e=[abs(number) for number in ratio2]

#############################################################################



# plotting -----------------------------
plt.rcParams['font.size'] = 20
plt.figure()
plt.scatter(ratio2[filter_by_ave_thickness], ratio1[filter_by_ave_thickness])
plt.xlabel('$|1-1/L^2|$')
plt.ylabel('$|u_{ez}|/|u_{iz}|$')
plt.grid()
plt.title('Filtered points')
plt.show()
plt.tight_layout()


plt.rcParams['font.size'] = 20
plt.figure()
plt.scatter(ratio2, ratio1)
plt.xlabel('$|1-1/L^2|$')
plt.ylabel('$|u_{ez}|/|u_{iz}|$')
plt.grid()
plt.title('All points')
plt.show()
plt.tight_layout()
