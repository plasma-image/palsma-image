#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:06:36 2020

@author: a.azizabadi
"""

import numpy as np
import os
    
# ----- temporary: test file functions -------
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Created new folder:", folder_name)

def get_folder_name(real_data,average,cs_condition, 
                   lx, ly, nx, ny,J_th,J_rms, 
                   ratio_of_jzboundary_to_jzmax,ratio_of_uez_to_uiz,
                   number_of_points_upto_local_boundary):

    simulation_params = 'lx'+str(int(lx))+'_ly'+str(int(ly))+'_nx'+str(nx)+'_ny'+str(ny)

    if cs_condition:
        algorithm_parameters = 'ratio_of_uez_to_uiz'+str(int(ratio_of_uez_to_uiz))+'_ratio_of_jzboundary_to_jzmax_'+str(ratio_of_jzboundary_to_jzmax)+'_N'+str(number_of_points_upto_local_boundary)
    else:
        algorithm_parameters = 'J_th'+str(np.round(J_th,2))+'_J_rms'+str(np.round(J_rms,2))+'_ratio_of_jzboundary_to_jzmax_'+str(ratio_of_jzboundary_to_jzmax)+'_N'+str(number_of_points_upto_local_boundary)

        
    if real_data :
        if average:            
            folder_name = 'real_data_average_'+simulation_params+'_PARAMS_'+algorithm_parameters
        else:
            folder_name = 'real_data_'+simulation_params+'_PARAMS_'+algorithm_parameters

        if cs_condition:
            folder_name = 'cs_condition_'+'nx'+str(nx)+'_ny'+str(ny)+'_PARAMS_'+algorithm_parameters
    else:
        folder_name = 'generated_data_'+simulation_params+'_PARAMS_'+algorithm_parameters            

    return folder_name


# ---------------------- TEST IT ------------------------------------------
## Save them 
#np.savez("cs_detections", 
#         indexes_of_local_jzmax = indexes_of_local_jzmax, 
#         indexes_of_points_of_all_cs = indexes_of_points_of_all_cs)
#
##halfthickness1
##halfthickness2
##lengths_in_global_space
##np.savez("cs_characterizations", 
##         ...)
#
## Load them (Note: loads them as array not list)
#data = np.load("cs_detections.npz")
#cs_maximas = data['indexes_of_local_jzmax']
#cs_points = data['indexes_of_points_of_all_cs']
