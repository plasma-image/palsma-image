#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:36:54 2019

"""

import numpy as np 
import math
import matplotlib.pyplot as plt
import os
import pickle
import sys
import pandas as pd

def generate_cs_in_turbulence():
    # This function generates a chosen number of current sheets embedded inside a background
    # noise in a 2-D box. It returns coordinates along x- and y-axis of the box and current
    # density in the variables "x" and "y" and "jz", respectively. It is called as
    # x,y,jz=generate_cs_in_turbulence().

    # A current sheet in the 2-D box is generated with its center (location
    # of the peak current density) at a random location (xc,yc) and its size controlled by its
    # extents (Lcs_x1 and Lcs_x2) perpendicular to  two mutually perpendicular lines with
    # slopes m1 and m2=-1/m1 (L1:y-yc=m1*(x-xc), L2:y-yc=m2*(x-xc)) intersecting at the CS
    # center using a mathematical function Jz(x,y)= jz0*sech^2(x1/Lcs_x1)*sech^2(x2/Lcs_x2).
    # Here x1=(y-yc-m1*(x-xc))/sqrt(1+m1^2) and x2=(y-yc-m2*(x-xc))/sqrt(1+m2^2) are the
    # perpendicular distances of the point (x,y) from the lines L1 and L2, respectively. The
    # slopes of the lines are chosen randomly to given current sheets random orientation in the
    # 2-D box. The peak current density (jz0)  and extents or thicknesses (Lcs_x1, Lcs_x2)
    # perpendicular to the lines L1 and L2 and peak current density are also randomly
    # distributed.     
    nx=512
    ny=512
    lx=256
    ly=256
    dx=lx/nx
    dy=ly/ny
    xx=np.linspace(-lx/2,lx/2,nx)
    yy=np.linspace(-ly/2,ly/2,ny)
    jz=np.zeros([nx,ny])
    
    #Total number of current sheets to be generated in the 2-D box
    number_of_CS=15
    
    #Slope of the line L1 for each current sheet will be randomly chosen
    #between -max_slope and +max_slope. 
    max_slope_L1=5
    
    #peak current density for each current sheet will be randomly chosen
    #between -jz_peak_max and jz_peak_max.
    jz_peak_max=1
    
    #noise level is a fraction of jz_pak_max
    noise_level=0.01*jz_peak_max
    
    #Thicknesses (perpendicular to the lines L1 and L2) for each current sheet is chosen
    #randomly from a normal distribution with a mean of mean_thickness and standard
    #deviation of spread_thickness, both specified in terms of grid spacing. 
    mean_thickness_L1=20*dx
    spread_thickness_L1=3*dx
    mean_thickness_L2=5*dx
    spread_thickness_L2=3*dx
    
    seed_val = 123456
    np.random.seed(seed_val)
    
    # getting a random distribution of x- and y-positions of CS centers 
    # THESE ARE DESIRED LOCATIONS OF MAXIMA POINTS *
    x_positions_of_cs=np.random.uniform(-lx/2,lx/2,number_of_CS)
    y_positions_of_cs=np.random.uniform(-ly/2,ly/2,number_of_CS)
    
    # getting a random distribution of the slopes of line L1 
    slopes_line1=np.random.uniform(-max_slope_L1,max_slope_L1,number_of_CS)
    slopes_line2=-1/slopes_line1 #two lines are perpendicular
    
    # getting a random distribution of peak values of curent density 
    peaks_jz=np.random.uniform(-jz_peak_max,jz_peak_max,number_of_CS)
    print ("Chosen peaks of current density:")
    print (peaks_jz)
    
    # getting a random distribution of CS thickneses perpendicular to L1 and L2    
    thicknesses_of_cs_x1=np.random.normal(mean_thickness_L1,spread_thickness_L1,number_of_CS)
    thicknesses_of_cs_x2=np.random.normal(mean_thickness_L2,spread_thickness_L2,number_of_CS)
   
    for ix, x in enumerate(xx):
        for iy, y in enumerate(yy):
            for xc, yc, m1, m2, jz0, Lcs_x1, Lcs_x2 in \
                zip(x_positions_of_cs,y_positions_of_cs,slopes_line1,slopes_line2,\
                    peaks_jz,thicknesses_of_cs_x1,thicknesses_of_cs_x2):
                #jz[i,j]=random.random()
                x1=(y-yc-m1*(x-xc))/np.sqrt(1+m1**2)
                x2=(y-yc-m2*(x-xc))/np.sqrt(1+m2**2)
                jz[ix,iy]=jz[ix,iy]+(jz0/np.cosh(x1/Lcs_x1)**2)*(1/np.cosh(x2/Lcs_x2)**2)
                
    jz_noise=jz+np.random.uniform(-noise_level,noise_level,[nx,ny])
    
    # Optional:
    #sheet_filename = 'seed'+str(seed_val)+'-maxslope'+str(max_slope_L1)+'-noise_level'+str(noise_level)+'-'
    #save_generated_sheets('generated_sheets', sheet_filename, xx,yy,jz_noise, x_positions_of_cs, y_positions_of_cs)
    
    return xx,yy,jz_noise, x_positions_of_cs, y_positions_of_cs, thicknesses_of_cs_x1, thicknesses_of_cs_x2, nx, ny, lx, ly
####End of the function generate_cs_in_turbulence
##########################################################################################3

def save_generated_sheets(folder, filename_prefix, x,y,jz, generated_x_positions_of_cs, generated_y_positions_of_cs):    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path_prefix = folder+'/'+filename_prefix
    
    with open(path_prefix+'x.data', 'wb') as file:
        pickle.dump(x, file)
    with open(path_prefix+'y.data', 'wb') as file:
        pickle.dump(y, file)
    with open(path_prefix+'jz.data', 'wb') as file:
        pickle.dump(jz, file)
    with open(path_prefix+'maxima_x_pos.data', 'wb') as file:
        pickle.dump(generated_x_positions_of_cs, file)
    with open(path_prefix+'maxima_y_pos.data', 'wb') as file:
        pickle.dump(generated_y_positions_of_cs, file)

def load_generated_sheets(filename_prefix):    
    with open(filename_prefix+'x.data', 'rb') as file:
        x=pickle.load(file)
    with open(filename_prefix+'y.data', 'rb') as file:
        y=pickle.load(file)
    with open(filename_prefix+'jz.data', 'rb') as file:
        jz=pickle.load(file)
    with open(filename_prefix+'maxima_x_pos.data', 'rb') as file:
        generated_x_positions_of_cs=pickle.load(file)
    with open(filename_prefix+'maxima_y_pos.data', 'rb') as file:
        generated_y_positions_of_cs=pickle.load(file)
        
    return x,y,jz, generated_x_positions_of_cs, generated_y_positions_of_cs
########################################################################

def get_indexes_where_array_exceeds_threshold(array, threshold):
    indexes_where_array_exceeds_threshold = []
    for index, value in np.ndenumerate (array):
        if (value > threshold):
            indexes_where_array_exceeds_threshold.append(index)
    return indexes_where_array_exceeds_threshold
#### End of the function get_indexes_where_array_exceeds_threshold
########################################################################
def get_indexes_where_uez_gt_uizrms(uez,uiz_rms, threshold):
    indexes_where_uez_gt_uizrms = []
    for index, value in np.ndenumerate (uez):
        if (np.abs(value) > threshold*uiz_rms):
            indexes_where_uez_gt_uizrms.append(index)
    return indexes_where_uez_gt_uizrms
#### End of the function get_indexes_where_array_exceeds_threshold
########################################################################

def get_detection_difference(Ax, Ay, Bx, By):
    # Ax, Ay: locations of reference points on plot 2D space (e.g., generated maxima points)
    # Bx, By: locations of points to be checked for accuract against the reference points
    deltas = []
    distances = []
    
    for bx, by in zip(Bx, By):
        min_distance = 100000.0
        delta_x_min = 100000.0
        delta_y_min = 100000.0
        
        for ax, ay in zip(Ax, Ay): 
            delta_x = np.abs(bx-ax)
            delta_y = np.abs(by-ay)
            dist = math.hypot( delta_x, delta_y)
            if dist < min_distance:
                min_distance = dist
                delta_x_min = delta_x
                delta_y_min = delta_y
            
        deltas.append((delta_x_min, delta_y_min))
        distances.append(min_distance)
        
    return deltas, distances
########################################################################

def convert_index_to_location(input_x,input_y, indexes_of_local_jzmax):
    x_indices=[index[0] for index in indexes_of_local_jzmax]
    y_indices=[index[1] for index in indexes_of_local_jzmax]    

    maped_x = input_x[x_indices]
    maped_y = input_y[y_indices]
    
    return maped_x, maped_y
########################################################################

def set_ploting_font(font_size):
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : font_size}
    plt.rc('font', **font)
    

def plot_locations_of_cs_points(x,y,jz,J_th,indexes_of_points_of_all_cs):
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ion()
    #plt.subplot(2,1,1)
    #plt.pcolor(x,y,np.abs(jz)-J_th,cmap='bwr',vmin=-0.5,vmax=0.5)
    #plt.pcolor(x,y, jz, cmap='bwr', vmin=np.min(jz),vmax=np.max(jz))
    #plt.colorbar()
    #plt.title(r'(color) $J_z$, (.) Identified points of current sheets')
    #plt.subplot(2,1,2)
    #plt.pcolor(x,y,np.abs(jz)-J_th,cmap='bwr',vmin=-0.5,vmax=0.5)
    plt.pcolor(x,y, jz, cmap='bwr', vmin=-1,vmax=1)
    #plt.contourf(x,y,jz,20,cmap='bwr',vmin=-0.5,vmax=0.5)
    plt.colorbar()
    #plt.contour(x,y,jz,[0.42*np.max(jz)])
    for indexes_of_points_of_a_cs in indexes_of_points_of_all_cs:
        x_indices=[index[0] for index in indexes_of_points_of_a_cs]
        y_indices=[index[1] for index in indexes_of_points_of_a_cs]

        plt.plot(y[y_indices],x[x_indices],'ok',Markersize=6,alpha=0.1)

    plt.xlabel('$x/d_i$',fontsize=17)
    plt.ylabel('$y/d_i$',fontsize=17)
    plt.title('$J_z/n_0ev_{A_i}$ (color),Detected current sheets',fontsize=17)
    plt.show()

# End of function plot_indexes_of_cs_points
########################################################################

def plot_locations_of_generated_local_jzmax(x,y,jz,J_th, locations_of_local_jzmax):
    x_indices=[index[0] for index in locations_of_local_jzmax]
    y_indices=[index[1] for index in locations_of_local_jzmax]

    #partial_deriv_of_jzmagnitude_in_x, partial_deriv_of_jzmagnitude_in_y=\
    #    np.gradient(np.abs(jz),x[2]-x[1],y[2]-y[1])
    #magnitude_of_gradient_of_jzmagnitude=\
    #    np.sqrt(partial_deriv_of_jzmagnitude_in_x**2+partial_deriv_of_jzmagnitude_in_y**2)
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.figure()
    plt.ion()
    #plt.pcolor(x,y,np.abs(jz)-J_th,cmap='bwr',vmin=-3*J_th,vmax=3*J_th)
    plt.pcolor(x,y, jz, cmap='bwr',vmin=np.min(jz),vmax=np.max(jz))
    #contour_values=20#np.linspace(0,1.1,25)
    #plt.contourf(x,y,np.abs(jz)-J_th,contour_values,color='k',cmap='bwr',\
    #             vmin=-3*J_th,vmax=3*J_th)
    plt.colorbar()
    plt.plot(y_indices, x_indices,'xk',Markersize=6)#,\
    #plt.plot(y[y_indices],x[x_indices],'xk',Markersize=6)#,\    
    #         label='identified point of local maxima')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.title(r'(color) $J_z$, Generated points of local maxima')
    #plt.legend()
    #plt.figure()
    #plt.ion()
    #plt.pcolor(x,y,magnitude_of_gradient_of_jzmagnitude,cmap='afmhot_r',vmin=0,vmax=0.25)
    #plt.colorbar()
    #plt.plot(y[y_indices],x[x_indices],'xk',Markersize=6)
    plt.show()
########################################################################

    
def plot_locations_of_local_jzmax(x,y,jz,J_th,indexes_of_local_jzmax):
    x_indices=[index[0] for index in indexes_of_local_jzmax]
    y_indices=[index[1] for index in indexes_of_local_jzmax]

    #partial_deriv_of_jzmagnitude_in_x, partial_deriv_of_jzmagnitude_in_y=\
    #    np.gradient(np.abs(jz),x[2]-x[1],y[2]-y[1])
    #magnitude_of_gradient_of_jzmagnitude=\
    #    np.sqrt(partial_deriv_of_jzmagnitude_in_x**2+partial_deriv_of_jzmagnitude_in_y**2)
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.figure()
    plt.ion()
    #plt.pcolor(x,y,np.abs(jz)-J_th,cmap='bwr',vmin=-3*J_th,vmax=3*J_th)    
    #plt.pcolor(x,y, jz, cmap='bwr',vmin=np.min(jz),vmax=np.max(jz))
    plt.pcolor(jz, cmap='bwr',vmin=-1,vmax=1)
    #plt.pcolor(x,y, np.abs(jz), cmap='bwr',vmin=-10,vmax=10)
    
    #contour_values=20#np.linspace(0,1.1,25)
    #plt.contourf(x,y,np.abs(jz)-J_th,contour_values,color='k',cmap='bwr',\
    #             vmin=-3*J_th,vmax=3*J_th)
    plt.colorbar()
    #plt.plot(y[y_indices],x[x_indices],'xk',Markersize=6)#,\
    plt.plot(y_indices,x_indices,'xk',Markersize=6)
    #         label='identified point of local maxima')
    plt.xlabel('$x/d_i$',fontsize=17)
    plt.ylabel('$y/d_i$',fontsize=17)
    #plt.xlim([0,50])
    #plt.ylim([-128,-50])
    #plt.title(r'(color) $|J_z|-J_{th}$, (x) identified points of local maxima')    
    plt.title('$J_z/n_0ev_{A_i}$ (color),Identified points of maxima',fontsize=17)    
    plt.show()
    #plt.savefig('maxima_detection_average_data_n25_Jth0.jpg')
# End of function plot_indexes_of_local_jzmax    
########################################################################


def plot_detection_and_generated_differences(deltas, distances, delta_range):    
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}
    plt.rc('font', **font)
    
    plt.plot(deltas)
    plt.ylim((-10,delta_range))
    plt.xlabel('Detected maxima points')
    plt.ylabel('Distance from reference maxima')
    plt.title('Detected vs. Generated')
    plt.legend(['on X axis', 'on Y axis'])
    plt.grid()    
    plt.show()
    
    X = [delta[0] for delta in deltas]
    Y = [delta[1] for delta in deltas]
    
#    plt.plot(X)
#    plt.ylim((-10,delta_range))
#    plt.xlabel('Detected maxima points')
#    plt.ylabel('Distance from reference maxima on X axis')
#    plt.title('Detected vs. Generated')
#    plt.legend(['on X axis'])
#    plt.grid()    
#    plt.show()
#    
#    plt.plot(Y)
#    plt.ylim((-10,delta_range))
#    plt.xlabel('Detected maxima points')
#    plt.ylabel('Distance from reference maxima on Y axis')
#    plt.title('Detected vs. Generated')
#    plt.legend(['on Y axis'])
#    plt.grid()
#    plt.show()
    
    scatter_range = np.max([np.max(X), np.max(Y)])
    plt.scatter(X, Y)
    plt.xlim((-1, scatter_range+1))
    plt.ylim((-1, scatter_range+1))
    plt.xlabel('Distance from reference maxima on X axis')
    plt.ylabel('Distance of detected from reference maxima on Y axis')
    plt.title('Detected vs. Generated')
    plt.legend(['detected maxima point'])
    plt.grid()
    plt.show()
    
    plt.hist(X)
    plt.xlabel('Distance of detected from reference maxima on X axis')
    plt.ylabel('Number of maxima points')
    plt.title('Histogram')
    plt.show()

    plt.hist(Y)
    plt.xlabel('Distance of detected from reference maxima on Y axis')
    plt.ylabel('Number of maxima points')
    plt.title('Histogram')
    plt.show()

    plt.hist(distances)
    plt.xlabel('Distance from reference maxima')
    plt.ylabel('Number of maxima points')
    plt.title('Histogram')
    plt.show()
    
########################################################################


def plot_maxima_point_one_dimension(values,y_axix_lable,title):
    values = np.sort(values)
    size = len(values)
    plt.plot(np.arange(1,size+1,1), values, marker="x")
    plt.xlabel(' maxima point index')
    plt.ylabel(y_axix_lable)
    plt.title(title)
    plt.show()
########################################################################
    
def findLocalMaxima_at_selected_indexes(array, selected_indexes, \
                                        number_of_points_upto_local_boundary):
    # This function checks if the values of "array" at selected indexes provided in
    # "selected_indexes" are local maxima in a surrounding box whose each boundary is away from
    # the candidate point by "number_of_points_upto_local_boundary" number of points.
    # It returns values and indexes of the so found local maxima.
    
    values_of_local_maxima = np.zeros_like(array)
    indexes_of_local_maxima = []
    value_of_pjz_peak=[]
    n =  number_of_points_upto_local_boundary
    
    for index in selected_indexes:
        i = index[0]
        j = index[1]
        #ignoring 'n' points close to the global boundary.   
        if i>n-1 and j>n-1 and i<array.shape[0]-n and j<array.shape[1]-n:
            # point being considered is a "candidate" point
            value_at_candidate_point = array[i, j] 
            
            # get the values at all points in the box surrounding the candidate point
            values_at_all_points_in_surrounding_box = array[i-n:i+n+1, j-n:j+n+1].copy()

            # exclude central candidate point from being considered when finding maximum value in
            # the surrounding box by assigning it the minimum possible value  
            values_at_all_points_in_surrounding_box[n,n] = -np.inf

            # Find maximum value at the surrounding points.
            maximum_value_at_surrounding_points = \
                np.max(values_at_all_points_in_surrounding_box)

            if (value_at_candidate_point > maximum_value_at_surrounding_points):
                values_of_local_maxima[i,j] = value_at_candidate_point
                value_of_pjz_peak.append(value_at_candidate_point)
                indexes_of_local_maxima.append(index)
                
    return values_of_local_maxima, indexes_of_local_maxima,value_of_pjz_peak
#### End of the function findLocalMaxima_at_selected_indexes


def check_adjacent_points_for_minimum_current_density \
            (current_density, flags_to_avoid_rechecking_of_indexes, i, j, \
             minimum_current_density_in_cs, indexes_of_points_of_a_cs):
    # This function checks the point (i,j), if not checked already,  for the condition 
    # that current density "current_density" is larger than a minimum current density
    # "minimum_current_density_in_cs" which is usually taken as a fraction of the local
    # maximum value. If the condition is satisfied, the function calls itself to check 
    # recursively the condition at the adjacent points (i-1,j), (i+1,j), (i,j-1) and (i,j+1).
    # On the first call to this function from the function "detect_current_sheets", the
    # condition is satisfied automatically as the point (i,j) correspond to the local
    # maxima in current density. On the recursive calls from itself, the condition is
    # checked on other points where it may or may not be satisfied. It adds the indexes of
    # points, where the condition is satisfied, to a list of indexes of points belonging to
    # the current sheet and continues checking the condition in the neighbourhood of the newly
    # found points by recursive calls to itself. The process continues until no point satisfying
    # the condition is found.

    if i>current_density.shape[0]-1 or i<0 or j>current_density.shape[1]-1 or j<0:
        return

    # Check the condition only if the point (i,j) has not been checked before
    if flags_to_avoid_rechecking_of_indexes[i, j] == 0:
        # Assign the value 1 to the flag to mark this point as 'checked' to prevent
        # incorrect repetition
        flags_to_avoid_rechecking_of_indexes[i, j] = 1 

        # check the condition
        if current_density[i,j] > minimum_current_density_in_cs:
            # add to the list as a current sheet point
            indexes_of_points_of_a_cs.append((i, j))
            
            # continue checking the condition at four adjacent points by recursive calls
            check_adjacent_points_for_minimum_current_density \
                (current_density, flags_to_avoid_rechecking_of_indexes, i-1, j, \
                 minimum_current_density_in_cs, indexes_of_points_of_a_cs)
            check_adjacent_points_for_minimum_current_density \
                (current_density, flags_to_avoid_rechecking_of_indexes, i, j-1, \
                 minimum_current_density_in_cs, indexes_of_points_of_a_cs) 
            check_adjacent_points_for_minimum_current_density \
                (current_density, flags_to_avoid_rechecking_of_indexes, i, j+1, \
                 minimum_current_density_in_cs, indexes_of_points_of_a_cs) 
            check_adjacent_points_for_minimum_current_density \
                (current_density, flags_to_avoid_rechecking_of_indexes, i+1, j, \
                 minimum_current_density_in_cs, indexes_of_points_of_a_cs)

# End of function "check_adjacent_points_for_minimum_current_density"
########################################################################


def detect_current_sheets(current_density, indexes_of_local_maxima,ratio_of_jzboundary_to_jzmax):
    # This function finds points belonging to current sheets in the "current_density" data.
    # For each index of the local maxima provided in "indexes_of_local_maxima", it stores
    # in a list "indexes_of_points_of_a_cs" the indexes of all the points belonging to the
    # current sheet corresponding to the local maxima. And then each such set of the current
    # sheet points is appended to another list "indexes_of_points_of_all_cs" which is returned
    # to the calling program.

    # flag has zero value for unchecked points and will be assigned 1 for each checked point
    flags_to_avoid_rechecking_of_indexes = np.zeros_like(current_density)

    # A list to store indexes of the points belonging to current sheets found in the
    # "current_density" data. Each item of this list is another list containing indexes
    # of the points of an individual current sheet. 
    indexes_of_points_of_all_cs = []
    indexes_of_valid_local_maxima = []
    
    for index in indexes_of_local_maxima:
        # A list to store indexes of the points of an individual current sheet 
        indexes_of_points_of_a_cs = []

        i = index[0]
        j = index[1]

        # minimum current density to define boundaries of the current sheets
        minimum_current_density_in_cs = ratio_of_jzboundary_to_jzmax*current_density[i, j]

        # Function call to check if the points adjacent to the local maxima with index (i,j)
        # satisfy the condition that  current density be larger than the minimum current density.
        # On return, "indexes_of_points_of_a_cs" contains indexes of points of an individual
        # current sheet and "flags_to_avoid_rechecking_of_indexes" has the values 0 and 1 for the
        # unchecked and checked points, respectively. 
        check_adjacent_points_for_minimum_current_density \
            (current_density, flags_to_avoid_rechecking_of_indexes, i, j, \
             minimum_current_density_in_cs,indexes_of_points_of_a_cs)

        # Add the list of indexes of points of an individual current sheet as an item to
        # the list "indexes_of_points_of_all_cs" only if the list for individual current
        # sheet is non-empty. 
        #print(index,len(indexes_of_points_of_a_cs))
        if len(indexes_of_points_of_a_cs) > 0: 
            indexes_of_points_of_all_cs.append(indexes_of_points_of_a_cs)
            indexes_of_valid_local_maxima.append(index)

    return indexes_of_points_of_all_cs, indexes_of_valid_local_maxima
####### End of the function detect_current_sheets
########################################################################
####################################################################
def plot_generated_jz(x,y,jz):
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.figure()
    plt.ion()
    plt.pcolor(x,y, jz, cmap='bwr',vmin=np.min(jz),vmax=np.max(jz))
    plt.colorbar()
    plt.xlabel('y')
    plt.ylabel('x')
    plt.title(r'$J_z$, Generated Jz')
    #plt.legend()
    #plt.figure()
    #plt.ion()
    #plt.pcolor(x,y,magnitude_of_gradient_of_jzmagnitude,cmap='afmhot_r',vmin=0,vmax=0.25)
    #plt.colorbar()
    #plt.plot(y[y_indices],x[x_indices],'xk',Markersize=6)
    plt.show()
