# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 00:57:00 2020

@author: amirc
"""

"""
Created on Mon May  4 02:45:44 2020

@author: amirc
"""



import numpy as np 
import math
import matplotlib.pyplot as plt
import sys
from functions_of_thickness import*
from general_functions import *
from file_functions import*
import pickle
sys.setrecursionlimit(10000)

#############################################################################
real_data=False
average=False
cs_condition=False

show_plots= False
save_plots= False
save_plots_for_characterization=False
############################################################################
t_of_input_file = ''
resolution = ''
if real_data :
    if average:
        data1=np.load('jz_t46point0.npz')
        data4=np.load('jz_t48point0.npz')
        data2=np.load('jz_t50point0.npz')
        data3=np.load('jz_t52point0.npz')
        data5=np.load('jz_t54point0.npz')
       
        
        t_of_input_file = 'avg.'
        resolution = 'avg'
        jz1=data1['jz']
        jz2=data2['jz']
        jz3=data3['jz']
        jz4=data4['jz']
        jz5=data5['jz']
        jz=(jz1+jz2+jz3)/3
        nx=data1['nx']
        ny=data1['ny']
        lx=data1['lx']
        ly=data1['ly']
        x=np.linspace(-lx/2,lx/2,nx)
        y=np.linspace(-ly/2,ly/2,ny)
    else:
        input_file = 'jz_t50point0.npz'        
        t_of_input_file = '40' # IMPORTANT
        resolution = '2048'
        data=np.load(input_file)
        
        jz=data['jz']
        nx=data['nx']
        ny=data['ny']
        lx=data['lx']
        ly=data['ly']
        x=np.linspace(-lx/2,lx/2,nx)
        y=np.linspace(-ly/2,ly/2,ny)
else:
    # First load/calculate jz-data
    x,y,jz, generated_x_positions_of_cs, generated_y_positions_of_cs, \
    thicknesses_of_cs_x1, thicknesses_of_cs_x2, nx, ny, lx, ly = generate_cs_in_turbulence()
    
#######################################################################################
jz_magnitude=np.abs(jz)
#plt.pcolormesh(y,x,jz_magnitude)
#plt.colorbar()
#plt.show()
nx = jz.shape[0]
ny = jz.shape[1]

jz_rms = np.sqrt( np.sum(jz**2)/(nx*ny) )
J_th = 1*jz_rms
ratio_of_uez_to_uizrms=300
ratio_of_jzboundary_to_jzmax=0.5

number_of_points_upto_local_boundary = 25
#Ask question it is important
#uez_max=uiz[x_indices,y_indices]-jz[x_indices,y_indices]/rho[x_indices,y_indices]

#################################################################################
uiz_data=np.load('uiz_t50point0.npz')
rho_data=np.load('rho_t50point0.npz')
uiz=uiz_data['uiz']
rho=rho_data['rho']
uez=uiz-jz/rho

if cs_condition:
    uiz_rms=np.sum(uiz**2)/(nx*ny)
    indexes_where_condition_is_satisfied = get_indexes_where_uez_gt_uizrms(uez,uiz_rms,ratio_of_uez_to_uizrms)        
else:
    # get indices of the points where |jz| > J_th
    indexes_where_condition_is_satisfied = get_indexes_where_array_exceeds_threshold(jz_magnitude, J_th)
local_jzmax, indexes_of_local_jzmax,value_of_pjz_peak = \
    findLocalMaxima_at_selected_indexes(jz_magnitude,indexes_where_condition_is_satisfied, \
                                       number_of_points_upto_local_boundary)
print ("number of detected maximas:" , len(indexes_of_local_jzmax))


# Find points belonging to current sheets
indexes_of_points_of_all_cs, indexes_of_valid_local_maxima = detect_current_sheets(jz_magnitude, \
                                                    indexes_of_local_jzmax, \
                                                    ratio_of_jzboundary_to_jzmax)
print ("number of detected current sheets:" , len(indexes_of_points_of_all_cs))



# Create the folder for output files and results, if not created before
customized_folder_name = get_folder_name(real_data,average,cs_condition,
                       lx, ly, nx, ny,J_th,jz_rms, 
                       ratio_of_jzboundary_to_jzmax,ratio_of_uez_to_uizrms,
                       number_of_points_upto_local_boundary)
create_folder('./output/'+customized_folder_name)
create_folder('./output/'+customized_folder_name+'/sheets_pics_with_length')


# Plot:
if show_plots:
    plot_locations_of_local_jzmax(x,y,jz,J_th,indexes_of_local_jzmax)
    plot_locations_of_cs_points(x,y,jz,J_th,indexes_of_points_of_all_cs)

if save_plots:     
    plt.ioff() # Call this, so that it does not display on screen in console

    plot_locations_of_local_jzmax(x,y,jz,J_th,indexes_of_local_jzmax)
    plt.savefig('./output/'+customized_folder_name+'/locations_of_local_jzmax.png', bbox_inches="tight", dpi=150)
    plt.close()
    
    plot_locations_of_cs_points(x,y,jz,J_th,indexes_of_points_of_all_cs)
    plt.savefig('./output/'+customized_folder_name+'/locations_of_cs_points.png', bbox_inches="tight", dpi=150)
    plt.close()
# --------------------------------------------
sheet_index = 1
halfthickness1=[]
halfthickness2=[]
ave_thicknesses=[]
lengths_pairwise = []
aspect_ratios=[]
min_half_thickness=[]

MIN_FRAME_SIZE = 3

for indexes_of_points_of_a_cs in indexes_of_points_of_all_cs:
    # Build a local CS frame and get data in it
    x_in_cs_frame_global_value, y_in_cs_frame_global_value, jz_in_cs_frame = \
    build_cs_frame(indexes_of_points_of_a_cs,x,y,jz_magnitude)

    j_min=np.max(jz_in_cs_frame)*0.42#ratio_of_jzboundary_to_jzmax
    
    print("current sheet to be characterized: size:" , jz_in_cs_frame.shape)
    
    if jz_in_cs_frame.shape[0] >= MIN_FRAME_SIZE and jz_in_cs_frame.shape[1] >= MIN_FRAME_SIZE:
        half_thickness_plus_side,half_thickness_minus_side=\
                characterize_cs(jz_in_cs_frame, 
                                x_in_cs_frame_global_value, y_in_cs_frame_global_value, 
                                j_min)
        
        average_thickness= (half_thickness_plus_side+half_thickness_minus_side)/2 #Average half thicknesses
        halfthickness1.append(half_thickness_plus_side)
        halfthickness2.append(half_thickness_minus_side)
        ave_thicknesses.append(average_thickness)
        minimum_thickness= min(half_thickness_plus_side,half_thickness_minus_side)
        min_half_thickness.append(minimum_thickness)
        
         
        #print("half_thickness1,half_thickness2")
        #print(half_thickness_plus_side,half_thickness_minus_side)
    
        if np.isnan(half_thickness_plus_side) or np.isnan(half_thickness_minus_side):
            print("!!! No thickness calculated for maxima point", np.max(jz_in_cs_frame))
        #plt.pcolormesh(y_in_cs_frame_global_value,x_in_cs_frame_global_value,jz_in_cs_frame,alpha=0.5)
        #plt.colorbar()
        #plt.show()    
    
        # Find length with the pair-wise comparison method
        length, p1,p2 = find_length_by_pariwise_distance(indexes_of_points_of_a_cs, x, y)
        aspect_ratio = length/average_thickness
        aspect_ratios.append(aspect_ratio)
        lengths_pairwise.append(length)
        # Save current sheet plot with the detected length lines ---------------
        if save_plots_for_characterization:     
            plt.ioff() # Call this, so that it does not display on screen in console
            fig, ax = plt.subplots()
            ax.pcolormesh(y_in_cs_frame_global_value,x_in_cs_frame_global_value,jz_in_cs_frame,alpha=0.8, zorder=1)    
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], linewidth=3, zorder=2 , color='black')
            plt.savefig('./output/'+customized_folder_name+'/sheets_pics_with_length/cs_'+str(sheet_index)+'.png', bbox_inches="tight", dpi=90)
            plt.close()
        # ----------------------------------------------------------------------
        
        #if frame_size > 2:
        #    Vx,Vy = eigenvec(CS_frame, CS_max_point_index, x_axis_CS_frame, y_axis_CS_frame)        
        #    print("Eigen for current sheet "+str(sheet_index)+":", Vx, Vy)
        #else:
        #    print("Frame too small")
    else:
        print("Frame too small for sheet_index=",sheet_index)
        ave_thicknesses.append(np.nan)
        lengths_pairwise.append(np.nan)
        min_half_thickness.append(np.nan)
        halfthickness1.append(np.nan)
        halfthickness2.append(np.nan)
        aspect_ratios.append(np.nan)

    ##################################
    sheet_index += 1
##############################################################################
thickness_gt_0point25=[thickness for thickness in ave_thicknesses if thickness > 0.5]
length_gt_0point25=[length for length in lengths_pairwise if length > 0.5] 
##############################################################################


np.savez('./output/'+customized_folder_name+'/characterization_2048_t60_Jthrjrms', 
         lengths_pairwise=length_gt_0point25, 
         ave_thicknesses=thickness_gt_0point25,
         aspect_ratios=aspect_ratios)


np.savez('./output/'+customized_folder_name+'/lengths_pairwise_2048_t60_Jthrjrms', 
         lengths_pairwise=length_gt_0point25)


np.savez('./output/'+customized_folder_name+'/ave_thicknesses_2048_t60_Jthrjrms', 
         ave_thicknesses=thickness_gt_0point25
)


np.savez('./output/'+customized_folder_name+'/aspect_ratios_2048_t60_Jthrjrms', 
         aspect_ratios=aspect_ratios)


df = pd.DataFrame(data={'t':[t_of_input_file],'resolution':[resolution]})
df.to_csv('./output/'+customized_folder_name+'/settings.csv')


#
#plt.hist(aspect_ratio1)
#plt.xlabel('aspect_ratio')
#plt.ylabel('Distribution')
#plt.show()
#plt.hist(lengths_pairwise)
#plt.xlabel('Length')
#plt.ylabel('Number')
#plt.show()
#plt.plot(np.sort(thicknesses_of_cs_x2),'r+',markersize=10)
#plt.plot(np.sort(halfthickness1),'ob',markerfacecolor='none',markersize=10)
#plt.plot(np.sort(halfthickness2),'ks',markerfacecolor='none',markersize=10)
#plt.pcolor(y,x,np.abs(uez)/uiz_rms,cmap='bwr',vmin=0,vmax=500)
#plt.colorbar()
#plt.xlabel('y')
#plt.ylabel('x')
#plt.savefig('uezDuizrms.png')
##plotting jz masked by value of uez/uiz_rms
#
#jz_masked=np.copy(jz)
#jz_masked[np.abs(uez) < 300*uiz_rms]=0
#plt.pcolormesh(x,y,jz_masked,cmap='bwr',vmin=-0.5,vmax=0.5);plt.colorbar()
#plt.xlabel('y')
#plt.ylabel('x')
#plt.savefig('jz_masked.png')
#############################
# Hsit plot with different bin for aspect reatio, thickness and length
def plot_compare(values1, values2, values3, bins=100, range=None):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111) 
    sns.distplot(values1, hist = False, kde = True,kde_kws = {'linewidth': 3},label='Aspect ratio')
    sns.distplot(values2, hist = False, kde = True,kde_kws = {'linewidth': 3},label='Length')
    sns.distplot(values3, hist = False, kde = True,kde_kws = {'linewidth': 3},label='Thickness')
    plt.legend(prop={'size': 16}, title = '')
    plt.title('Density Plot of characterization part')
    plt.xlabel('')
    plt.ylabel('Density')

    ax.legend(loc='upper right', prop={'size':14})
    plt.show()

#################################
def plot_compare1(values1, values2, values3, bins=100, range=None):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111) 
    sns.distplot(values1, hist=True, kde=True, bins=int(180/5), color = 'darkblue',hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4},label='Aspect ratio')
    sns.distplot(values2, hist=True, kde=True, bins=int(180/5), color = 'darkblue',hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4},label='Length')
    sns.distplot(values3, hist=True, kde=True, bins=int(180/5), color = 'darkblue',hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4},label='Thickness')

    plt.legend(prop={'size': 16}, title = '')
    plt.title('Density Plot of characterization part')
    plt.xlabel('')
    plt.ylabel('Density')

    ax.legend(loc='upper right', prop={'size':14})
    plt.show()

####################################################################   
#plot_compare(aspect_ratio1,lengths_pairwise,thicknesses,range(0,100))
#colors = ['#E69F00', '#56B4E9', '#F0E442']
#names = ['Aspect ratio','Thicknesses','Length']
#plt.hist([aspect_ratio1,lengths_pairwise,thicknesses], normed=True, color = colors, label=names)
#
## Plot formatting
#plt.legend()
#plt.xlabel('Aspact ratio , thickness, length')
#plt.ylabel('number of parameters')
#plt.title('Side-by-Side Histogram Characterization part')
################################################3
#sns.distplot(lengths_pairwise, hist=True, kde=True, bins=int(180/5), color = 'green',hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4},label='Length')
#plt.legend(prop={'size': 16}, title = '')
#plt.title('Density Plot of characterization part')
#plt.xlabel('')
#plt.ylabel('Density')
#plt.legend(loc='upper right', prop={'size':14})
#plt.show()
#sns.distplot(thicknesses, hist=True, kde=True, bins=int(180/5), color = 'red',hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4},label='Thickness')
#plt.legend(prop={'size': 16}, title = '')
#plt.title('Density Plot of characterization part')
#plt.xlabel('')
#plt.ylabel('Density')
#plt.legend(loc='upper right', prop={'size':14})
#plt.show()
#sns.distplot(aspect_ratio1, hist=True, kde=True, bins=int(180/5), color = 'darkblue',hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4},label='Aspect ratio')
#plt.legend(prop={'size': 16}, title = '')
#plt.title('Density Plot of characterization part')
#plt.xlabel('')
#plt.ylabel('Density')
#plt.legend(loc='upper right', prop={'size':14})
#plt.show()
################################################################
#plot of maask from speed of particles in the simulation
#plt.pcolor(y,x,np.abs(uez)/uiz_rms,cmap='bwr',vmin=0,vmax=500)
#plt.colorbar()
#plt.xlabel('$x/d_i$')
#plt.ylabel('$y/d_i$')
#plt.title('$u_{ez}/u_{iz-rms}$ (color)')
#plt.savefig('uezDuizrms.png')
#plt.show()
##plotting jz masked by value of uez/uiz_rms
#
#jz_masked=np.copy(jz)
#jz_masked[np.abs(uez) < 300*uiz_rms]=0
#plt.pcolormesh(x,y,jz_masked,cmap='bwr',vmin=-0.5,vmax=0.5);plt.colorbar()
#plt.title('$Masked J_z:Jz[(|u(ez)|/u(iz))rms<300]=0$')
#plt.xlabel('$x/d_i$')
#plt.ylabel('$y/d_i$')
#plt.savefig('jz_masked.png')
#plt.show()
################################################################
# Extracting thicknesses and length greater than 0.5    
thickness_gt_0point25=[thickness for thickness in ave_thicknesses if thickness > 0.25]
length_gt_0point25=[length for length in lengths_pairwise if length > 0.25]

# filter_by_ave_thickness = np.array(ave_thicknesses) > 0.5
filter_by_ave_thickness = np.where(np.array(ave_thicknesses) > 0.5)[0]

thickness_gt_filtered = np.array(ave_thicknesses)[filter_by_ave_thickness]


####################################################################  
#hist plot of jz_peak for different data and different parameters


##########################################################
if show_plots:
    plt.rcParams['font.size'] = 20
    #plt.rcParams['axes.labelweight'] = 'bold'
    #plt.rcParams['font.weight'] = 'bold'
    plt.scatter(ave_thicknesses,lengths_pairwise)
    plt.legend(prop={'size': 14}, title = '$J_{thr}=j_{rms} $\n$n=25$ \n$\omega_{ci}t=46$',title_fontsize=25)
    plt.xlabel('Thickness/$d_{i}$',fontsize=20)
    plt.ylabel('Length',fontsize=20) 



#########################################################
#with open("Thickness t50 j_{thr} = jrms n=30 .npz", "wb") as fp:   
#    pickle.dump(thickness_gt_0point5, fp)
#    
#with open("lengtht t54j_{thr} = jrms n=30.npz", "wb") as fp:   
#    pickle.dump(length_gt_0point5, fp)
#    
#with open("Aspect ratio t54 j_{thr}=jrms n=30.npz", "wb") as fp:   
#    pickle.dump(aspect_ratio1, fp)
 
#    
###############################################
#with open("Thickness t50 j_{thr} = ueui n=30 .npz" , "wb") as fp:   
#    pickle.dump(thickness_gt_0point5, fp)
#    
#with open("lengtht t50j_{thr} = ueui n=30 .npz", "wb") as fp:   
#    pickle.dump(length_gt_0point5, fp)
#    
#with open("Aspect ratio t50 j_{thr}=ueui n=30 .npz", "wb") as fp:   
#    pickle.dump(aspect_ratio1, fp)    
#######################################################
#with open("Values of jzpeak t54 j_{thr} = jrms n=30 .npz" , "wb") as fp:   
#    pickle.dump(value_of_pjz_peak, fp)
#with open("Values of jzpeak t54 j_{thr} = ueui n=30 .npz" , "wb") as fp:   
#    pickle.dump(value_of_pjz_peak, fp)    
#with open("Values of jzpeak t46j_{thr} = jrms n=20 .npz", "wb") as fp:   
#    pickle.dump(value_of_pjz_peak, fp)
#    
#with open("Values of jzpeak t46 j_{thr}=jrms n=20 .npz", "wb") as fp:   
#    pickle.dump(value_of_pjz_peak, fp)    
