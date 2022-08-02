#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 22:02:36 2020

@author: amir
"""

from skimage.feature import peak_local_max


# Use skimage's functions to detect local maximas ---------------
boundary_for_skimage = number_of_points_upto_local_boundary
# boundary_for_skimage = 20
peak_indexes = peak_local_max(image = jz_magnitude, min_distance = boundary_for_skimage, threshold_abs = J_th)

# Compare the results with our own code -------------
# Both look exactly the same!!
plt.figure()
plt.imshow(jz_magnitude)
array_of_our_indexes = np.asarray(indexes_of_local_jzmax)
plt.plot(array_of_our_indexes[:, 1], array_of_our_indexes[:, 0], 'X', color='w', label='our algorithm (boundary '+str(number_of_points_upto_local_boundary)+')')
plt.plot(peak_indexes[:, 1], peak_indexes[:, 0], '.', color='r', label='skimage (boundary '+str(boundary_for_skimage)+')')
plt.legend()
plt.show()

#print(len(indexes_of_local_jzmax), len(peak_indexes))

