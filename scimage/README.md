Algorithms for data analysis and image processing (Software for automation of feature extraction).

This project performs image processing and data analysis of the test data and data obtained from pic hybrid simulation of plasma.

The simulation data can be visualized as pictures.

Start by running the file 'newanalysis_for_characterization.py'(the core of the algorithm). After that, you can run other .py files for further analysis. You can also run 'test_image_processing.py' to compare your own algorithms with Python image processing functions.

There are 8 files that contain each stage of identification and characterization of current sheets in the collision plasma turbulence as follow:

1- The file "file_functions.py" belongs to the managing and sorting data by creating folders.

2- The "functions_of_thickness.py" contains the codes for the characterization part of the algorithm, e.g. functions for the measurement of thickness and length of each current sheet.

3- The "general_functions.py" contains all functions for the identification part of the algorithm. Two main functions in this file are the detection of peak current density in surrounded by the local region and detection of point belong to each current sheets which form each individual current sheets in the simulation.

4- "test_image_processing.py" is for the comparison between the algorithm part for detection of local peak current density and Python skimage.

5- "thermal and electron velocity calculatio.py ", "uezuizplot.py" is for the data analysis and statistical analysis of thickness, length, and aspect ratio (Length/Half_thickness) of each current sheets. The results are presented in the publication in the Journal of Physics of Plasmas (https://doi.org/10.1063/5.0040692).
