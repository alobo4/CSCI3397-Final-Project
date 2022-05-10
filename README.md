# CSCI3397-Final-Project
Multi-nucleated Muscle Cell Image Analysis

This project is using bio-medical image anaylsis techniques to help determine features of multi-nucleated muscle cells. All images were provided  in the /Cell_images folder from Prof. Eric Folker's Lab at Boston College. There are two main parts of this project with functions included:

1. Density of Microtubules at ranging radii from nuclei in image
   -Python function which will determine the density of microtubules at various radii away from nucleus center.  
   -Code Contribution(Alec Lobanov)
   
   Packages Required:
   Numpy
   Imageio
   cv2
   skimage
   
   density_of_nuclei(nuclei_image,microtubule_image,list_of_radii,pixel_threshold)
   
   nuclei_image: has to be upload as .png and read into function
   microtubule_image: has to be upload as .png and read into function
   list_of_radii: radii in list for for which density of microtubules is calculated ex. [10 20 30 40 50]
   pixel_threshold: integer pixel lower-bound threshold value for what is considered a microtubule

2. Deep-Learning Model to remove noise from input images
   -Current updated model that is train on image data from /Cell_images folder to recognize noise in image which can be removed in later stage
   -Code Contribution(Alec Lobanov)
   
   Packages Required:
   Numpy
   Torch
   cv2
   skimage
   Matplotlib
   Imageio
   Sklearn
   Scipy
   
   
