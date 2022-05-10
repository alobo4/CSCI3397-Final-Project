import numpy as np
from imageio import imread
import cv2
from skimage.measure import label

def density_of_nuclei(nuclei_image,microtubule_image,list_of_radii,pixel_threshold):
    #nuclei_image: has to be upload and read into function
    #microtubule_image: has to be upload and read into function
    #list_of_radii: radii in list for for which density of microtubules is calculated
    #pixel_threshold: pixel lower-bound threshold value for what is considered a microtubule
    
    
    #upload and read in image as png
    n_image = imread(nuclei_image)
    #plt.subplot(221)
    #plt.imshow(test_image, cmap='gray', vmin=0, vmax=255) #prints out input image
    
    #Apply mask to find nuclei
    kernel = 55
    n_image_mask = cv2.medianBlur(n_image,kernel)
    nuclei_threshold = 6
    bin_n_image = n_image_mask>nuclei_threshold
    seg_n_image = label(bin_n_image[:,:,1])
    #plt.subplot(224)
    #plt.imshow(label2rgb(seg_inst, bg_label=0)) #prints out labelled nuclei
    
    #Find Centers of Nuclei
    centers_img = np.zeros([bin_n_image.shape[0],bin_n_image.shape[1]])
    x_centers = []
    y_centers = []
    for i in range(1,seg_inst.max()+1):
        coords = []
        for j in range(bin_image.shape[0]):
            for k in range(bin_image.shape[1]):
                if seg_inst[j,k] == i:
                    coords.append([j,k])
        xsum = 0
        ysum = 0
        for j in coords:
            xsum += j[0]
            ysum += j[1]
        x_centers.append(round(xsum/len(coords)))
        y_centers.append(round(ysum/len(coords)))
    #plt.subplot(222)
    #plt.imshow(label2rgb(seg_inst, bg_label=0))
    #plt.scatter(y_centers,x_centers) #prints image with labelled centers
    #Looping through radii
    m_image = imread(microtubule_image)
    for nucleus in range(len(y_centers)): #loop through num of nuclei found in image
        for radius in list_of_radii: #loop through radii (this can change to be auto)
            square = m_image[x_centers[nucleus]-radius:x_centers[nucleus]+radius,y_centers[nucleus]-radius:y_centers[nucleus]+radius]
            
            #Convert image to polar
            square2 = square.astype(np.float32)
            value = np.sqrt(((square2.shape[0]/2.0)**2.0)+((square2.shape[1]/2.0)**2.0))
            polar_image2 = cv2.linearPolar(square2,(square2.shape[0]/2, square2.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
            polar_image2 = polar_image2.astype(np.uint8)
            
            #Calculate Density
            threshold = pixel_threshold
            microtubules = 0
            total_pixels = polar_image2.shape[0]**2
            for x in range(polar_image2.shape[0]):
                for y in range(polar_image2.shape[1]):
                    if polar_image2[x,y,0] > threshold:
                        microtubules += 1
            density = microtubules/total_pixels
            print('density of microtubules at nuclei',nucleus+1,'with radius',radius,'pixels is',density)
        print('-----------------------------')
        
