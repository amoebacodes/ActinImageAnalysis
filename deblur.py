#image crap 
import cv2
#import argparse
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
channel = 'ch2'
fm_dict = {}
pathlist = paths.list_images(channel)
i = 0
for imagePath in pathlist:
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
		i+=1
		image = cv2.imread(imagePath)
		try:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			fm =  cv2.Laplacian(gray, cv2.CV_64F).var()
			fm_mean = cv2.Laplacian(gray, cv2.CV_64F).mean()

			fm_dict[imagePath] = [fm,fm_mean]
			
		except:
			print 'bad image'
	
fm_ser = pd.DataFrame(fm_dict).T
fm_ser.to_csv('fm_{}.csv'.format(channel))

#plt.hist(vals,bins = 100)
#plt.show()
        

	



