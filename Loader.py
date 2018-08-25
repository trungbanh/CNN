import cv2 
import numpy as np 

def load_data () :
	image = cv2.imread("../hinh/digits.png")
	image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	data = np.array(image)
	data = data.reshape(5000,-1)
	label = range(10)
	label = np.repeat(label,500)[:,np.newaxis]