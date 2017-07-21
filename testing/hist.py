import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
def compute_hist(img=[],prj=0,h=0,w=0) :
	x=0
	y=0
	freq=[]
	if(prj==0 ) :
		x=h
		y=w
	else :
		x=w
		y=h
	freq = np.zeros((x))
	for x1 in range (x) :

			for y1 in range(y) :
				if(prj==0) :
					if(img[x1][y1] ==255) :
						freq[x1]+=1
				elif(prj==2) :
					if(img[y1][x1] ==255) :
						freq[x1]+=1
				else :
					if(img[y1][x1] ==0) :
						freq[x1]+=1



	if(prj==0) :
			hist = np.zeros((h,w))
			for x1 in range (x) :
					for y1 in range( int( round(freq[x1]))) :
								hist[x1][y1]=255
	else :
		hist = np.ones((h,w))
		for x1 in range (x) :
					y1=h-1
					k=0
					while(k<= round(freq[x1])) :
						hist[y1][x1]=0
						y1-=1
						k+=1

	if(prj==2)  :
		return freq
	else :
		return freq,hist


def search_last(freq, start,h) :
	i=start
	while(freq[i] == h) :
		if(i== len(freq)-1 ) :
			break
		else :
			i+=1
	return i

def search_last1(freq, start,h) :
	i=start
	while(freq[i] != h) :
		if(i==0 ) :
			break
		else :
			i-=1
	return i


def segment2 (name,mean, img, freq) :
	stack_start = []
	stack_end= []
	i=0
	x1=-1
	x2=-1
	h,w = img.shape[:2]
	#print(freq)
	while(i<len(freq)) :
		if(freq[i]!=h) :
			if(x1==-1) :
				x1=i
			else :

				x2=i+1
		else :
			if(i==0) :
				x1=i
				x2=i

			if(x1!=-1 and x2!=-1) :
				stack_start.append(x1)
				stack_end.append(x2)
			x1=-1
			x2=-1

		i+=1


	i=1
	while(i<len(stack_end)) :
		cv2.rectangle(img,(stack_end[i-1],0),(stack_start[i],h),(0,255,0),2)
		if(i==len(stack_end)-1) :
			Li = search_last(freq, stack_end[i],h)
			cv2.rectangle(img,(stack_end[i],0),(Li,h),(0,255,0),2)

		i+=1

	#cv2.imshow(name,img)
	#cv2.imshow(1000000)

	return stack_start,stack_end


def threshold(freq , mean, h, freq1) :
	i=0
	while(i<len(freq)) :
		if(freq1[i]<mean) :
			freq[i]=h
		i+=1
	return freq
def segment3 (name,mean, img, freq,w_or) :
	stack_start = []
	stack_end= []
	i=0
	x1=-1
	x2=-1
	h,w = img.shape[:2]
	w = w_or
	#print(freq)
	while(i<len(freq)) :
		if(freq[i]==0) :
			if(x1==-1) :
				x1=i
			else :

				x2=i+1
		else :
			if(i==0) :
				x1=i
				x2=i

			if(x1!=-1 and x2!=-1) :
				stack_start.append(x1)
				stack_end.append(x2)
			x1=-1
			x2=-1

		i+=1


	i=1
	while(i<len(stack_end)) :

		cv2.rectangle(img,(0,stack_end[i-1]),(w,stack_start[i]),(0,255,0),2)
		if(i==len(stack_end)-1) :
			Li = search_last(freq, stack_end[i],0)
			cv2.rectangle(img,(0,stack_end[i]),(w,Li),(0,255,0),2)


		i+=1



	return stack_start,stack_end


def segment(mean, img, freq) :
	stack = []
	h,w = img.shape[:2]
	min_val = 0
	max_val =0
	stack.append(0)
	i=1
	flag =0
	while(i<len(freq)) :
		if(freq[i]==h) :
				dist = min_val-i
				if(i-stack[len(stack)-1] >1) :
					print(i)
					stack.append(i)
				else :
					stack[len(stack)-1]=i


				cv2.rectangle(img,(min_val,0),(min_val+dist,h),(0,255,0),2)
				min_val = i

		i+=1
	print(len(stack))
	i=0
	while(i<len(stack)-1) :
		cv2.rectangle(img,(stack[i],0),(stack[i+1],h),(0,255,0),2)
		i+=1
	return img

def compute_mean(prj) :
	mean =0
	i=0
	while(i<len(prj)) :
		mean +=prj[i]
		i+=1

	mean = mean/len(prj)
	return mean
def pre(pth, j) :
	# Load the image
	kernel = np.ones((3,3), np.uint8)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = cv2.imread(pth)
	h_or = img.shape[0]
	w_or = img.shape[1]

	w_or*=5
	h_or*=5
	img = cv2.resize(img,(w_or,h_or))
	img1 = cv2.resize(img,(w_or,h_or))

	# convert to grayscale


	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


	gray = clahe.apply(gray)
	gray = cv2.bilateralFilter(gray,10,97,75*2)
	#gray = cv2.bilateralFilter(gray,10,97,75)

	gray=cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel )

	#gray = cv2.bilateralFilter(gray,10,97,75)


	# smooth the image to avoid noises



	# Apply adaptive threshold
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)




	#thresh = cv2.adaptiveThreshold(thresh,255,1,1,11,2)
	thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

	# apply some dilation and erosion to join the gaps


	thresh = cv2.erode(thresh,None,iterations = 1)
	#thresh = cv2.dilate(thresh,None,iterations =3)
	# Find the contours
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	# For each contour, find the bounding rectangle and draw it
	for cnt in contours:
	    x,y,w,h = cv2.boundingRect(cnt)
	    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
	    #cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)

	x_proj_ori,histx =compute_hist(img=thresh,prj=1,h=h_or,w=w_or)
	y_proj_ori,histy =compute_hist(img=thresh,prj=0,h=h_or,w=w_or)

	x_proj_ori1 =compute_hist(img=thresh,prj=2,h=h_or,w=w_or)

	mean = compute_mean(x_proj_ori1)
	#print(compute_mean(x_proj_ori))
	#print(compute_mean(y_proj_ori))
	# pos = np.arange(len(x_proj))
	# width = 100    # gives histogram aspect to the bar diagram

	# ax = plt.axes()
	# ax.set_xticks(pos + (width / 2))
	# #ax.set_xticklabels(x_proj)
	# plt.bar(pos, x_proj, width, color='b')
	# plt.show()


	# pos = np.arange(len(y_proj))
	# width = 1    # gives histogram aspect to the bar diagram

	# ax = plt.axes()
	# ax.set_xticks(pos + (width / 2))
	# #ax.set_xticklabels(y_proj)
	# plt.bar(pos, y_proj, width, color='b')
	# plt.show()

	# Finally show the image
	#x_proj_ori = 	threshold(x_proj_ori , mean, h_or, x_proj_ori1)
	img2 = img1
	stack_start1,stack_end1 = segment2("ori",mean, img1, x_proj_ori)
	i=1

	while(i<len(stack_end1)) :
		kernel = np.ones((4,4), np.uint8)
		roi=[]
		roi1=[]

		roi=img[0:h_or-1,stack_end1[i-1]:stack_start1[i]]
		h_roi = int(round(roi.shape[0]))
		w_roi = int(round(roi.shape[1]))

		# roi = cv2.morphologyEx(roi,cv2.MORPH_OPEN,kernel )
		# h,w = roi.shape[:2]
		# x_proj,histx =compute_hist(img=roi,prj=1,h=h,w=w)
		# y_proj,histy =compute_hist(img=roi,prj=0,h=h,w=w)
		# r = img2[0:h_or-1,stack_end1[i-1]:stack_start1[i]]
		# segment2(str(i),mean, r, x_proj)
		# if(i==len(stack_end1)-1) :
		# 	Li = search_last(x_proj_ori, stack_end1[i],h_or)

		# 	roi2=thresh[0:h_or-1,stack_end1[i]:Li]
		# 	roi2 = cv2.morphologyEx(roi2,cv2.MORPH_OPEN,kernel )
		# 	h,w = roi2.shape[:2]
		# 	x_proj,histx =compute_hist(img=roi2,prj=1,h=h,w=w)
		# 	y_proj,histy =compute_hist(img=roi2,prj=0,h=h,w=w)

		# 	r = img2[0:h_or-1,stack_end1[i]:Li]
		# 	segment2(str(i),mean, r, x_proj)

		i+=1


	#cv2.imshow('segments2',img2)
	#img11 = segment(mean, img1, x_proj)

	cv2.imshow('histx',histx)
	cv2.imshow('histy',histy)
	cv2.imshow('segments',img1)
	cv2.imshow('img',img)
	cv2.imshow('res',thresh_color)
	cv2.waitKey(1000000)
	cv2.destroyAllWindows()
	#sys.exit()




def horizontal(pth) :
	kernel = np.ones((3,3), np.uint8)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = cv2.imread(pth)
	h_or = img.shape[0]
	w_or = img.shape[1]

	#w_or*=5
	#h_or*=5
	#img = cv2.resize(img,(w_or,h_or))
	#img1 = cv2.resize(img,(w_or,h_or))
	# convert to grayscale
	img1 = img
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	x1=0
	for (x,y,w,h) in faces:
		 #cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
		 x1=x
	if(x1!=0) :
		gray=gray[0:h_or,0:x1+1]
		w_or = x1+1



	#gray = clahe.apply(gray)
	gray = cv2.bilateralFilter(gray,10,97,75)
	gray = cv2.bilateralFilter(gray,10,97,75)
	#gray = cv2.bilateralFilter(gray,10,97,75)
	#gray = cv2.bilateralFilter(gray,10,97,75)

	#gray=cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel )

	#gray = cv2.bilateralFilter(gray,10,97,75)
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
	#thresh = cv2.erode(thresh,None,iterations = 1)
	# smooth the image to avoid noises
	# Apply adaptive threshold
	#thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)




	i=0
	while(i<h_or) :
		c = 1
		j=0
		while(j<w_or) :

			if (thresh[i][j]) == 0:
				if (j-c) <= 3:
					thresh[i][c:j] = 0

				c = j
			j+=1

		if (w_or - c) <= 3:
			thresh[i][c:w_or] = 0
		i+=1






	x_proj_ori,histx =compute_hist(img=thresh,prj=1,h=h_or,w=w_or)
	if(x1!=0) :
		i=search_last1(x_proj_ori, w_or-1,h_or)
		thresh=thresh[0:h_or,0:i+1]
		w_or=i+1

	y_proj_ori,histy =compute_hist(img=thresh,prj=0,h=h_or,w=w_or)

	mean = compute_mean(y_proj_ori)

	stack_start1,stack_end1 = segment3("ori",mean, img1, y_proj_ori, w_or)
	cv2.imshow('gray',gray)
	cv2.imshow('histy',histy)
	cv2.imshow('histx',histx)
	cv2.imshow('thresh',thresh)
	cv2.imshow('segments',img1)
	cv2.waitKey(1000000)


#horizontal("ktp.jpg")
#re("./detect.png",0)
pre("./sof_ori.png", 1)
