import cv2
import os
import numpy as np
#import test_model as model
from PIL import Image
import matplotlib.pyplot as plt
import shutil

def compute_hist(img=[],prj=0,h=0,w=0) :
	x=0
	y=0
	freq=[]
	if(prj==0) :
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

	return freq,hist

def search_last(freq, start,h) :
	i=start
	while(freq[i] != h) :
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

def crop_field() :
	images=os.listdir('./Scrapped_n_scaled\\')
	for j, im in enumerate (images) :
		pth = "./Scrapped_n_scaled/"+str(im)

		img = cv2.imread(pth)
		h = img.shape[0]
		w = img.shape[1]
		roi = img[60:h-1,0:110]
		path = "./output_field/"+im
		img = cv2.imwrite(path,roi)
		print(im)

def compute_mean(prj) :
	mean =0
	i=0
	while(i<len(prj)) :
		mean +=prj[i]
		i+=1

	mean = mean/len(prj)
	return mean

def segment2 (name,mean, img, freq) :
	stack_start = []
	stack_end= []
	i=0
	x1=-1
	x2=-1
	h,w = img.shape[:2]
	#print(freq)
	while(i<len(freq)) :
		if(freq[i]==h) :
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

	return stack_start,stack_end

def vertical(ori1,ori,img, j,k,hmin, hmax) :
	# Load the image
	kernel = np.ones((3,3), np.uint8)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

	h_or = img.shape[0]
	w_or = img.shape[1]

	w_or*=7
	h_or*=7
	img = cv2.resize(img,(w_or,h_or))
	img1 = cv2.resize(img,(w_or,h_or))

	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = clahe.apply(gray)
	gray = cv2.bilateralFilter(gray,10,97,75)

	gray=cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel )

	# Apply adaptive threshold
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

	# apply some dilation and erosion to join the gaps
	thresh = cv2.erode(thresh,None,iterations = 1)

	x_proj_ori,histx =compute_hist(img=thresh,prj=1,h=h_or,w=w_or)
	y_proj_ori,histy =compute_hist(img=thresh,prj=0,h=h_or,w=w_or)

	mean = compute_mean(x_proj_ori)

	# Finally show the image
	stack_start,stack_end = segment2("ori",mean, img1, x_proj_ori)
	word_queue = []
	i=1

	while(i<len(stack_end)) :
		if os.path.exists("./segment/" + str(j)) == False:
			os.makedirs("./segment/" + str(j))
		path ="./segment/"+str(j)+"/"+str(i)+".png"
		roi=img[0:h_or,stack_end[i-1]:stack_start[i]]
		wmin = int(round(stack_end[i-1]/7))
		wmax = int(round(stack_start[i]/7))
		#roi = ori[hmin:hmax,wmin:wmax]

		h_roi = int(round(roi.shape[0]))
		w_roi = int(round(roi.shape[1]))

		if(h_roi>=1 and w_roi>=1) :
			cv2.imwrite(path,roi)
			cv2.rectangle(ori1,(wmin,hmin),(wmax,hmax),(0,255,0),2)
			#word_queue.append(model.predict(roi))
			#print(model.predict(roi),end='')
		if(i==len(stack_end)-1) :
				path ="./segment/"+str(j)+"/"+str(i+1)+".png"
				Li = search_last(x_proj_ori, stack_end[i],h_or)
				roi=img[0:h_or,stack_end[i]:Li]
				wmin = int(round(stack_end[i]/7))
				wmax = int(round(Li/7))
				#roi = ori[hmin:hmax,wmin:wmax]
				h_roi = int(round(roi.shape[0]))
				w_roi = int(round(roi.shape[1]))

				if(h_roi>=1 and w_roi>=1) :
					cv2.imwrite(path,roi)
					cv2.rectangle(ori1,(wmin,hmin),(wmax,hmax),(0,255,0),2)
					#word_queue.append(model.predict(roi))
					#print(model.predict(roi),end='')
		i+=1
	i=0
	'''while (i<len(word_queue)):
		if(word_queue[i]=="background_spacing") :
			word_queue[i] =""
		print(word_queue[i],end='')
		i+=1

	print()'''

def segment3 (name,mean, img, freq,w_or,k,ori) :
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
		#cv2.rectangle(img,(0,stack_end[i-1]-1),(w-3,stack_start[i]+1),(0,255,0),2)
		roi =  img[stack_end[i-1]-1:stack_start[i]+1, 0:w]
		h_roi = roi.shape[0]
		w_roi = roi.shape[1]

		if(h_roi>0 and w_roi>0) :
			hmin = stack_end[i-1]
			hmax = stack_start[i]
			vertical(ori,img,roi, i,k, hmin, hmax)

		if(i==len(stack_end)-1) :
			Li = search_last(freq, stack_end[i],0)
			roi =  img[stack_end[i]-1:Li+1, 0:w]
			h_roi = roi.shape[0]
			w_roi = roi.shape[1]
			hmin = stack_end[i]
			hmax = Li

			if(h_roi>0 and w_roi>0) :
					vertical(ori,img,roi, i+1,k,hmin,hmax)

		i+=1

	#cv2.imshow('segments',ori)
	#plt.imshow(ori)
	#plt.show()
	return stack_start,stack_end

def horizontal(pth,k) :
	kernel = np.ones((3,3), np.uint8)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = cv2.imread(pth)
	ori = cv2.imread(pth)
	h_or = img.shape[0]
	w_or = img.shape[1]

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

	gray = cv2.bilateralFilter(gray,10,97,75)
	gray = cv2.bilateralFilter(gray,10,97,75)

	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

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

	stack_start1,stack_end1 = segment3("ori",mean, img1, y_proj_ori, w_or,k,ori)

	#cv2.imshow('gray',gray)
	#cv2.imshow('histy',histy)

	#cv2.imshow('thresh',thresh)
	#cv2.waitKey(1000000)

#shutil.rmtree('./segment/')
#os.mkdir("./segment")
#horizontal("./ktp_ori2.png",0)
