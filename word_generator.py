#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Import python imaging libs
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Import operating system lib
import os, random

# Import random generator
from random import randint

def Init():
    # check if output directory exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    w = []
    f = open('words.txt', 'r')

    for l in f:
        w.append(l.strip())

    return w

def Cleanup():
    # Delete ds_store file
   	if os.path.isfile(font_dir + '.DS_Store'):
   		os.unlink(font_dir + '.DS_Store')

	# Delete all files from output directory
	for file in os.listdir(out_dir):
		file_path = os.path.join(out_dir, file)
		if os.path.isfile(file_path):
			os.unlink(file_path)
   	return

def WordGenerator():
    myword = random.sample(words, 3)

    return myword

def ListFont():
    lsfont = []
    for dirname, dirnames, filenames in os.walk(font_dir):
        for filename in filenames:
            font_path = os.path.join(dirname, filename)
            lsfont.append(font_path)

    return lsfont

def WordPainter():
    lsfont = ListFont()

    for i in range(10):
        w = WordGenerator()                 # Get words
        cur_font = random.choice(lsfont)    # Get font
        rand_font_size = random.choice(font_sizes)

        # Generate image
        word_image = Image.new('RGB', image_size, 'white')

        # Draw word
        draw = ImageDraw.Draw(word_image)

        # Specify font
        font = ImageFont.truetype(cur_font, rand_font_size)

        # Draw word
        x = 10
        y = 10
        label = ''
        for j in range(len(w)):
            draw.text((x, y), w[j], 'black', font=font)
            label = label + w[j] + '\n'
            y += 30
            x += 10 + random.uniform(1, 30)

        # Checking output folder exist or not
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Final filename
        img_filename = os.path.join(out_dir, str(i))

        # Create label
        f = open(img_filename+'.txt', 'w')
        f.write(label)
        f.close()

        # Save image
        word_image.save(img_filename+'.png')

        # Print filename
        print(img_filename)
        print("DONE")

    return

# Directory containing fonts
font_dir = 'fonts'

# Output
out_dir = 'test/'

# background color
background_colors = (66, 110, 244)

# Character size
font_sizes = (20, 22)

# Image size
image_size = (800, 600)

# Initialization
words = Init()

# Do Clean up
Cleanup()

# Save the words
WordPainter()
