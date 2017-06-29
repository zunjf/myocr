#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:19:15 2017

@author: srin
"""

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

    return

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

def CharGenerator():
    for dirname, dirnames, filenames in os.walk(font_dir):
        # For each font do
        for filename in filenames:
            # Get full path for each font
            font_path = os.path.join(dirname, filename)

            # for each character do
            for char in characters:

                # Generate character in random location
                for i in range(300):

                    # Convert the character into unicode
                    c = unicode(char, 'utf-8')

                    # Random backround and font
                    rand_background = random.choice(background_colors)
                    rand_font_size = random.choice(font_sizes)

                    # Generate: Grayscale, image size, background color
                    char_image = Image.new('RGB', (image_size, image_size), 'white')

                    # Draw character image_size
                    draw = ImageDraw.Draw(char_image)

                    # Specify font: Resource file, font size
                    font = ImageFont.truetype(font_path, rand_font_size)

                    # Get character width and height
                    (font_width, font_height) = font.getsize(c)

                    # Calculate x position
                    x = random.uniform(0, (image_size - font_width) - random.uniform(1, 2))

                    # Calculate y position
                    y = random.uniform(0, (image_size - font_height) - random.uniform(1,2))

                    # Draw text : Position, string
                    draw.text((x, y), c, (245-rand_background)+randint(0,10), font=font)

                    # Checking Character folder exist or not
                    if not os.path.exists(out_dir+c):
                        os.makedirs(out_dir+c)

                    # Final file name
                    add_filename = os.path.splitext(os.path.basename(filename))[0]
                    file_name = out_dir+c+'/'+c+str(i)+add_filename+'.png'

                    # Save image
                    char_image.save(file_name)

                    # Print character
                    print file_name

    return



# Directory containing fonts
font_dir = 'fonts'

# Output
out_dir = 'characters/'

# background color
background_colors = (66, 110, 244)

# Character size
font_sizes = (16, 18, 20)

# Image size
image_size = 24

# Numbers
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Small letters
small_letters = ['a', 'b', 'c', 'd','e', 'f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# Capital letters
capital_letters = ["A", 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W', 'X','Y','Z']

# Select characters
characters = numbers + small_letters + capital_letters

# Initialization
Init()

# Do cleanup
Cleanup()

# Generate characters
CharGenerator()
