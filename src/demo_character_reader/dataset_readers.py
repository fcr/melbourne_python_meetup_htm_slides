# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
This module provides routines that read in image recognition data sets
consisting of images and their associated ground truth.  Image recognition data
sets come in different formats so there are routines for reading each format
that has been used with nupic.vision.
"""

import os
from xml.dom import minidom

from PIL import Image

DEBUG = 0



def getImagesAndTags(filename):
  """
  This routine reads the XML files that contain the paths to the images and the
  tags which indicate what is in the image (i.e. "ground truth").
  """
  filename = os.path.join("data", filename)
  print
  print "Reading data set: ", filename
  print
  xmldoc = minidom.parse(filename)
  # Find the path to the XML file so it can be used to find the image files.
  directoryPath = filename.replace(filename.split("/")[-1], "")
  # Read the image list from the XML file and populate images and tags.
  imageList = xmldoc.getElementsByTagName('image')
  images = []
  tags = []
  count = 0
  for image in imageList:
    tags.append(image.attributes['tag'].value)
    filename = image.attributes['file'].value
    fp = open(directoryPath + filename, 'rb')
    im = Image.open(fp)
    im.load()
    images.append(im)
    fp.close()
    count += 1
    #imagePatches[-1].show()
    
  num_fonts = count/(26*2 + 10)
  if num_fonts == 1:
    print("1 font read")
  else:
    print("{} fonts read".format(num_fonts))
    
  return images, tags
