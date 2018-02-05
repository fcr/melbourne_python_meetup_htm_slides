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
These routines convert images to bit vectors that can be used as input to
the spatial pooler.
"""

import numpy

from PIL import Image



def imageToVector(image):
  '''
  Returns a bit vector representation (list of ints) of a PIL image.
  '''
  # Convert the image to black and white
  image = image.convert('1',dither=Image.NONE)
  # Pull out the data, turn that into a list, then a numpy array,
  # then convert from 0 255 space to binary with a threshold.
  # Finally cast the values into a type CPP likes
  vector = (numpy.array(list(image.getdata())) < 100).astype('uint32')

  return vector



def imagesToVectors(images):
  vectors = [imageToVector(image) for image in images]
  return vectors
