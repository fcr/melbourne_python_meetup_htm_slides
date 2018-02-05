#! /usr/bin/env python
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
This script trains the spatial pooler (SP) on a set of images that are
listed in the XML file specified by trainingDataset.  The SP is trained
for a maximum number of training cycles given by maxTrainingCycles and then its
classification abilities are tested on the images listed in the XML file
specified by testingDataset.
"""

from nupic.algorithms.spatial_pooler import SpatialPooler

import dataset_readers
import image_encoders as encoder
from vision_testbench import VisionTestBench
from classifiers import exactMatch


maxTrainingCycles = 20
trainingDataset = "OCR/characters/cmr_normal.xml"
# testingDataset = "OCR/characters/characters.xml"
testingDataset = "OCR/characters/cmr_normal.xml"



if __name__ == "__main__":
  # Get training images and convert them to vectors.
  trainingImages, trainingTags = dataset_readers.getImagesAndTags(trainingDataset)
  trainingVectors = encoder.imagesToVectors(trainingImages)

  # Instantiate the python spatial pooler
  sp=SpatialPooler(
    inputDimensions=(32,32), # Size of image patch
    columnDimensions=(8,8), # Number of potential features
    potentialRadius=10000, # Ensures 100% potential pool
    potentialPct=1, # Neurons can connect to 100% of input
    globalInhibition=True,
#     localAreaDensity=-1, # Using numActiveColumnsPerInhArea
    localAreaDensity=0.02, # two percent of columns active at a time
    numActiveColumnsPerInhArea=-1, # Using percentage instead
#     numActiveColumnsPerInhArea=1, # Only one feature active at a time
    # All input activity can contribute to feature output
    stimulusThreshold=0,
    synPermInactiveDec=0.3,
    synPermActiveInc=0.3,
    synPermConnected=0.5, # Connected threshold
    boostStrength=100,
    seed=1956, # The seed that Grok uses
    spVerbosity=1)

  # Instantiate the spatial pooler test bench.
  tb = VisionTestBench(sp)

  # Instantiate the classifier
  clf = exactMatch()

  # Train the spatial pooler on trainingVectors.
  numCycles = tb.train(trainingVectors, trainingTags, clf, maxTrainingCycles)

  # View the permanences and connections after training.
  tb.showConnections()
  #tb.savePermsAndConns('perms_and_conns.jpg')

  # Get testing images and convert them to vectors.
  testingImages, testingTags = dataset_readers.getImagesAndTags(testingDataset)
  testingVectors = encoder.imagesToVectors(testingImages)

  # Test the spatial pooler on testingVectors.
  accuracy = tb.test(testingVectors, testingTags, clf, verbose=1)
