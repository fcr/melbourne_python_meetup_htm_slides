#!/usr/bin/env python
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
This script trains the spatial pooler (SP) on a set of images until it achieves
either a minimum specified image recognition accuracy on the training data set
or until a maximum number of training cycles is reached.  Then its image
recognition abilities are tested on another set of images.

trainingDataset - name of XML file that lists the training images

testingDataset - name of XML file that lists the testing images

minAccuracy - minimum accuracy requred to stop training before
              maxTrainingCycles is reached

maxTrainingCycles - maximum number of training cycles to perform

"""

#from nupic.bindings.algorithms import SpatialPooler
from nupic.research.spatial_pooler import SpatialPooler
from nupic.support.unittesthelpers.algorithm_test_helpers import convertSP

import nupic.vision.ocr.dataset_readers as data
import nupic.vision.ocr.image_encoders as encoder
from nupic.vision.ocr.vision_testbench import VisionTestBench
from nupic.vision.ocr.classifiers import KNNClassifier

trainingDataset = "OCR/characters/cmr_all.xml"
testingDataset = "OCR/characters/cmr_all.xml"
minAccuracy = 100.0  # force max training cycles
maxTrainingCycles = 5



if __name__ == "__main__":
  # Instantiate our spatial pooler
  sp = SpatialPooler(
    inputDimensions=(32, 32), # Size of image patch
    columnDimensions=(32, 32),
    potentialRadius=10000, # Ensures 100% potential pool
    potentialPct=0.8,
    globalInhibition=True,
    localAreaDensity=-1, # Using numActiveColumnsPerInhArea
    #localAreaDensity=0.02, # one percent of columns active at a time
    #numActiveColumnsPerInhArea=-1, # Using percentage instead
    numActiveColumnsPerInhArea=64,
    # All input activity can contribute to feature output
    stimulusThreshold=0,
    synPermInactiveDec=0.001,
    synPermActiveInc=0.001,
    synPermConnected=0.3,
    minPctOverlapDutyCycle=0.001,
    dutyCyclePeriod=1000,
    boostStrength=1.0,
    seed=1956, # The seed that Grok uses
    spVerbosity=1)

  # Instantiate the spatial pooler test bench.
  tb = VisionTestBench(sp)

  # Instantiate the classifier
  clf = KNNClassifier()

  # Get training images and convert them to vectors.
  trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
  trainingVectors = encoder.imagesToVectors(trainingImages)

  # Train the spatial pooler on trainingVectors.
  numCycles = tb.train(trainingVectors, trainingTags, clf, maxTrainingCycles,
    minAccuracy)

  # Save the permanences and connections after training.
  #tb.savePermanences('perms.jpg')
  #tb.showPermanences()
  #tb.showConnections()

  # Get testing images and convert them to vectors.
  testingImages, testingTags = data.getImagesAndTags(testingDataset)
  testingVectors = encoder.imagesToVectors(testingImages)

  # Reverse the order of the vectors and tags for testing
  testingTags = [testingTag for testingTag in reversed(testingTags)]
  testingVectors = [testingVector for testingVector in reversed(testingVectors)]

  # Test the spatial pooler on testingVectors.
  accuracy = tb.test(testingVectors, testingTags, clf, learn=True)
  print "Number of training cycles:", numCycles
