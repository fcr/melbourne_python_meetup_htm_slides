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
This module contains different classifiers to try when using NuPIC on visual
tasks.
"""

from nupic.algorithms.knn_classifier import KNNClassifier
from nupic.algorithms.sdr_classifier import SDRClassifier



class exactMatch(object):
  """
  This classifier builds a list of SDRs and their associated categories.  When
  queried for the category of an SDR it returns the first category in the list
  that has a matching SDR.
  """


  def __init__(self):
    """
    This classifier has just two things to keep track off:

    - A list of the known categories 

    - A list of the SDRs associated with each category
    """
    self.SDRs = []
    self.categories = []


  def clear(self):
    self.SDRs = []
    self.categories = []


  def learn(self, inputPattern, inputCategory, isSparse=0):
    inputList = inputPattern.astype('int32').tolist()
    if inputList not in self.SDRs:
      self.SDRs.append(inputList)
      self.categories.append([inputCategory])
    else:
      self.categories[self.SDRs.index(inputList)].append(inputCategory)


  def infer(self, inputPattern):
    inputList = inputPattern.astype('int32').tolist()
    if inputList in self.SDRs:
      winner = self.categories[self.SDRs.index(inputList)][0]
      # format return value to match KNNClassifier
      result = (winner, [], [], [])
      return result
