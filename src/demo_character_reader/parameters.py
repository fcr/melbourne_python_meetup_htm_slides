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

import random



def linearRange(start, stop, step):
  """Make a list of allowed parameter values."""
  pval = start
  plist = [pval]
  while pval < stop:
    pval = pval + step
    plist.append(pval)
  return plist



class Parameters(object):
  """
  This class provides methods for searching ranges of parameters to see how
  they affect performance.
  """


  def __init__(self):
    """
    Have to keep track of the names and valid values of each parameter
    defined by the user.
    """
    # list of parameter names
    self._names = []

    # list of allowed parameter values
    self._allowedValues = []

    # list of past and present parameter value indexes
    self._valueIndexes = []

    # list of past and present results that correspond to each set of parameter
    # values
    self._results = []

    # the number of possible combinations of parameter values for all parameters
    self.numCombinations = 1


  def define(self, name, allowedValues):
    """
    This method allows users to define a parameter by providing its name and
    a list of values for the parameter.
    """
    if name not in self._names:
      self._names.append(name)
      self._allowedValues.append(allowedValues)
      self.numCombinations = self.numCombinations * len(allowedValues)
    else:
      print "Parameter: ", name, " is already defined!"


  def getNames(self):
    """
    This method returns the names of all defined parameters.
    """
    return self._names


  def getValue(self, name):
    """
    This method returns the current value of the parameter specified by name.
    """
    assert name in self._names
    i = self._names.index(name)
    assert len(self._valueIndexes[-1]) > i
    return self._allowedValues[i][self._valueIndexes[-1][i]]


  def getAllValues(self):
    """
    This method returns the current values of all defined parameters.
    """
    return [self._allowedValues[i][j] for i,j in 
      enumerate(self._valueIndexes[-1])]


  def appendResults(self,item):
    """
    This method adds an item to the results list.
    """
    print "Just completed parameter Combination: ", self.getAllValues()
    self._results.append(item)
    print
    print "Parameter combinations completed: ",
    print len(self._results), "/", self.numCombinations
    print


  def getNumResults(self):
    """
    This method returns the number of items in the results list.
    """
    return len(self._results)


  def printResults(self, resultNames, formatStrings):
    """
    This method prints a summary of all the results.
    """
    print
    print "Summary of Results"
    print
    headerList = self.getNames()
    headerList.extend(resultNames)
    headerString = ", ".join(headerList)
    print headerString
    for i, result in enumerate(self._results):
      valueString = str([self._allowedValues[j][k] for j,k in 
        enumerate(self._valueIndexes[i])])[1:-1]
      for j,formatString in enumerate(formatStrings):
        valueString += formatString % result[j]
      print valueString


  def nextRandomCombination(self):
    """
    This method randomly selects a value for each parameter from its list of
    allowed parameter values.  If the resulting combination has already been
    used then it tries again.
    """
    random_combination = [random.choice(self._allowedValues[i])
      for i in range(len(self._names))]
    if random_combination in self._values:
      self.nextRandomCombination()
    else:
      self._values.append(random_combination)
      print "Parameter Combination: ", self.getAllValues()
      print


  def nextCombination(self):
    """
    This method finds the next combination of parameter values using the
    allowed value lists for each parameter.
    """
    if len(self._valueIndexes) == 0:
      # list of value indexes is empty so this is the first combination, 
      # each parameter gets the first value in its list of allowed values
      self._valueIndexes.append([0 for i in range(len(self._names))])
    else:
      newValueIndexes = self._valueIndexes[-1][:]
      i = 0
      while i < len(self._names):
        # if current value is not the last in the list
        if self._valueIndexes[-1][i] != len(self._allowedValues[i]) - 1:
          # change parameter to next value in allowed value list and return
          newValueIndexes[i] += 1
          break
        else:
          # change parameter to first value in allowed value list
          newValueIndexes[i] = 0
          # move next parameter to next value in its allowed value list
          i = i + 1
      self._valueIndexes.append(newValueIndexes)

    print "Parameter Combination: ", self.getAllValues()
    print
