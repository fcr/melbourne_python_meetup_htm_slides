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

from PIL import Image
import numpy
import hashlib

DEBUG = 0



class VisionTestBench(object):
  """
  This class provides methods for characterizing nupic's image recognition
  capabilities.  The goal is to put most of the details in here so the top
  level can be as clear and concise as possible.
  """


  def __init__(self, sp):
    """
    The test bench has just a few things to keep track off:

    - A list of the output SDRs that is shared between the training and testing
      routines

    - Height and width of the spatial pooler's inputs and columns which are
      used for producing images of permanences and connected synapses

    - Images of permanences and connected synapses so these images do not have
      to be generated more than necessary

    """
    self.sp = sp

    self.SDRs = []

    self.tags = []

    # These images are produced together so these properties are used to allow
    # them to be saved separately without having to generate the images twice.
    self.permanencesImage = None
    self.connectionsImage = None

    # Limit inputs and columns to 1D and 2D layouts for now
    inputDimensions = sp.getInputDimensions()
    try:
      assert len(inputDimensions) < 3
      self.inputHeight = inputDimensions[0]
      self.inputWidth = inputDimensions[1]
    except IndexError:
      self.inputHeight = int(numpy.sqrt(inputDimensions[0]))
      self.inputWidth = int(numpy.sqrt(inputDimensions[0]))
    except TypeError:
      self.inputHeight = int(numpy.sqrt(inputDimensions))
      self.inputWidth = int(numpy.sqrt(inputDimensions))

    columnDimensions = sp.getColumnDimensions()
    try:
      assert len(columnDimensions) < 3
      self.columnHeight = columnDimensions[0]
      self.columnWidth = columnDimensions[1]
    except IndexError:
      self.columnHeight = columnDimensions[0]
      self.columnWidth = 1
    except TypeError:
      self.columnHeight = columnDimensions
      self.columnWidth = 1


  def train(self, trainingVectors, trainingTags, classifier, maxCycles=10,
      minAccuracy=100.0):
    """
    This routine trains the spatial pooler using the bit vectors produced from
    the training images by using these vectors as input to the SP.  It continues
    training until either the minimum specified accuracy is met or the maximum
    number of training cycles is reached.  It records each output SDR as the
    index of that SDR in a list of all SDRs seen during training.  This list of
    indexes is used to generate the SDRs for evaluating recognition accuracy
    after each training cycle.  It also creates a list of all tags (ground
    truth) seen during training.  This list is used to establish the integer
    categories for the classifier so they can be used again during testing to
    establish the correct categories even if the order of the input vectors is
    changed.
    """
    
    # Get rid of permanence and connection images from previous training
    self.permanencesImage = None
    self.connectionsImage = None

    # print starting stats
    cyclesCompleted = 0
    accuracy = 0
    self.printTrainingStats(cyclesCompleted, accuracy)

    # keep training until minAccuracy or maxCycles is reached
    while (minAccuracy - accuracy) > 1.0/len(trainingTags) and cyclesCompleted < maxCycles:
      # increment cycle number
      cyclesCompleted += 1

      # Feed each training vector into the spatial pooler and then teach the
      # classifier to associate the tag and the SDR
      SDRIs = []
      classifier.clear()
      activeArray = numpy.zeros(self.sp.getNumColumns())
      for j,trainingVector in enumerate(trainingVectors):
        self.sp.compute(trainingVector, True, activeArray)
        # Build a list of indexes corresponding to each SDR
        activeList = activeArray.astype('int32').tolist()
        if activeList not in self.SDRs:
          self.SDRs.append(activeList)
        SDRI = self.SDRs.index(activeList)
        SDRIs.append(SDRI)
        # tell classifier to associate SDR and training Tag
        # if there are repeat tags give the index of the first occurrence
        if trainingTags[j] in self.tags:
          category = self.tags.index(trainingTags[j])
        else:
          self.tags.append(trainingTags[j])
          category = len(self.tags) - 1
        classifier.learn(activeArray, category)

      # Check the accuracy of the SP, classifier combination
      accuracy = 0.0
      for j in range(len(SDRIs)):
        SDRI = SDRIs[j]
        activeArray = numpy.array(self.SDRs[SDRI])
        # if there are repeat tags give the index of the first occurrence
        category = self.tags.index(trainingTags[j])
        inferred_category = classifier.infer(activeArray)[0]
        if inferred_category == category:
          accuracy += 100.0/len(trainingTags)

      # print updated stats
      self.printTrainingStats(cyclesCompleted, accuracy)

    print
    return cyclesCompleted


  def test(self, testVectors, testingTags, classifier, verbose=0, learn=True):
    """
    This routine tests the spatial pooler on the bit vectors produced from the
    testing images.
    """
    print "\nTesting:\n"

    # Get rid of old permanence and connection images
    self.permanencesImage = None
    self.connectionsImage = None

    # Feed testing vectors into the spatial pooler and build a list of SDRs.
    SDRIs = []
    activeArray = numpy.zeros(self.sp.getNumColumns())
    for j, testVector in enumerate(testVectors):
      self.sp.compute(testVector, learn, activeArray)
      # Build a list of indexes corresponding to each SDR
      activeList = activeArray.astype('int32').tolist()
      if activeList not in self.SDRs:
        self.SDRs.append(activeList)
      SDRIs.append(self.SDRs.index(activeList))
      if learn:
        # tell classifier to associate SDR and testing Tag
        category = self.tags.index(testingTags[j])
        classifier.learn(activeArray, category)

    # Check the accuracy of the SP, classifier combination
    accuracy = 0.0
    recognitionMistake = False
    if verbose > 1:
      print "%5s" % "Input", "Output"
    for j in range(len(SDRIs)):
      activeArray = numpy.array(self.SDRs[SDRIs[j]])
      category = self.tags.index(testingTags[j])
      inferences = classifier.infer(activeArray)
      if inferences is not None:        
        inferred_category = classifier.infer(activeArray)[0]
      else:
        inferred_category = None
      if inferred_category is not None and inferred_category == category:
        accuracy += 100.0/len(testingTags)
        if verbose > 1:
          print "%-5s" % testingTags[j], testingTags[inferred_category]
      else:
        if not recognitionMistake:
          recognitionMistake = True
          print "Recognition mistakes:"
          print "%5s" % "Input", "Output"
        if inferred_category is not None:
          inferred_char = testingTags[inferred_category]
        else:
          inferred_char = 'na'
        print "%-5s" % testingTags[j], inferred_char

    print
    print "Accuracy: %.1f" % accuracy, "%"
    print

    return accuracy


  def printTrainingStats(self, trainingCyclesCompleted, accuracy):
    """
    This routine prints the mean values of the connected and unconnected synapse
    permanences along with the percentage of synapses in each.
    It also returns the percentage of connected synapses so it can be used to
    determine when training has finished.
    """
    # Print header if this is the first training cycle
    if trainingCyclesCompleted == 0:
      print "\nTraining:\n"
      print "%5s" % "",
      print "%16s" % "Connected",
      print "%19s" % "Unconnected",
      print "%16s" % "Recognition"
      print "%5s" % "Cycle",
      print "%10s" % "Percent",
      print "%8s" % "Mean",
      print "%10s" % "Percent",
      print "%8s" % "Mean",
      print "%13s" % "Accuracy"
      print
    # Calculate permanence stats
    pctConnected = 0
    pctUnconnected = 0
    connectedMean = 0
    unconnectedMean = 0
    #perms = numpy.zeros(self.sp.getInputDimensions())
    perms = numpy.zeros(self.sp.getNumInputs())
    numCols = self.sp.getNumColumns()
    for i in range(numCols):
      self.sp.getPermanence(i, perms)
      numPerms = perms.size
      connectedPerms = perms >= self.sp.getSynPermConnected()
      numConnected = connectedPerms.sum()
      total_connected = max(numConnected*numCols, 1)
      pctConnected += 100.0/total_connected/numPerms
      sumConnected = (perms*connectedPerms).sum()
      connectedMean += sumConnected/total_connected
      unconnectedPerms = perms < self.sp.getSynPermConnected()
      numUnconnected = unconnectedPerms.sum()
      pctUnconnected += 100.0/numCols*numUnconnected/numPerms
      sumUnconnected = (perms*unconnectedPerms).sum()
      unconnectedMean += sumUnconnected/(numUnconnected*numCols)
    print "%5s" % trainingCyclesCompleted,
    print "%10s" % ("%.4f" % pctConnected),
    print "%8s" % ("%.3f" % connectedMean),
    print "%10s" % ("%.4f" % pctUnconnected),
    print "%8s" % ("%.3f" % unconnectedMean),
    print "%13s" % ("%.5f" % accuracy)




  def printOutputHash(self,trainingCyclesCompleted):
    """This routine prints the MD5 hash of the output SDRs."""
    # Print header if this is the first training cycle
    if trainingCyclesCompleted == 0:
      print "\nTraining begins:\n"
      print "%5s" % "Cycle",
      print "%34s" % "Connected MD5", "%34s" % "Permanence MD5"
      print ""
    # Calculate an MD5 checksum for the permanences and connected synapses so
    # we can see when learning has finished.
    permsMD5 = hashlib.md5()
    connsMD5 = hashlib.md5()
    perms = numpy.zeros(self.sp.getInputDimensions())
    for i in range(self.columnHeight):
      self.sp.getPermanence(i, perms)
      connectedPerms = perms >= self.sp.getSynPermConnected()
      perms = perms.astype('string')
      [permsMD5.update(word) for word in perms]
      connectedPerms = connectedPerms.astype('string')
      [connsMD5.update(word) for word in connectedPerms]
    print "%5s" % trainingCyclesCompleted,
    print "%34s" % connsMD5.hexdigest(), "%34s" % permsMD5.hexdigest()



  def calcPermsAndConns(self):
    """
    These routines generates images of the permanences and connections of each
    column so they can be viewed and saved.
    """
    size = (self.inputWidth*self.columnWidth,self.inputHeight*self.columnHeight)
    self.permanencesImage = Image.new('RGB', size)
    self.connectionsImage = Image.new('RGB', size)
    #perms = numpy.zeros(self.sp.getInputDimensions())
    perms = numpy.zeros(self.sp.getNumInputs())
    for j in range(self.columnWidth):
      for i in range(self.columnHeight):
        self.sp.getPermanence(i*self.columnWidth + j, perms)
        # Convert perms to RGB (effective grayscale) values
        allPerms = [(v, v, v) for v in ((1 - perms) * 255).astype('int')]

        connectedPerms = perms >= self.sp.getSynPermConnected()
        connectedPerms = (numpy.invert(connectedPerms) * 255).astype('int')
        connectedPerms = [(v, v, v) for v in connectedPerms]

        allPermsReconstruction = self._convertToImage(allPerms, 'RGB')
        connectedReconstruction = self._convertToImage(connectedPerms, 'RGB')
        size = allPermsReconstruction.size

        # Add permanences and connections for each column to the images
        x = j * self.inputWidth
        y = i * self.inputHeight
        self.permanencesImage.paste(allPermsReconstruction, (x,y))
        self.connectionsImage.paste(connectedReconstruction, (x,y))


  def showPermsAndConns(self):
    if self.permanencesImage == None:
      self.calcPermsAndConns()
    size = (2*self.permanencesImage.size[0],self.permanencesImage.size[1])
    pAndCImage = Image.new('RGB', size)
    pAndCImage.paste(self.permanencesImage, (0,0))
    pAndCImage.paste(self.connectionsImage, (size[0]/2,0))
    pAndCImage.show()


  def showPermanences(self):
    if self.permanencesImage == None:
      self.calcPermsAndConns()
    self.permanencesImage.show()


  def showConnections(self):
    if self.connectionsImage == None:
      self.calcPermsAndConns()
    self.connectionsImage.show()


  def savePermsAndConns(self, filename):
    if self.permanencesImage == None:
      self.calcPermsAndConns()
    size = (2*self.permanencesImage.size[0],self.permanencesImage.size[1])
    pAndCImage = Image.new('RGB', size)
    pAndCImage.paste(self.permanencesImage, (0,0))
    pAndCImage.paste(self.connectionsImage, (size[0]/2,0))
    pAndCImage.save(filename,'JPEG')


  def savePermanences(self, filename):
    if self.permanencesImage == None:
      self.calcPermsAndConns()
    self.permanencesImage.save(filename,'JPEG')


  def saveConnections(self, filename):
    if self.connectionsImage == None:
      self.calcPermsAndConns()
    self.connectionsImage.save(filename,'JPEG')


  # take an SDR index and return the corresponding SDR
  def getSDR(self, SDRI):
    assert SDRI < len(self.SDRs)
    return self.SDRs[SDRI]


  # take an SDR index and print the corresponding SDR
  def printSDR(self, SDRI):
    assert SDRI < len(self.SDRs)
    bitLength = len(self.SDRs[SDRI])
    lineLength = int(numpy.sqrt(bitLength))
    for i in range(bitLength):
      if i != 0 and i % lineLength == 0:
        print
      if self.SDRs[SDRI][i] == 1:
        print "1",
      else:
        print "_",
    print


  def _convertToImage(self, listData, mode = '1'):
    """
    Takes in a list and returns a new square image
    """

    # Assume we're getting a square image patch
    side = int(len(listData) ** 0.5)
    # Create the new image of the right size
    im = Image.new(mode, (side, side))
    # Put the data into that patch
    im.putdata(listData)

    return im
