import numpy as np
import cv2
import random
import sys
from video import *
import json
import time
from images2gif import writeGif


def readVideo(path):
    if len(path) == 0:
        cap = cv2.VideoCapture('1.MOV')
    else:
        cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print 'Cannot initialize video capture'
        sys.exit(-1)

    result = Video()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    # while(ret):
    for i in range(2):
        result.addFrame(frame)
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()

    cap.release()
    return result


def readFlowVideo(path):
    result = FlowVideo()

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print 'Cannot initialize video capture'
        sys.exit(-1)

    # the parameter for Optical Flow
    # windowName = "test"
    # frameNunber = 0
    prevFrame = None
    pyramidScale = 0.5
    pyramidLevels = 3
    windowSize = 15
    iterations = 3
    polynomialNeighborhoodSize = 5
    polynomialSigma = 1.2
    flags = cv2.OPTFLOW_USE_INITIAL_FLOW  # cv2.OPTFLOW_FARNEBACK_GAUSSIAN

    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    ret, frame = cap.read()
    prevFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, frame = cap.read()
    # prevFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # while (ret):
    for i in range(2):
        nextFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow(windowName, frame)
        result.addFrame(cv2.calcOpticalFlowFarneback(prevFrame,
                                                     nextFrame,
                                                     pyramidScale,
                                                     pyramidLevels,
                                                     windowSize,
                                                     iterations,
                                                     polynomialNeighborhoodSize,
                                                     polynomialSigma,
                                                     flags))
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        prevFrame = nextFrame

        # print flows
    cap.release()
    # return flows
    # cv2.destroyAllWindows()

    return result


def writeColorSegVideo(path, video, fig, orivideo, distill, source):
    colors = []
    for i in range(video.getSegmentNumber()):
        colors += [[0, 0, 0]]

    t = 0
    numToC = {}
    gif = []
    for t in range(video.getFrameNumber()):
        labels = video.getFrame(t)
        coloredLabels = np.zeros((len(labels), len(labels[0]), 3), dtype=np.uint8)
        if (distill) and (t == 0):
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    label = labels[i][j]
                    while colors[label][0] == 0:
                        colors[label][0] = random.randrange(1, 256)
                        colors[label][1] = random.randrange(1, 256)
                        colors[label][2] = random.randrange(1, 256)
                    coloredLabels[i][j] = colors[label]

            # cv2.imwrite(str(path) + '.' + str(t) + '.jpg', coloredLabels)
            cv2.imwrite("./public/images/result/" + source + '.jpg', coloredLabels)
            sys.stdout.flush()
            sys.stdout.flush()
            time.sleep(1)
            print "409"
            sys.stdout.flush()
            name = raw_input("enter the name for user selection JSON file")
            fig = json.loads(name)
            print "fig width"+str(len(fig))
            print "fig h"+str(len(fig[0]))
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if (fig[i][j] == 1):
                        numToC[labels[i][j]] = [255, 255, 255]

        for i in range(len(labels)):
            for j in range(len(labels[i])):
                label = labels[i][j]
                if (distill):
                    if (numToC.has_key(label)):
                        #if numToC[label] == [255, 255, 255]:
                        coloredLabels[i][j] = orivideo.get(t, i, j)
                    else:
                        coloredLabels[i][j] = [0, 0, 0]
                else:
                    while colors[label][0] == 0:
                        colors[label][0] = random.randrange(0, 256)
                        colors[label][1] = random.randrange(0, 256)
                        colors[label][2] = random.randrange(0, 256)
                    coloredLabels[i][j] = colors[label]
        # write as image
        # coloredLabels
        cv2.imwrite(str(path) + '.' + str(t) + '.jpg', coloredLabels)
        if (distill):
            gif.append(coloredLabels.copy())
    if (distill):
        writeGif("gif.gif", gif, duration=0.5, subRectangles=False)

    return t
