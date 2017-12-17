import numpy as np
import cv2

class Video:
    def __init__(self):
        self.frameWidth = 0
        self.frameHeight = 0
        self.frameChannels  =0
        # self.frameType = 0
        self.frames = []

    # def __del__(self):

    def addFrame(self, frame):
        assert len(frame) >0 and len(frame[0])> 0

        if len(self.frames) == 0:
            self.frameHeight = len(frame)
            self.frameWidth = len(frame[0])
            # self.frameType = 0
            self.frameChannels = frame.shape

        assert len(frame) == self.frameHeight and len(frame[0]) == self.frameWidth
        assert self.frameChannels == frame.shape # and frame.type() == self.frameType

        self.frames += [frame.copy()]
        return len(self.frames)

    def getFrameWidth(self):
        return self.frameWidth

    def getFrameHeight(self):
        return self.frameHeight

    # def getFrameType(self):

    def getFrameChannels(self):
        return self.frameChannels

    def getFrameNumber(self):
        return len(self.frames)

    def getFrame(self, t):
        assert t >= 0 and t < len(self.frames)

        return self.frames[t]

    def get(self, t, i, j):
        assert t >= 0 and t < len(self.frames)
        assert i >= 0 and i < self.frameHeight
        assert j >= 0 and j < self.frameWidth

        return self.frames[t][i][j]


class segVideo(Video):
    def __init__(self):
        Video.__init__(self)
        self.segments = []

    # def __del__(self):


    def getSegmentNumber(self):
        if len(self.segments) == 0:
            self._initializeSegments()

        return len(self.segments)


    def getSegmentSize(self, label):
        assert label >= 0 and label < len(self.segments)

        return self.segments[label]


    def relabel(self):
        segmentNumber = self.getSegmentNumber()
        mapping = [-1] * segmentNumber

        label = 0

        for t in range(len(self.frames)):
            for i in range(self.frameHeight):
                for j in range(self.frameWidth):
                    if mapping[self.frames[t][i][j]] < 0:
                        mapping[self.frames[t][i][j]] = label
                        label += 1

                    self.frames[t][i][j] = mapping[self.frames[t][i][j]]
        self.segments = []

    # def relabelSemantic(self, mapping):



    @staticmethod
    def add(videoA, videoB, relabel = True):
        assert videoA.getFrameNumber() == videoB.getFrameNumber()
        assert videoA.getFrameHeight() == videoB.getFrameHeight()
        assert videoA.getFrameWidth() == videoB.getFramewidth()

        T = videoA.getFrameNumber()
        rows = videoA.getFrameHeight()
        cols = videoA.getFrameWidth()

        videoBSegments = videoB.getSegmentNumber()

        result = SegmentationVideo()

        for t in range(T):
            frame = np.zeros((rows, cols), np.int32)
            for i in range(rows):
                for j in range(cols):
                    frame[i][j] = videoBSegments * videoA.get(t, i, j)+videoB.get(t, i, j)

            result.addFrame(frame)

        if relabel:
            result.relabel()

        return result

    def _initializeSegments(self):
        segmentCount = 0
        for t in range(len(self.frames)):
            for i in range(self.frameHeight):
                for j in range(self.frameWidth):
                    if self.frames[t][i][j] > segmentCount:
                        segmentCount = self.frames[t][i][j]

        self.segments = [0] * (segmentCount+1)
        for t in range(len(self.frames)):
            for i in range(self.frameHeight):
                for j in range(self.frameWidth):
                    self.segments[self.frames[t][i][j]] += 1



class FlowVideo(Video):
    def __init__(self):
        Video.__init__(self)
        self.Flow = True

    # def __del__(self):






