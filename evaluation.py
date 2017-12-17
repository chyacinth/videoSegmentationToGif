import math
import numpy as np



def isSpatialBoundaryPixel(video, t, i, j):
    H = video.getFrameHeight()
    W = video.getFrameWidth()
    if i < H - 1:
        if video.get(t, i, j) != video.get(t, i + 1, j):
            return True

    if j < W - 1:
        if video.get(t, i, j) != video.get(t, i, j+1):
            return False

    return False

def isTemporalBoundaryPixel(video, t, i, j):
    T = video.getFrameNumber()
    if t < T -1:
        if video.get(t, i, j) != video.get(t + 1, i, j):
            return True

    return False

def computeSegmentNumber(sp_image):
    assert len(sp_image) > 0
    assert len(sp_image[0]) >0
    assert len(sp_image.shape) == 2
    # assert sp_image.type() == CV_32S

    segmentNumber = 0

    for i in range(len(sp_image)):
        for j in range(len(sp_image[i])):
            if sp_image[i][j] > segmentNumber:
                segmentNumber = sp_image[i][j]
    return segmentNumber+1

class Evaluation:
    @staticmethod
    def compute3DBoundaryRecall(sv_video, gt_video, d = 0.0025):
        assert sv_video.getFrameNumber() == gt_video.getFrameNumber()
        assert sv_video.getFrameHeight() == gt_video.getFrameHeight()
        assert sv_video.getFrameWidth() == gt_video.getFrameWidth()

        T = gt_video.getFrameNumber()
        H = gt_video.getFrameHeight()
        W = gt_video.getFrameWidth()

        r = round(d*math.sqrt(H*H+W*W))

        tp = 0.0
        fn = 0.0
        for t in range(T):
            for i in range(H):
                for j in range(W):
                    if isSpatialBoundaryPixel(gt_video, t, i, j):
                        pos = False
                        for k in range(max(0, i-r), min(H-1,i+r)+1, 1):
                            for l in range(max(0, j -r), min(W-1, j+r)+1, 1):
                                if isSpatialBoundaryPixel(sv_video, t, k, l):
                                    pos = True
                        if pos:
                            tp += 1
                        else:
                            fn += 1

                    if isTemporalBoundaryPixel(gt_video, t, i, j):
                        pos = False
                        for k in range(max(0, i -r), min(H-1, i+r)+1, 1):
                            for l in range(max(0,j-r), min(W-1, j+r)+1, 1):
                                if isTemporalBoundaryPixel(sv_video, t, k, l):
                                    pos = True
                        if pos:
                            tp += 1
                        else:
                            fn+= 1
        return tp/(tp+fn)


    @staticmethod
    def compute3DBoundaryPrecision(sv_video, gt_video, d = 0.0025):
        assert sv_video.getFrameNumber() == gt_video.getFrameNumber()
        assert sv_video.getFrameHeight() == gt_video.getFrameHeight()
        assert sv_video.getFrameWidth() == gt_video.getFrameWidth()

        T = gt_video.getFrameNumber()
        H = gt_video.getFrameHeight()
        W = gt_video.getFrameWidth()

        r = round(d * math.sqrt(H * H + W * W))

        tp = 0.0
        fp = 0.0
        for t in range(T):
            for i in range(H):
                for j in range(W):
                    if isSpatialBoundaryPixel(gt_video, t, i, j):
                        pos = False
                        for k in range(max(0, i - r), min(H - 1, i + r) + 1, 1):
                            for l in range(max(0, j - r), min(W - 1, j + r) + 1, 1):
                                if isSpatialBoundaryPixel(sv_video, t, k, l):
                                    pos = True
                        if pos:
                            tp += 1
                    elif isSpatialBoundaryPixel(sv_video, t, i, j):
                        pos = False
                        for k in range(max(0, i - r), min(H - 1, i + r) + 1, 1):
                            for l in range(max(0, j - r), min(W - 1, j + r) + 1, 1):
                                if isSpatialBoundaryPixel(gt_video, t, k, l):
                                    pos = True
                        if not pos:
                            fp += 1

                    if isTemporalBoundaryPixel(gt_video, t, i, j):
                        pos = False
                        for k in range(max(0, i - r), min(H - 1, i + r) + 1, 1):
                            for l in range(max(0, j - r), min(W - 1, j + r) + 1, 1):
                                if isTemporalBoundaryPixel(sv_video, t, k, l):
                                    pos = True
                        if pos:
                            tp += 1
                    elif isTemporalBoundaryPixel(sv_video, t, i, j):
                        pos = False
                        for k in range(max(0, i - r), min(H - 1, i + r) + 1, 1):
                            for l in range(max(0, j - r), min(W - 1, j + r) + 1, 1):
                                if isTemporalBoundaryPixel(gt_video, t, k, l):
                                    pos = True
                        if not pos:
                            fp += 1


        if(tp + fp > 0): return tp/(tp+fp)
        return 0


    @staticmethod
    def compute2DBoundaryRecallPerFrame(sv_video, gt_video, d = 0.0025):
        assert sv_video.getFrameNumber() == gt_video.getFrameNumber()
        assert sv_video.getFrameHeight() == gt_video.getFrameHeight()
        assert sv_video.getFrameWidth() == gt_video.getFrameWidth()

        T = gt_video.getFrameNumber()
        H = gt_video.getFrameHeight()
        W = gt_video.getFrameWidth()

        r = round(d * math.sqrt(H * H + W * W))
        recall = [0.0]*T
        for t in range(T):
            tp = 0.0
            fn = 0.0
            for i in range(H):
                for j in range(W):
                    if isSpatialBoundaryPixel(gt_video, t, i, j):
                        pos = False
                        for k in range(max(0, i-r), min(H-1,i+r)+1,1):
                            for l in range(max(0, j-r), min(W-1, j+r)+1, 1):
                                if isSpatialBoundaryPixel(sv_video, t, k, l):
                                    pos = True
                        if pos: tp += 1
                        else: fn += 1
            recall[t] = tp/(tp+fn)
        return recall


    @staticmethod
    def compute2DBoundaryPrecisionPerFrame(sv_video, gt_video, d = 0.0025):
        assert sv_video.getFrameNumber() == gt_video.getFrameNumber()
        assert sv_video.getFrameHeight() == gt_video.getFrameHeight()
        assert sv_video.getFrameWidth() == gt_video.getFrameWidth()

        T = gt_video.getFrameNumber()
        H = gt_video.getFrameHeight()
        W = gt_video.getFrameWidth()

        r = round(d * math.sqrt(H * H + W * W))

        precision = [0.0] * T
        for t in range(T):
            tp = 0.0
            fp = 0.0
            for i in range(H):
                for j in range(W):
                    if isSpatialBoundaryPixel(gt_video, t, i, j):
                        pos = False
                        for k in range(max(0, i -r), min(H-1, i+r)+1, 1):
                            for l in range(max(0, j-r), min(W-1, j+r)+1, 1):
                                if isSpatialBoundaryPixel(sv_video, t, k, l):
                                    pos = True
                        if pos:
                            tp += 1

                    elif isSpatialBoundaryPixel(sv_video, t, i, j):
                        pos = False
                        for k in range(max(0, i-r), min(H-1,i+r)+1, 1):
                            for l in range(max(0, j-r), min(W-1, j+r)+1, 1):
                                if isSpatialBoundaryPixel(gt_video, t, k, l):
                                    pos = True
                        if not pos:
                            fp += 1
            precision[t] = 0.0;
            if tp+fp>0:
                precision[t] = tp/(tp+fp)
        return precision




    @staticmethod
    def compute3DXCUndersegmentationError(sv_video, gt_video):
        intersectionMatrix = Evaluation.__computeIntersectionMatrix(sv_video, gt_video)

        error = 0.0

        for i in range(len(intersectionMatrix)):
            if gt_video.getSegmentSize(i) > 0:
                gtIError = 0.0
                for j in range(len(intersectionMatrix[i])):
                    if intersectionMatrix[i][j] > 0:
                        gtIError += sv_video.getSegmentSize(j)

                gtIError -= gt_video.getSegmentSize(i)
                gtIError /= gt_video.getSegmentSize(i)

                error += gtIError

        return error/gt_video.getSegmentNumber()


    @staticmethod
    def compute3DNPUndersegmentationError(sv_video, gt_video):
        intersectionMatrix = Evaluation.__computeIntersectionMatrix(sv_video, gt_video)

        N = sv_video.getFrameHeight() * sv_video.getFrameWidth() * sv_video.getFrameNumber()
        error = 0.0

        for i in range(len(intersectionMatrix)):
            for j in range(len(intersectionMatrix[i])):
                if intersectionMatrix[i][j] > 0:
                    error += min(intersectionMatrix[i][j], sv_video.getSegmentSize - intersectionMatrix[i][j])

        return error/N


    @staticmethod
    def compute2DNPUndersegmentationErrorPerFrame(sv_video, gt_video):
        assert sv_video.getFrameNumber() == gt_video.getFrameNumber()

        T = sv_video.getFrameNumber()
        N = sv_video.getFrameHeight()*sv_video.getFrameWidth()

        errors = [0.0] * T
        for t in range(T):
            superpixelSizes = []
            gtSizes = []

            intersectionMatrix = Evaluation.__computeIntersectionMatrix(sv_video.getFrame(t), gt_video.getFrame(t), superpixelSizes, gtSizes)

            for i in range(len(intersectionMatrix)):
                for j in range(len(intersectionMatrix[i])):
                    if intersectionMatrix[i][j] > 0:
                        errors[t] += min(intersectionMatrix[i][j], superpixelSizes[j] - intersectionMatrix[i][j])
            errors[t] /= N

        return errors

    @staticmethod
    def compute3DAchievableSegmentationAccuracy(sv_video, gt_video):
        intersectionMatrix = Evaluation.__computeIntersectionMatrix(sv_video, gt_video)

        N = sv_video.getFrameNumber()*sv_video.getFrameHeight()*sv_video.getFrameWidth()
        accuracy = 0.0

        for j in range(len(intersectionMatrix[0])):
            max = 0
            for i in range(len(intersectionMatrix)):
                if intersectionMatrix[i][j] > max:
                    max = intersectionMatrix[i][j]
            accuracy += max

        return accuracy/N


    # @staticmethod
    # def computeTemporal2DAchievableSegmentationAccuracyPerFrame(sv_video, gt_video):

    @staticmethod
    def compute2DAchievableSegmentationAccuracyPerFrame(sv_video, gt_video):
        assert sv_video.getFrameNumber() == gt_video.getFrameNumber()

        T = sv_video.getFrameNumber()
        N = sv_video.getFrameHeight() * sv_video.getFrameWidth()

        accuracies = [0.0] * T

        for t in range(T):
            superpixelSizes = []
            gtSizes = []

            intersectionMatrix = Evaluation.__computeIntersectionMatrix(sv_video.getFrame(t), gt_video.getFrame(t), superpixelSizes, gtSizes)

            for j in range(len(intersectionMatrix[0])):
                max = 0
                for i in range(len(intersectionMatrix)):
                    if intersectionMatrix[i][j] > max:
                        max = intersectionMatrix[i][j]

                accuracies[t] += max
            accuracies[t] /= N
        return accuracies

    @staticmethod
    def __computeIntersectionMatrix(sp_image, gt_image, superpixelSizes, gtSizes):
        assert len(sp_image) == len(gt_image)
        assert len(sp_image[0]) == len(gt_image[0])
        assert len(sp_image.shape) == 2 and len(gt_image.shape) == 2

        rows = len(sp_image)
        cols = len(sp_image)

        superpixels = computeSegmentNumber(sp_image)
        gtSegments = computeSegmentNumber(gt_image)

        assert superpixels >0
        assert gtSegments >0

        superpixelSizes += [0]*superpixels
        gtSizes += [0]*gtSegments

        intersectionMatrix = np.zeros((gtSegments, superpixels), np.int32)
        for i in range(rows):
            for j in range(cols):
                intersectionMatrix[gt_image[i][j]][sp_image[i][j]] += 1
                superpixelSizes[sp_image[i][j]] += 1
                gtSizes[gt_image[i][j]] += 1
        return intersectionMatrix


    @staticmethod
    def __computeIntersectionMatrix(sv_video, gt_video):
        assert sv_video.getFrameNumber() == gt_video.getFrameNumber()
        assert sv_video.getFrameWidth() == gt_video.getFrameWidth()
        assert sv_video.getFrameHeight() == gt_video.getFrameHeight()
        assert sv_video.getFrameChannels() == gt_video.getFrameChannels()

        rows = sv_video.getFrameHeight()
        cols = sv_video.getFrameWidth()

        supervoxels = sv_video.getSegmentNumber()
        gtSegments = gt_video.getSegmentNumber()

        intersectionMatrix = np.zeros((gtSegments, supervoxels), np.int32)

        for t in range(sv_video.getFrameNumber()):
            for i in range(rows):
                for j in range(cols):
                    intersectionMatrix[gt_video.get(t, i, j)][sv_video.get(t, i, j)] += 1

        return intersectionMatrix

