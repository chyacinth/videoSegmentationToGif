from __future__ import print_function
import numpy as np
import cv2
import sys
from video import *
import IO
from graph_segmentation import *
if len(sys.argv) == 1:
    source = '13.mp4'
else:
    source = sys.argv[1]
# groundTruth = False
# visualize = False
readSource = "./videoTemp/" + source

video = IO.readVideo(readSource)
flowVideo = IO.readFlowVideo(readSource)

assert video.getFrameNumber() > 0
assert video.getFrameNumber() == flowVideo.getFrameNumber()
M = 300
L = 2
c = 0.2
beta = 0.2
alpha = 1 - beta

magic = segThres(c)
distance = segEuclidRGBFlow(alpha, beta)
segmenter = Segmentation(distance, magic)
# timer

fig = []

segmenter.buildGraph(video, flowVideo)
segmenter.buildEdges()
print("----- Level 0")
segmenter.Oversegment()
print("Oversegmented graph")
segmenter.minimumSeg(M)
print("Enforced minimum region size")
svVideo = segmenter.deriveLabels()
IO.writeColorSegVideo(0, svVideo, fig, None, False,source)
# save

hmagic = segHierarchyThres(c, 2)
hdistance = segHierarchyRGB(alpha, beta)
segmenter.setHierarchyM(hmagic)
segmenter.setHierarchyDist(hdistance)

for l in range(L):
    segmenter.buildRegionGraph()
    print("----- Level " + str(l + 1))
    segmenter.addLevel()
    print("print Segmented region graph")
    segmenter.minimumSeg(l / 2 * M)
    # M = l / 2 * M
    print("Enforce minimum segment size")
    svVideo = segmenter.deriveLabels()
    if (l < L - 1):
        IO.writeColorSegVideo(l + 1, svVideo, fig, None, False,source)
    else:
        IO.writeColorSegVideo(l + 1, svVideo, fig, video, True,source)
    # save

print("all Finished! ")
