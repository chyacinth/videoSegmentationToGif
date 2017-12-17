# videoSegmentationToGif

The program is a python implementation for hierarchial video segmentation algorithm. It also provides an interface for GIF generation. Enter the name of user selection JSON file (a file converted from a jpg indicating which part of the segmented parts the user wants in the final gif), you can get a segmented GIF from the video.

To use the segmentation algorithm, simply type: python <video>
You need to prepare additional JSON file converted from a jpg indicating which part of the segmented parts the user wants in the final gif in order to get a segmented GIF from the video. For example, if one pixel in the user selection JSON file is "1", then the segmented part that contains a pixel with the same location as the pixel in user selection file will be preserved in the generated GIF file.

You will need the following dependency:
python2.7
cv2
images2gif