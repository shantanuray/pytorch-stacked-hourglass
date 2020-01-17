"""Opertions on images from an Image set."""
import cv2

def mean_stddev(imageset):
    """Get the mean and standard deviation of all images from an imageset."""
    mean = 0
    stddev = 0
    for i in imageset.images:
        img = cv2.imread(i.path)
        if img.max() > 1:
            img /= 255
        mean_i, stddev_i = cv2.meanStdDev(img)
        mean += mean_i
        stddev += stddev_i
    mean = mean / len(imageset.images)
    stddev = stddev / len(imageset.images)
    return(mean, stddev)
