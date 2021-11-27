"""
Author: ANTenna on 2021/11/20 4:58 下午
aliuyaohua@gmail.com

Description:

"""

import cv2
import numpy as np
from stereo_utils import calculate_census, calculate_disparity


if __name__ == '__main__':

    # read images
    img_left_path = 'StereoDataset/cones/im2.png'
    img_right_path = 'StereoDataset/cones/im6.png'

    # open images
    img_left = cv2.imread(img_left_path, 0)
    img_right = cv2.imread(img_right_path, 0)

    # parse option
    min_disparity = 0
    max_disparity = 64
    disparity_range = max_disparity - min_disparity

    # calculate census

    # cost aggregation

    # calculate disparity

