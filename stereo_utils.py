"""
Author: ANTenna on 2021/11/20 6:55 下午
aliuyaohua@gmail.com

Description:
implementation of SGM stereo matching algorithm
"""
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
local cost calculation
"""


# compare given two value, if abs(a - b) < 1, return 1, else return 2
def trunc_func(a, b):
    if abs(a - b) == 1:
        return 1
    else:
        return 2


def calculate_census(img, census_size, is_padding=0):
    """
    calculate pixel-wise census cost
    :param img: grey image
    :param census_size: window size to calculate census response
    :param is_padding: padding size before calculate census
    :return: pixel-wise census response of given image
    """

    h, w = img.shape
    census_h_ori, census_w_ori = census_size
    img_census = np.zeros((h, w, census_w_ori * census_h_ori), dtype=bool)
    census_h, census_w = census_h_ori // 2, census_w_ori // 2  # 将全尺寸转化为左右增量
    # if is_padding:

    for i in range(h):
        if i % 10 == 0:
            print('calculate_census: i=[{}/{} : {:.3f}]'.format(i, h, i / h))
        for j in range(w):
            # for each pixel, get its census value
            center_pixel = img[i][j]
            # sliding window
            for m in range(-census_h, census_h + 1):
                img_idx_h = i + m
                for n in range(-census_w, census_w + 1):
                    img_idx_w = j + n
                    if (img_idx_w < 0) or (img_idx_h < 0) or (img_idx_w >= w) or (img_idx_h >= h):
                        # img_census[i][j][0] = 0  # 不在范围内，不用给谁赋值
                        # print("m: {} n: {}".format(m, n))
                        continue  # 不在范围内，不用给谁赋值
                    else:
                        nbr_pixel = img[img_idx_h][img_idx_w]
                        if center_pixel > nbr_pixel:
                            img_census[i][j][m * census_w_ori + n] = 1
                        else:
                            img_census[i][j][m * census_w_ori + n] = 0
    return img_census


def calculate_ssd(img, ssd_size):
    image_ssd = 0
    return image_ssd


def calculate_ncc(img, ncc_size):
    img_ncc = 0
    return img_ncc


def calculate_hamming(census_a, census_b) -> int:
    """
    calculate hamming distance of given census elements
    :param census_a: fist single unit census response [0, 1, 0, ...]
    :param census_b: second single unit census response [1, 0, 0, ...]
    :return: hamming distance of given census elements: sum of (a XOR b)
    """
    return sum(census_a != census_b)


def calculate_init_cost_volume(left_census, right_census, disparity_range=(0, 64)):
    """
    calculate initial cost volume according to given census responses:
    calculate hamming distance to get cost volume

    :param left_census: left census response
    :param right_census: right census response
    :param disparity_range: range of disparity
    :return: pixel-wise disparity [X, Y, D]
    """

    img_h, img_w, _ = left_census.shape  # h, w, census_vector_len
    min_disparity, max_disparity = disparity_range
    range_disparity = max_disparity - min_disparity
    cost_volume = range_disparity * np.ones((img_h, img_w, range_disparity), dtype=int)

    for i in range(img_h):
        print('calculate_init_cost_volume: i=[{}/{} : {:.3f}]'.format(i, img_h, i / img_h))
        for j in range(img_w):
            # pick up center pixel census
            left_pixel_census = left_census[i][j]
            for d in range(min_disparity, max_disparity):  # line search on right census
                if j - d < 0:
                    continue
                else:
                    # -d: 原理见 视差-深度转换 这是因为右视图的同名点，不可能出现在xl的右边
                    right_pixel_census = right_census[i][j - d]
                    cost_volume[i][j][d] = calculate_hamming(left_pixel_census, right_pixel_census)
    return cost_volume


def calculate_aggregation(init_cost_volume):
    """
    calculate aggregation
    聚合代价=原始匹配代价+平滑项
    input: initial cost volume [X, Y, D]
    output: cost volume after aggregation [X, Y, D]

    # NOTE: a lot of redundancy! For example: when we get N-1 col's aggregation,
    the N-2 col's aggregation have been calculated, BUT when we want to calculate
    the N-th col's aggregation, the N-2 col calculated twice!
    SOLUTION: save the N-2 col's value as tmp
    """

    # penalty
    p1 = 1
    p2 = 1

    volume_size = init_cost_volume.shape  # [X, Y, D]
    aggregation_volume = np.zeros(volume_size)

    # update cost volume
    for i in range(1, volume_size[0]-1):  # [H] each row
        if i % 5 == 0:
            print('calculate_aggregation_volume: i=[{}/{} : {:.3f}]'.format(i, volume_size[0], i / volume_size[1]))

        for j in range(volume_size[1]):  # [W] each col
            # p: [i, j]
            # left[i, 0] -> p[i, j] path
            if j == 0:
                aggregation_volume[i][j] = init_cost_volume[i][j]  # init the first col
            elif j == volume_size[1]-1:
                aggregation_volume[i][j] = init_cost_volume[i][j]
            elif 0 < j < volume_size[1] - 1:
                aggregation_volume[i][j][0] = init_cost_volume[i][j][0]
                aggregation_volume[i][j][volume_size[2] - 1] = init_cost_volume[i][j][volume_size[2] - 1]

                for d in range(1, volume_size[2] - 1):  # [D] for each disparity
                # for d in range(0, volume_size[2]):  # [D] for each disparity
                    # Q: the aggregation volume's logic
                    # how to check? in one world, i'm just not familiar with it currently
                    # left->right,is j-1 -> j
                    aggregation_volume[i][j][d] = (init_cost_volume[i][j][d] +
                                                   # init_cost_volume[i][j][d-1] +
                                                   # init_cost_volume[i][j][d+1] +
                                                   init_cost_volume[i-1][j][d] +
                                                   init_cost_volume[i+1][j][d] +
                                                   init_cost_volume[i][j-1][d] +
                                                   init_cost_volume[i][j+1][d])/5  # \
                                              # min(aggregation_volume[i][j - 1][d],
                                              #     aggregation_volume[i][j - 1][d - 1] + 1,  # d_disparity=-1 + p1,
                                              #     aggregation_volume[i][j - 1][d + 1] + 1,  # d_disparity=1 + p1,
                                              #     min(aggregation_volume[i][j - 1] + 10)
                                              #     )
                # print()

    return aggregation_volume


# get initial cost volume from census
# def calculate_disparity(left_census, right_census, disparity_range=(0, 64)):
def calculate_disparity(cost_volume, disparity_range=(0, 64)):
    """
    calculate disparity according to given census responses:
    calculate hamming distance to get cost volume
    and then perform winner-take-all

    视差计算的过程:
    根据local cost的结果，给定一个source image pixel， 在dst image的视差范围内遍历 执行winner-take-all
    [这是可以*并行*或者是*基于学习的方式*改进的地方]

    :param disparity_range: range of disparity
    :return: pixel-wise disparity [X, Y]
    """
    img_h, img_w, _ = cost_volume.shape  # H, W, D
    disparity_res = np.zeros((img_h, img_w), dtype=int)

    for i in range(img_h):
        if i % 10 == 0:
            print('calculate_disparity: i=[{}/{} : {:.3f}]'.format(i, img_h, i / img_h))
        for j in range(img_w):
            # winner-take-all performed by arg-min operator
            disparity_res[i][j] = np.argmin(cost_volume[i][j])
    return disparity_res


# post processing
def lr_check():
    return 0


"""
transform disparity into depth
process triangulation
"""


def disparity_to_depth(disparity_image, baseline=193, focal_length=100):
    image_size = disparity_image.shape
    depth_image = np.zeros(image_size)
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            depth_image[i][j] = np.clip(baseline * focal_length / (disparity_image[i][j] + 1e-10), a_min=0, a_max=1000)
    return depth_image


def color_depth_to_pcd(depth_image, rgb_image, pcd_path, focal_length=100):
    # TODO: get real XY according to depth
    img_size = depth_image.shape
    num_point = img_size[0] * img_size[1]
    cx, cy = img_size[1]/2, img_size[0]/2
    # i: row
    # j: col
    with open(pcd_path, 'w') as f:
        ply_head = "ply  \n" + \
                   "format ascii 1.0  \n" + \
                   "element vertex {}\n".format(num_point) + \
                   "property float x  \n" + \
                   "property float y  \n" + \
                   "property float z  \n" + \
                   "property uchar blue  \n" + \
                   "property uchar green  \n" + \
                   "property uchar red  \n" + \
                   "end_header  \n"
        f.write(ply_head)

        # "property uchar red  \n" + \
        # "property uchar green  \n" + \
        # "property uchar blue  \n" +

        for i in range(img_size[0]):  # H
            for j in range(img_size[1]):  # W
                depth = depth_image[i][j]
                X = depth / focal_length * (j - cx)
                Y = depth / focal_length * (i - cy)
                xyz_rgb = '{} {} {} {} {} {}\n'.format(X, Y, depth,
                                                       rgb_image[i][j][0],
                                                       rgb_image[i][j][1],
                                                       rgb_image[i][j][2]
                                                       )
                f.write(xyz_rgb)


if __name__ == '__main__':

    # run-time parameter
    is_calculate_cost_volume = 0
    is_calculate_aggregation = 1

    # pic_name = 'Woold2'
    pic_name = 'cones'

    # read images
    if pic_name == 'Woold2':
        img_left_path = 'StereoDataset/Wood2/view1.png'
        img_right_path = 'StereoDataset/Wood2/view5.png'
    elif pic_name == 'cones':
        img_left_path = 'StereoDataset/cones/im2.png'
        img_right_path = 'StereoDataset/cones/im6.png'
    elif pic_name == 'Cloth3':
        img_left_path = 'StereoDataset/Cloth3/view1.png'
        img_right_path = 'StereoDataset/Cloth3/view5.png'
    else:
        raise KeyError('ERROR PIC_NAME: {}'.format(pic_name))

    assert os.path.exists(img_left_path)
    assert os.path.exists(img_right_path)

    img_left = cv2.imread(img_left_path, 0)
    img_right = cv2.imread(img_right_path, 0)
    img_left_color = cv2.imread(img_left_path)
    assert img_left.shape == img_right.shape
    print('image_size: {}'.format(img_left.shape))

    # 1. calculate census
    print('calculate census ... ')
    # census_size = (3, 3)
    census_size = (5, 5)
    # census_size = (7, 5)
    # census_size = (9, 7)
    # census_size = (11, 9)

    if is_calculate_cost_volume:
        census_left = calculate_census(img_left, census_size=census_size)
        census_right = calculate_census(img_right, census_size=census_size)
    else:
        pass

    disparity_range = (0, 64)

    # get init cost volume
    print('calculate init cost volume ... ')
    init_volume_path = 'tmp/' + pic_name + '_init_volume.npy'
    if is_calculate_cost_volume:
        init_cost_volume = calculate_init_cost_volume(left_census=census_left, right_census=census_right,
                                                      disparity_range=disparity_range)
        np.save(init_volume_path, init_cost_volume)
    else:
        init_cost_volume = np.load(init_volume_path)

    # 2. calculate aggregation
    if is_calculate_aggregation:
        print('calculate aggregation ... ')
        cost_volume_aggregation = calculate_aggregation(init_cost_volume=init_cost_volume)
    else:
        cost_volume_aggregation = init_cost_volume

    # 3. calculate disparity
    print('calculate disparity ... ')
    disparity_image = calculate_disparity(cost_volume=cost_volume_aggregation, disparity_range=disparity_range)

    # show disparity
    plt.imshow(disparity_image)
    plt.title('agg_census:{}-{}'.format(census_size[0], census_size[1]))
    plt.show()

    # 4. disparity refinement [post processing ]
    # print('disparity refine ... ')

    # to rgb-depth
    depth_image = disparity_to_depth(disparity_image=disparity_image)
    # show depth
    plt.imshow(depth_image)
    plt.title('depth:{}-{}'.format(census_size[0], census_size[1]))
    plt.show()

    depth_image_name = 'Result/{}_depth.npy'.format(pic_name)
    np.save(depth_image_name, depth_image)
    # cv2.imwrite('Result/, depth_image)

    # to point cloud
    print('saving to point cloud ... ')
    pcd_path = 'Result/pointcloud.ply'
    color_depth_to_pcd(depth_image=depth_image, rgb_image=img_left_color, pcd_path=pcd_path)
