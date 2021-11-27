"""
Author: ANTenna on 2021/11/25 8:42 下午
aliuyaohua@gmail.com

Description:

"""

import open3d as o3d


if __name__ == '__main__':
    pcd_path = 'Result/pointcloud.ply'

    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd],
                                      window_name='ANTenna3D',)
