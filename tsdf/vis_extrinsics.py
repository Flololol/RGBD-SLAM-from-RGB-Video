import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pose_refiner import pose_refiner

peter = False
include_opt = False
size = (1920, 1080)

if __name__ == "__main__":
    eps_euler = .01 #x degree step size in terms of rotation
    eps_translation = .0005 #this is a relative value that depends on the depth scale refiner.scale
    extr_opt = "extrinsics_opt_{}_{}".format(eps_euler, eps_translation)
    # extr_opt = "extrinsics_opt_0.01_0.0005_NO_EPS"
    extr_opt = "./{}.npz".format(extr_opt)

    base_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/desk/"
    depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/"
    if peter:
        base_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/"
        depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/"

    # color_dir = base_dir+"color_down_png/"
    color_dir = base_dir+"color_full/"
    metadata = base_dir+"R_hierarchical2_mc/"

    refiner = pose_refiner(color_dir, depth_dir, metadata, size=size)
    refiner.load_data()

    extrinsics = refiner.extrinsics
    old_N = refiner.extrinsics.shape[0]

    if include_opt:
        with np.load(extr_opt) as extr_opt:
            extrinsics_opt = extr_opt["extrinsics_opt"]
        extrinsics = np.concatenate((refiner.extrinsics, extrinsics_opt), axis=0)

    cam_loc = np.empty((np.shape(extrinsics)[0], 4))
    point_cloud = np.empty((np.shape(extrinsics)[0], 2, 4))
    extra_row = np.zeros((extrinsics.shape[0],1,4))
    extra_row[:,0,3] = 1
    extrinsics = np.concatenate((extrinsics, extra_row), axis=1)
    for i in range(extrinsics.shape[0]):
        cam_loc[i] = extrinsics[i].dot(np.array([0,0,0,1]))
        cam_loc[i] /= cam_loc[i,3]

        point_cloud[i,0] = extrinsics[i].dot(np.array([0,0,0.2,1]))
        point_cloud[i,0] /= point_cloud[i,0,3]

        point_cloud[i,1] = extrinsics[i].dot(np.array([0,0,0.8,1]))
        point_cloud[i,1] /= point_cloud[i,1,3]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #camera locations for original extrinsics
    ax.plot(cam_loc[0,0], cam_loc[0,1], cam_loc[0,2], 'ro', markersize=6)
    ax.plot(cam_loc[:old_N,0], cam_loc[:old_N,1], cam_loc[:old_N,2], 'gx', markersize=5)

    #camera locations for optimized extrinsics
    if include_opt:
        ax.plot(cam_loc[old_N,0], cam_loc[old_N,1], cam_loc[old_N,2], 'rx', markersize=8)
        ax.plot(cam_loc[old_N:,0], cam_loc[old_N:,1], cam_loc[old_N:,2], 'yx', markersize=7)

    #point 1 for original extrinsics
    ax.plot(point_cloud[0,0,0], point_cloud[0,0,1], point_cloud[0,0,2], 'ro', markersize=6)
    ax.plot(point_cloud[:old_N,0,0], point_cloud[:old_N,0,1], point_cloud[:old_N,0,2], 'bo', markersize=3)

    #point 2 for original extrinsics
    ax.plot(point_cloud[0,1,0], point_cloud[0,1,1], point_cloud[0,1,2], 'ro', markersize=6)
    ax.plot(point_cloud[:old_N,1,0], point_cloud[:old_N,1,1], point_cloud[:old_N,1,2], 'bo', markersize=3)

    #points 1 & 2 for optimized extrinsics
    if include_opt:
        ax.plot(point_cloud[old_N:,0,0], point_cloud[old_N:,0,1], point_cloud[old_N:,0,2], 'yo', markersize=4)
        ax.plot(point_cloud[old_N:,1,0], point_cloud[old_N:,1,1], point_cloud[old_N:,1,2], 'yo', markersize=4)

    # ax.quiver(cam_loc[:,0], cam_loc[:,1], cam_loc[:,2],point_cloud[:,0],point_cloud[:,1],point_cloud[:,2], length=1.0)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()
