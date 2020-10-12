import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import struct
from PIL import Image

def resize_intrinsics(intrinsics, size_old, size_new):
    fx, fy, cx, cy = intrinsics[0]
    ratio = np.array(size_new) / np.array(size_old)
    fx *= ratio[0]
    fy *= ratio[1]
    cx *= ratio[0]
    cy *= ratio[1]
    return fx, fy, cx, cy

peter = True

# color_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/color_down_png/"
color_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/color_full/"
depth_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/depth/"
metadata = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/metadata_scaled.npz"
metad = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/colmap_dense/metadata.npz"
size_new = (1280, 720)

if peter:
    color_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/color_down_png/"
    color_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/color_full/"
    depth_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/depth/"
    metadata = "/home/noxx/Documents/projects/consistent_depth/results/debug03/R_hierarchical2_mc/metadata_scaled.npz"
    metad = "/home/noxx/Documents/projects/consistent_depth/results/debug03/colmap_dense/metadata.npz"
    size_new = (1920, 1080)

extr_opt = "./extrinsics_opt.npz"

size_old = (384, 224)
fmt = "frame_{:06d}.png"
fmt_raw = "frame_{:06d}.raw"

with np.load(metadata) as meta_colmap:
    intrinsics = meta_colmap["intrinsics"]
    extrinsics = meta_colmap["extrinsics"]
    scales = meta_colmap["scales"]

with np.load(extr_opt) as extr_opt:
    extrinsics_opt = extr_opt["extrinsics_opt"]

old_N = extrinsics.shape[0]
extrinsics = np.concatenate((extrinsics, extrinsics_opt), axis=0)

scale = scales[:,1].mean()
print("mean scale: {}".format(scale))
# with np.load(metad) as meta_colmap:
#     extrinsics = meta_colmap["extrinsics"]
# extrinsics[:,:,-1] /= scale**2 #warum müssen wir die normalen extrinsics 2 mal durch die scale teilen damit es passt?

fx, fy, cx, cy = resize_intrinsics(intrinsics, size_old, size_new)
intr = o3d.camera.PinholeCameraIntrinsic(*size_new, fx, fy, cx, cy)
print("-----------")
print('initial cam pos, unmodified: {}'.format(extrinsics[0,:3,3]))

COL = np.diag([1, -1, -1])
cam_loc = np.empty((np.shape(extrinsics)[0], 4))
point_cloud = np.empty((np.shape(extrinsics)[0], 2, 4))
extra_row = np.zeros((extrinsics.shape[0],1,4))
extra_row[:,0,3] = 1
extrinsics = np.concatenate((extrinsics, extra_row), axis=1)
for i in range(extrinsics.shape[0]):
    extrinsics[i,:3,:3] = COL.dot(extrinsics[i,:3,:3]).dot(COL.T)
    extrinsics[i,:3,3] = COL.dot(extrinsics[i,:3,3])

    extrinsics[i,:3,3] = extrinsics[i,:3,3]/scale

    extrinsics[i] = np.linalg.inv(extrinsics[i])

    cam_loc[i] = np.linalg.inv(extrinsics[i]).dot(np.array([0,0,0,1]))
    cam_loc[i] /= cam_loc[i,3]
    point_cloud[i,0] = np.linalg.inv(extrinsics[i]).dot(np.array([0,0,0.2,1]))
    point_cloud[i,0] /= point_cloud[i,0,3]
    point_cloud[i,1] = np.linalg.inv(extrinsics[i]).dot(np.array([0,0,0.8,1]))
    point_cloud[i,1] /= point_cloud[i,1,3]

print('initial cam pos, modified: {}'.format(np.linalg.inv(extrinsics[0])[:3,3]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(cam_loc[0,0], cam_loc[0,1], cam_loc[0,2], 'ro', markersize=6)
ax.plot(cam_loc[old_N,0], cam_loc[old_N,1], cam_loc[old_N,2], 'rx', markersize=8)
ax.plot(cam_loc[:old_N,0], cam_loc[:old_N,1], cam_loc[:old_N,2], 'gx', markersize=5)
ax.plot(cam_loc[old_N:,0], cam_loc[old_N:,1], cam_loc[old_N:,2], 'yx', markersize=7)
ax.plot(point_cloud[0,0,0], point_cloud[0,0,1], point_cloud[0,0,2], 'ro', markersize=6)
ax.plot(point_cloud[:old_N,0,0], point_cloud[:old_N,0,1], point_cloud[:old_N,0,2], 'bo', markersize=3)
ax.plot(point_cloud[old_N:,0,0], point_cloud[old_N:,0,1], point_cloud[old_N:,0,2], 'yo', markersize=4)
ax.plot(point_cloud[0,1,0], point_cloud[0,1,1], point_cloud[0,1,2], 'ro', markersize=6)
ax.plot(point_cloud[:old_N,1,0], point_cloud[:old_N,1,1], point_cloud[:old_N,1,2], 'bo', markersize=3)
ax.plot(point_cloud[old_N:,1,0], point_cloud[old_N:,1,1], point_cloud[old_N:,1,2], 'yo', markersize=4)
# ax.quiver(cam_loc[:,0], cam_loc[:,1], cam_loc[:,2],point_cloud[:,0],point_cloud[:,1],point_cloud[:,2], length=1.0)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.show()
