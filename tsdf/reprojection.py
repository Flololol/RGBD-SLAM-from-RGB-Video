from pose_refiner import pose_refiner
import numpy as np

peter = False
img1_idx = 0
img2_idx = 20
size = (384, 224)

base_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/"
depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/"
if peter:
    base_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/"
    depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/"

color_dir = base_dir+"color_down_png/"
# color_dir = base_dir+"color_full/"
metadata = base_dir+"R_hierarchical2_mc/"

def reprojection_pair(Tij, pi, diks, djks, shp):
    print(shp)
    diks = diks.reshape(-1,3)
    diks = np.concatenate((diks, np.ones((diks.shape[0],1))), axis=1)
    
    left = np.tensordot(Tij, diks, axes=([1],[1])).T[:,:3]
    right = np.tensordot(pi, left, axes=([1],[1])).T

    left = left.reshape(shp[0],shp[1],3)
    right = right.reshape(shp[0],shp[1],3)

    xLim = np.logical_and(right[:,:,0]>0, right[:,:,0]<shp[0])
    yLim = np.logical_and(right[:,:,1]>0, right[:,:,1]<shp[1])
    lim = np.logical_and(xLim, yLim)

    err = 0
    for x in range(shp[1]):
        for y in range(shp[0]):
            if lim[y,x]:
                pos = right[y,x]
                imgX = int(pos[0])
                imgY = int(pos[1])
                lft = left[y,x]
                rgt = right[imgY, imgX]
                err += np.linalg.norm(lft-rgt)
    return err


refiner = pose_refiner(color_dir, depth_dir, metadata, size=size)
refiner.load_data()

img1 = refiner.RGB[img1_idx]
dpt1 = refiner.depth[img1_idx]
T1 = np.vstack((refiner.extrinsics[img1_idx], np.array([0,0,0,1])))

img2 = refiner.RGB[img2_idx]
dpt2 = refiner.depth[img2_idx]
T2 = np.vstack((refiner.extrinsics[img2_idx], np.array([0,0,0,1])))
T2_inv = np.linalg.inv(T2)

T12 = T2_inv.dot(T1)

diks = (refiner.diks * refiner.depth[img1_idx, :, :, np.newaxis])
djks = (refiner.diks * refiner.depth[img2_idx, :, :, np.newaxis])

error = reprojection_pair(T12, refiner.intrinsics, diks, djks, img1.shape)

print(error)