from refine import pose_refiner
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    peter = True

    base_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/"
    if peter:
        base_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/"

    color_dir = base_dir+"color_down_png/"
    # color_dir = base_dir+"color_full/"
    depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/depth/"
    metadata = base_dir+"R_hierarchical2_mc/metadata_scaled.npz"

    
    img2_idx = 7
    # size_old = (6,4)
    size_old = (384, 224)
    refiner = pose_refiner(color_dir, depth_dir, metadata, size=size_old)
    refiner.load_data()

    fmt = "frame_{:06d}.png"
    fmt_raw = "frame_{:06d}.raw"

    #comparing frame 0 to img2_idx for testing
    img1 = refiner.RGB[0]
    dpt1 = refiner.depth[0]
    T1 = np.vstack((refiner.extrinsics[0], np.array([0,0,0,1])))

    img2 = refiner.RGB[img2_idx]
    dpt2 = refiner.depth[img2_idx]
    T2 = np.vstack((refiner.extrinsics[img2_idx], np.array([0,0,0,1])))

    transformed = np.zeros_like(img1)

    print("transform..")
    for x in range(size_old[0]):
        for y in range(size_old[1]):
            curDepth = dpt1[y, x]
            curRGB = img1[y, x]
            # print(curRGB)
            pos = np.array([x, y, 1])
            # print(pos)
            # print(pxyz[y,x])
            # dik = np.linalg.inv(refiner.intrinsics).dot(pos)
            # print(dik)
            # dik *= curDepth
            # print(dik)
            dik = refiner.diks[y, x] * curDepth
            # print(dik)
            # dik *= curDepth
            # print(dik)
            # dik /= dik[-1]
            dik = np.append(dik, 1)
            # print(dik)
            tgt = np.linalg.inv(T2).dot(T1.dot(dik))
            tgt = np.delete(tgt, 3)
            # print(tgt)
            imgPos = refiner.intrinsics.dot(tgt)
            imgPos = imgPos / imgPos[-1]
            # print(imgPos)
            imgX = int(imgPos[0])
            imgY = int(imgPos[1])
            if imgX > 0 and imgX < img2.shape[1]:
                if imgY > 0 and imgY < img2.shape[0]:
                    transformed[imgY, imgX] = curRGB
                    # img2[imgY, imgX] = curRGB
            # exit()

    plt.imshow(img1)
    plt.show()
    plt.imshow(transformed)
    plt.show()
    plt.imshow(img2)
    plt.show()