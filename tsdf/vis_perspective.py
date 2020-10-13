from pose_refiner import pose_refiner
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img1_idx = 0
img2_idx = 20
size = (384, 224)

if __name__ == "__main__":
    peter = False

    base_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/"
    depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/depth/"
    if peter:
        base_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/"
        depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/depth/"

    color_dir = base_dir+"color_down_png/"
    # color_dir = base_dir+"color_full/"
    metadata = base_dir+"R_hierarchical2_mc/metadata_scaled.npz"

    refiner = pose_refiner(color_dir, depth_dir, metadata, size=size)
    refiner.load_data()

    #comparing frame 0 to img2_idx for testing
    img1 = refiner.RGB[img1_idx]
    dpt1 = refiner.depth[img1_idx]
    T1 = np.vstack((refiner.extrinsics[img1_idx], np.array([0,0,0,1])))

    img2 = refiner.RGB[img2_idx]
    dpt2 = refiner.depth[img2_idx]
    T2 = np.vstack((refiner.extrinsics[img2_idx], np.array([0,0,0,1])))
    T2_inv = np.linalg.inv(T2)

    transformed = np.zeros_like(img1)

    #vectorized diks transformation
    diks = (refiner.diks * dpt1[:, :, np.newaxis]).reshape(-1,3)
    tmp = np.ones((diks.shape[0]))
    diks = np.concatenate((diks, tmp[:,np.newaxis]), axis=1)
    diks = np.tensordot(T1,diks,axes=([1],[1])).T
    diks = np.tensordot(T2_inv, diks, axes=([1],[1])).T
    diks = np.tensordot(refiner.intrinsics, diks[:,:3], axes=([1],[1])).T
    diks = (diks / diks[:,2][:,np.newaxis]).reshape(img1.shape)

    xLim = np.logical_and(diks[:,:,0]>0, diks[:,:,0]<size[0])
    yLim = np.logical_and(diks[:,:,1]>0, diks[:,:,1]<size[1])
    lim = np.logical_and(xLim, yLim)
    frac = np.sum(lim) / np.multiply(*size)
    print(frac)
    print(lim.shape)
    
    print("transform..")
    for x in range(size[0]):
        for y in range(size[1]):
            if lim[y,x]:
                pos = diks[y,x]
                imgX = int(pos[0])
                imgY = int(pos[1])
                transformed[imgY, imgX] = img1[y, x]
            
            # curDepth = dpt1[y, x]
            # curRGB = img1[y, x]
            # dik = refiner.diks[y, x] * curDepth

            # dik = np.append(dik, 1)
            # print(dik)
            # print(T1.dot(dik))
            # testT1[y,x] = T1.dot(dik)
            # tgt = np.linalg.inv(T2).dot(T1.dot(dik))
            # testT2[y,x] = tgt
            # print(tgt)
            # exit()
            # tgt = np.delete(tgt, 3)
            # print(tgt)
            # imgPos = refiner.intrinsics.dot(tgt)
            # testPos[y,x] = imgPos
            # imgPos = imgPos / imgPos[-1]
            # testScales[y,x] = imgPos
            # print(imgPos)
            # imgX = int(imgPos[0])
            # imgY = int(imgPos[1])
            # if imgX > 0 and imgX < img2.shape[1]:
            #     if imgY > 0 and imgY < img2.shape[0]:
            #         transformed[imgY, imgX] = curRGB
            #         img2[imgY, imgX] = curRGB
            # exit()
    # testT1 = np.array(testT1)
    # testT2 = np.array(testT2)
    # print(testT1.shape, testT2.shape)
    # print(np.allclose(result, testT1))
    # print(np.allclose(result2, testT2))
    # print(np.allclose(result3, testPos))
    # print(np.allclose(result4, testScales))
    # exit()
    plt.imshow(img1)
    plt.show()
    plt.imshow(transformed)
    plt.show()
    plt.imshow(img2)
    plt.show()