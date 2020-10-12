import numpy as np
from pose_refiner import pose_refiner

stride = 10
fresh = False
peter = True

if __name__ == "__main__":
    base_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/"
    depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/depth/"
    if peter:
        base_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/"
        depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/depth/"

    color_dir = base_dir+"color_down_png/"
    # color_dir = base_dir+"color_full/"
    metadata = base_dir+"R_hierarchical2_mc/metadata_scaled.npz"

    refiner = pose_refiner(color_dir, depth_dir, metadata)
    refiner.fresh = fresh
    refiner.prepare()
    refiner.resize_stride(stride)

    print('total energy with x0: {}'.format(refiner.total_energy_mt(refiner.extrinsics)))
    extrinsics_opt = refiner.optim()
    print('total energy after opt: {}'.format(refiner.total_energy_mt(refiner.extrinsics_opt)))

    # COL = np.diag([1, -1, -1])
    # for i in range(refiner.N):
    #     extrinsics_opt[i,:3,3] = extrinsics_opt[i,:3,3]*refiner.scale

    #     extrinsics_opt[i,:3,:3] = COL.dot(extrinsics_opt[i,:3,:3]).dot(COL.T)
    #     extrinsics_opt[i,:3,3] = COL.dot(extrinsics_opt[i,:3,3])

    np.savez('extrinsics_opt', extrinsics_opt=extrinsics_opt)
