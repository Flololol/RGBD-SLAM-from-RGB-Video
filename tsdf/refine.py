import numpy as np
from pose_refiner import pose_refiner

stride = 10
fresh = False
peter = False

if __name__ == "__main__":
    eps_euler = .2 #x degree step size in terms of rotation
    eps_translation = .005 #this is a relative value that depends on the depth scale refiner.scale
    file_name = "extrinsics_opt_{}_{}".format(eps_euler, eps_translation)
    print('optimizing to "{}.npz"'.format(file_name))

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

    print('total energy with x0: {}'.format(refiner.total_energy_mt(refiner.extrinsics_euler)))
    extrinsics_opt = refiner.optim(eps_euler, eps_translation, maxIter=5)
    print('total energy after opt: {}'.format(refiner.total_energy_mt(refiner.extrinsics_euler_opt)))

    # COL = np.diag([1, -1, -1])
    # for i in range(refiner.N):
    #     extrinsics_opt[i,:3,3] = extrinsics_opt[i,:3,3]*refiner.scale

    #     extrinsics_opt[i,:3,:3] = COL.dot(extrinsics_opt[i,:3,:3]).dot(COL.T)
    #     extrinsics_opt[i,:3,3] = COL.dot(extrinsics_opt[i,:3,3])

    np.savez(file_name, extrinsics_opt=extrinsics_opt)
    print('saved optimized extrinsics to "{}.npz"'.format(file_name))
