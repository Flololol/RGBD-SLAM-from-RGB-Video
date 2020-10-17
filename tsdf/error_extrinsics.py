import numpy as np
from pose_refiner import pose_refiner
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

peter = True
size = (640, 480)
use_opt = False
global_align = False
stride = 1

class ICP:
    def __init__(self, source, target):
        assert source.shape == target.shape
        self.source = source
        self.target = target

    def alignment_error(self, unknowns, source_trans, target_trans):
        scale = unknowns[0]
        rot_euler = unknowns[1:4]
        roti = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
        t = unknowns[4:]

        error = np.empty((source_trans.shape[0], 1))
        for i in range(source_trans.shape[0]):
            error[i] = np.sum((target_trans[i] - scale * roti.dot(source_trans[i]) - t)**2)

        error = np.sum(error)

        return error
    
    def fit(self):
        unknowns = np.array([1, 0, 0, 0, 0, 0, 0])
        bnds = [(0,None),(0,360),(0,360),(0,360),(None,None),(None,None),(None,None)]
        # res = minimize(self.alignment_error, unknowns, args=(self.source[:,:3,3], self.target[:,:3,3]), method = 'Nelder-Mead', options={"disp":True, "maxiter":50000, "xatol":1e-10, "fatol":1e-10}).x
        res = minimize(self.alignment_error, unknowns, bounds=bnds, args=(self.source[:,:3,3], self.target[:,:3,3]), options={"disp":True, "maxiter":50000}).x
        align_scale = res[0]
        align_rot = res[1:4]
        align_rot = R.from_euler('xyz', align_rot, degrees=True).as_matrix()
        align_trans = res[4:]
        print('found best values for alignment: {}'.format(res))

        for i in range(self.source.shape[0]):
            self.source[i,:3,3] = align_scale*align_rot.dot(self.source[i,:3,3])+align_trans
            self.source[i,:3,:3] = align_rot.dot(self.source[i,:3,:3])

        return align_scale, align_rot, align_trans

if __name__ == "__main__":
    extr_opt = "extrinsics_opt_nm_lrkt2"
    # extr_opt = "extrinsics_opt_nm_OPT"
    extr_opt = "./{}.npz".format(extr_opt)
    
    base_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/room/"
    depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/"

    if peter:
        base_dir = "/home/noxx/Documents/projects/consistent_depth/results/lr_kt2_flo/"
        depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/"

    color_dir = base_dir+"color_full/"
    metadata = base_dir+"R_hierarchical2_mc/"
    
    refiner = pose_refiner(color_dir, depth_dir, metadata, size=size, GRND_TRTH=True)
    refiner.load_data()

    if use_opt:
        with np.load(extr_opt) as extr_opt:
            extr_ours = extr_opt["extrinsics_opt"]
        stride = int((refiner.extrinsics.shape[0])/extr_ours.shape[0]+1)
        print('calculated stride {}'.format(stride))
        refiner.resize_stride(stride)
        extr_truth = refiner.extrinsics_truth
    else:
        refiner.resize_stride(stride)
        extr_ours = refiner.extrinsics
        extr_truth = refiner.extrinsics_truth

    print('extr_ours.shape: {}'.format(extr_ours.shape))
    print('extr_truth.shape: {}'.format(extr_truth.shape))

    if extr_ours.shape[0] != extr_truth.shape[0]:
        min_n = np.min([extr_ours.shape[0], extr_truth.shape[0]])
        extr_ours = extr_ours[:min_n]
        extr_truth = extr_truth[:min_n]

    if global_align:
        icp = ICP(extr_ours, extr_truth)
        icp.fit()
        extr_ours = icp.source
    else:
        rotation_fix = extr_truth[0,:3,:3].dot(np.linalg.inv(extr_ours[0,:3,:3]))
        for i in range(refiner.N):
            extr_ours[i,:3,:3] = rotation_fix.dot(extr_ours[i,:3,:3])
        
        #fix translation offset by moving both initial images to the origin
        extr_ours[:,:3,3] = extr_ours[:,:3,3] - extr_ours[0,:3,3]
        extr_truth[:,:3,3] = extr_truth[:,:3,3] - extr_truth[0,:3,3]

        align_scale = np.mean(abs(extr_truth[:,:3,3])) / np.mean(abs(extr_ours[:,:3,3]))
        print('best scale: {}'.format(align_scale))

        extr_ours[:,:3,3] = align_scale*extr_ours[:,:3,3]
    
    extra_row = np.zeros((refiner.N,1,4))
    extra_row[:,0,3] = 1

    cam_t = np.empty((refiner.N, 4))
    cam_o = np.empty((refiner.N, 4))
    points_t = np.empty((refiner.N, 2, 4))
    points_o = np.empty((refiner.N, 2, 4))
    extr_ours = np.concatenate((extr_ours, extra_row), axis=1)
    extr_truth = np.concatenate((extr_truth, extra_row), axis=1)
    for i in range(refiner.N):
        #ground truth
        cam_t[i] = extr_truth[i].dot(np.array([0,0,0,1]))
        cam_t[i] /= cam_t[i,3]
        points_t[i,0] = extr_truth[i].dot(np.array([0,0,0.2,1]))
        points_t[i,0] /= points_t[i,0,3]
        points_t[i,1] = extr_truth[i].dot(np.array([0,0,0.8,1]))
        points_t[i,1] /= points_t[i,1,3]
        #ours
        cam_o[i] = extr_ours[i].dot(np.array([0,0,0,1]))
        cam_o[i] /= cam_o[i,3]
        points_o[i,0] = extr_ours[i].dot(np.array([0,0,0.2,1]))
        points_o[i,0] /= points_o[i,0,3]
        points_o[i,1] = extr_ours[i].dot(np.array([0,0,0.8,1]))
        points_o[i,1] /= points_o[i,1,3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #camera locations for true extrinsics
    ax.plot(cam_t[0,0], cam_t[0,1], cam_t[0,2], 'ro', markersize=6)
    ax.plot(cam_t[1:,0], cam_t[1:,1], cam_t[1:,2], 'go', markersize=4)

    #camera locations for our extrinsics
    ax.plot(cam_o[0,0], cam_o[0,1], cam_o[0,2], 'ro', markersize=6)
    ax.plot(cam_o[1:,0], cam_o[1:,1], cam_o[1:,2], 'bo', markersize=4)

    # #point 1 & 2 for true extrinsics
    ax.plot(points_t[0,0,0], points_t[0,0,1], points_t[0,0,2], 'rx', markersize=6)
    # ax.plot(points_t[1:,0,0], points_t[1:,0,1], points_t[1:,0,2], 'gx', markersize=4)
    ax.plot(points_t[0,1,0], points_t[0,1,1], points_t[0,1,2], 'rx', markersize=6)
    # ax.plot(points_t[1:,1,0], points_t[1:,1,1], points_t[1:,1,2], 'gx', markersize=4)

    # #points 1 & 2 for our extrinsics
    ax.plot(points_o[0,0,0], points_o[0,0,1], points_o[0,0,2], 'yx', markersize=6)
    # ax.plot(points_o[:,0,0], points_o[:,0,1], points_o[:,0,2], 'bx', markersize=4)
    ax.plot(points_o[0,1,0], points_o[0,1,1], points_o[0,1,2], 'yx', markersize=6)
    # ax.plot(points_o[:,1,0], points_o[:,1,1], points_o[:,1,2], 'bx', markersize=4)

    # ax.quiver(cam_loc[:,0], cam_loc[:,1], cam_loc[:,2],point_cloud[:,0],point_cloud[:,1],point_cloud[:,2], length=1.0)
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-2, 1])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()
