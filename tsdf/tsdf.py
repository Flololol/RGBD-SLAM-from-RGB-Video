import open3d as o3d
import numpy as np
import struct
from PIL import Image
from tqdm import tqdm
from pose_refiner import pose_refiner
from error_extrinsics import ICP
from error_depth import DepthScale

peter = True
use_extr = 'our'
# use_extr = 'opt'
# use_extr = 'tru'
use_depth = 'our'
# use_depth = 'tru'
img1_idx = 0
size = (640, 480)
CUT = False
CUT_N = 1
stride = 1

if __name__ == "__main__":
    eps_euler = .01 #x degree step size in terms of rotation
    eps_translation = .0005 #this is a relative value that depends on the depth scale refiner.scale
    extr_opt = "extrinsics_opt_{}_{}".format(eps_euler, eps_translation)
    # extr_opt = "extrinsics_opt_nelder_mead_it5000"
    extr_opt = "./{}.npz".format(extr_opt)

    base_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/room/"
    depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/"
    if peter:
        base_dir = "/home/noxx/Documents/projects/consistent_depth/results/room01/"
        depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/"

    # color_dir = base_dir+"color_down_png/"
    color_dir = base_dir+"color_full/"
    metadata = base_dir+"R_hierarchical2_mc/"


    truth_relevant = (use_depth=='tru' or use_extr=='tru')
    refiner = pose_refiner(color_dir, depth_dir, metadata, size=size, GRND_TRTH=truth_relevant)
    refiner.load_data()

    if use_extr == 'opt':
        with np.load(extr_opt) as extr_opt:
            extrinsics_opt = extr_opt["extrinsics_opt"]
        stride = int(refiner.extrinsics.shape[0]/extrinsics_opt.shape[0]+1)
        refiner.resize_stride(stride)
        if truth_relevant:
            icp = ICP(extrinsics_opt, refiner.extrinsics_truth)
            icp.fit()
            extrinsics = icp.source
        else:
            extrinsics = extrinsics_opt
    elif use_extr == 'tru':
        refiner.resize_stride(stride)
        extrinsics = refiner.extrinsics_truth
    else:
        refiner.resize_stride(stride)
        if truth_relevant:
            icp = ICP(refiner.extrinsics, refiner.extrinsics_truth)
            icp.fit()
            extrinsics = icp.source
        else:
            extrinsics = refiner.extrinsics
    
    if use_depth == 'tru':
        refiner.depth_truth = refiner.depth_truth.astype(np.float32) #o3d doesnt like doubles
        depth = refiner.depth_truth
    elif use_depth == 'our':
        if truth_relevant:
            depthscale = DepthScale(refiner)
            scale = depthscale.fit_all(mode='all')
            depth = refiner.depth * scale
        else:
            depth = refiner.depth

    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length = 1.0 / 200,
        sdf_trunc = 0.1,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8
    )

    fx, fy, cx, cy = refiner.intrinsics[0,0], refiner.intrinsics[1,1], refiner.intrinsics[0,2], refiner.intrinsics[1,2]
    # fx, fy, cx, cy = 481.2, 480, 319.5, 239.5
    intr = o3d.camera.PinholeCameraIntrinsic(*refiner.size, fx, fy, cx, cy)
    
    # single image visualization:
    print("integrating into single.ply")

    color = refiner.RGB[img1_idx]

    cur_d = o3d.geometry.Image(depth[img1_idx])
    color = o3d.geometry.Image(color)
    # color = o3d.io.read_image(color_dir+fmt.format(img1_idx))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, cur_d, depth_scale=1.0, convert_rgb_to_intensity=False, depth_trunc=50.0)

    ext = np.vstack((extrinsics[img1_idx], np.array([0,0,0,1])))
    ext = np.linalg.inv(ext)
    volume.integrate(rgbd, intr, ext, )

    single = volume.extract_triangle_mesh()
    single.compute_vertex_normals()
    
    o3d.io.write_triangle_mesh("single.ply", single)
    #ptc = volume.extract_voxel_point_cloud()
    #o3d.io.write_point_cloud("single_cld.pcd", ptc)
    volume.reset()
    print("single.ply done.")
    # exit()

    # full reconstruction
    print("integrating into mesh.ply")
    for i, ext in enumerate(tqdm(extrinsics)):
        ext = np.vstack((ext, np.array([0,0,0,1])))
        ext = np.linalg.inv(ext)
        cur_d = depth[i]
        cur_c = refiner.RGB[i]
        if CUT:
            cut = CUT_N
            cur_d[:cut,:] = 0
            cur_d[:,:cut] = 0
            cur_c[:cut,:] = 0
            cur_c[:,:cut] = 0
            cur_d[-cut:,:] = 0
            cur_d[:,-cut:] = 0
            cur_c[-cut:,:] = 0
            cur_c[:,-cut:] = 0
        cur_d = o3d.geometry.Image(cur_d)
        color = o3d.geometry.Image(cur_c)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, cur_d, depth_scale=1.0, convert_rgb_to_intensity=False, depth_trunc=50.0)
        volume.integrate(rgbd, intr, ext)

        # if i >= 10:
        #     break

    print("Extracting triangle mesh from volume..")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("mesh.ply", mesh)
    # o3d.visualization.draw_geometries([mesh])
    #ptc = volume.extract_voxel_point_cloud()
    #o3d.io.write_point_cloud("cld.pcd", ptc)
    #o3d.visualization.draw_geometries([mesh], front=[0.5297, -0.1873, -0.8272], lookat=[2.0712, 2.0312, 1.7251], up=[-0.0558, -0.9809, 0.1864], zoom=0.47)
    print("mesh.ply done.")