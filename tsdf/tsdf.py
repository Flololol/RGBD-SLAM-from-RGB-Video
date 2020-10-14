import open3d as o3d
import numpy as np
import struct
from PIL import Image
from pose_refiner import pose_refiner

peter = False
use_opt = False
img1_idx = 0
size = (1920, 1080)

if __name__ == "__main__":
    eps_euler = .2 #x degree step size in terms of rotation
    eps_translation = .0005 #this is a relative value that depends on the depth scale refiner.scale
    extr_opt = "extrinsics_opt_{}_{}".format(eps_euler, eps_translation)
    extr_opt = "./{}.npz".format(extr_opt)

    base_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/"
    depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/depth/"
    if peter:
        base_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/"
        depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/depth/"

    color_dir = base_dir+"color_down_png/"
    color_dir = base_dir+"color_full/"
    metadata = base_dir+"R_hierarchical2_mc/metadata_scaled.npz"

    refiner = pose_refiner(color_dir, depth_dir, metadata, size=size)
    refiner.load_data()
    extrinsics = refiner.extrinsics

    if use_opt:
        with np.load(extr_opt) as extr_opt:
            extrinsics_opt = extr_opt["extrinsics_opt"]
        refiner.fresh = False
        refiner.preprocess_data()
        refiner.resize_stride(int(refiner.extrinsics.shape[0]/extrinsics_opt.shape[0]+1))
        extrinsics = extrinsics_opt
        # extrinsics = refiner.extrinsics

    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length = 1.0 / 512,
        sdf_trunc = 0.1,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8
    )

    fx, fy, cx, cy = refiner.intrinsics[0,0], refiner.intrinsics[1,1], refiner.intrinsics[0,2], refiner.intrinsics[1,2]
    intr = o3d.camera.PinholeCameraIntrinsic(*refiner.size, fx, fy, cx, cy)

    # single image visualization:
    print("starting on single.ply")
    depth = refiner.depth[img1_idx]
    color = refiner.RGB[img1_idx]
    # plt.clf()
    # plt.imshow(depth, cmap='gray')
    # plt.show()
    # exit()

    depth = o3d.geometry.Image(depth)
    color = o3d.geometry.Image(color)
    # color = o3d.io.read_image(color_dir+fmt.format(img1_idx))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, convert_rgb_to_intensity=False, depth_trunc=1.0)

    ext = np.vstack((extrinsics[img1_idx], np.array([0,0,0,1])))
    ext = np.linalg.inv(ext)
    volume.integrate(rgbd, intr, ext)

    single = volume.extract_triangle_mesh()
    single.compute_vertex_normals()
    o3d.io.write_triangle_mesh("single.ply", single)
    #ptc = volume.extract_voxel_point_cloud()
    #o3d.io.write_point_cloud("single_cld.pcd", ptc)
    volume.reset()
    print("single.ply done.")

    # full reconstruction
    print("starting on mesh.ply")
    for i, ext in enumerate(extrinsics):
        ext = np.vstack((ext, np.array([0,0,0,1])))
        ext = np.linalg.inv(ext)
        depth = o3d.geometry.Image(refiner.depth[i])
        color = o3d.geometry.Image(refiner.RGB[i])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, convert_rgb_to_intensity=False, depth_trunc=1.0)
        volume.integrate(rgbd, intr, ext)

        # if i >= 10:
        #     break

    print("Extracting triangle mesh from volume..")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("mesh.ply", mesh)
    #ptc = volume.extract_voxel_point_cloud()
    #o3d.io.write_point_cloud("cld.pcd", ptc)
    #o3d.visualization.draw_geometries([mesh], front=[0.5297, -0.1873, -0.8272], lookat=[2.0712, 2.0312, 1.7251], up=[-0.0558, -0.9809, 0.1864], zoom=0.47)
    print("mesh.ply done.")