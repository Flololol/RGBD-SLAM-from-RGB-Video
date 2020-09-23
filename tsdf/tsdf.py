import open3d as o3d
#import matplotlib.pyplot as plt
import numpy as np
import struct

def load_raw_float32_image(file_name):
    with open(file_name, "rb") as f:
        CV_CN_MAX = 512
        CV_CN_SHIFT = 3
        CV_32F = 5
        I_BYTES = 4
        Q_BYTES = 8

        h = struct.unpack("i", f.read(I_BYTES))[0]
        w = struct.unpack("i", f.read(I_BYTES))[0]

        cv_type = struct.unpack("i", f.read(I_BYTES))[0]
        pixel_size = struct.unpack("Q", f.read(Q_BYTES))[0]
        d = ((cv_type - CV_32F) >> CV_CN_SHIFT) + 1
        assert d >= 1
        d_from_pixel_size = pixel_size // 4
        if d != d_from_pixel_size:
            raise Exception(
                "Incompatible pixel_size(%d) and cv_type(%d)" % (pixel_size, cv_type)
            )
        if d > CV_CN_MAX:
            raise Exception("Cannot save image with more than 512 channels")

        data = np.frombuffer(f.read(), dtype=np.float32)
        result = data.reshape(h, w) if d == 1 else data.reshape(h, w, d)
        return result

def resize_to_target(image, target, align=1, suppress_messages=False):
    if not suppress_messages:
        print("Original size: %d x %d" % (image.shape[1], image.shape[0]))

    resized_height = target[0]
    resized_width = target[1]
    if resized_width % align != 0:
        resized_width = align * round(resized_width / align)
        if not suppress_messages:
            print("Rounding width to closest multiple of %d." % align)
    if resized_height % align != 0:
        resized_height = align * round(resized_height / align)
        if not suppress_messages:
            print("Rounding height to closest multiple of %d." % align)

    if not suppress_messages:
        print("Resized: %d x %d" % (resized_width, resized_height))
    image = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    return image

volume = o3d.integration.ScalableTSDFVolume(
    voxel_length = 4.0 / 512,
    sdf_trunc = 0.04,
    color_type=o3d.integration.TSDFVolumeColorType.RGB8
)
color_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/color_down/"
depth_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/tsdf/data2/"
metadata = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/metadata_scaled.npz"
    
with np.load(metadata) as meta_colmap:
    intrinsics = meta_colmap["intrinsics"]
    extrinsics = meta_colmap["extrinsics"]
    scales = meta_colmap["scales"]

tmp = intrinsics[0]
print(tmp)
intr = o3d.camera.PinholeCameraIntrinsic(384, 224, tmp[0], tmp[1], tmp[2], tmp[3])
extraRow = [0,0,0,1]

for i, ext in enumerate(extrinsics):
    ext = np.vstack((ext, extraRow))
    fmt = "frame_{:06d}.raw"
    #color = o3d.io.read_image(color_dir+fmt.format(i)[:-4]+"png")
    color = o3d.geometry.Image(load_raw_float32_image(color_dir+fmt.format(i)))
    depth = o3d.geometry.Image(load_raw_float32_image(depth_dir+fmt.format(i)))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
    depth_scale=1., convert_rgb_to_intensity=False)
    volume.integrate(rgbd, intr, np.linalg.inv(ext))


print("Extract a triangle mesh from the volume and visualize it.")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("mesh_tsdf.ply", mesh)
#o3d.visualization.draw_geometries([mesh], front=[0.5297, -0.1873, -0.8272], lookat=[2.0712, 2.0312, 1.7251], up=[-0.0558, -0.9809, 0.1864], zoom=0.47)