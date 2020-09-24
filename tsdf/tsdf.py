import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import struct
#from utils import load_colmap

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
            raise Exception("Incompatible pixel_size(%d) and cv_type(%d)" % (pixel_size, cv_type))
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
    image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return image

volume = o3d.integration.ScalableTSDFVolume(
    voxel_length = 2.0 / 1024,
    sdf_trunc = 0.04,
    color_type=o3d.integration.TSDFVolumeColorType.RGB8
)
data_dir = "./colmap/"

color_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/color_down_png/"
depth_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/depth/"
metadata = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/metadata_scaled.npz"
metad = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/colmap_dense/metadata.npz"

# color_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/color_down_png/"
# depth_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/depth/"
# metadata = "/home/noxx/Documents/projects/consistent_depth/results/debug03/R_hierarchical2_mc/metadata_scaled.npz"
# metad = "/home/noxx/Documents/projects/consistent_depth/results/debug03/colmap_dense/metadata.npz"

fmt = "frame_{:06d}.png"

img = open(data_dir+"images.txt", "r").readlines()
n_imgs = [int(s) for s in img[3].replace(",", "").split() if s.isdigit()][0]
print("number of images: {}".format(n_imgs))

    
with np.load(metadata) as meta_colmap:
    intrinsics = meta_colmap["intrinsics"]
    extrinsics = meta_colmap["extrinsics"]
    scales = meta_colmap["scales"]

scale = scales[:,1].mean()
print(scale)

tmp = intrinsics[0]
print("-----------")
print(extrinsics[0])
e = np.array([0,0,1])
t = extrinsics[:,0:3,-1]
R = -extrinsics[:,0:3,:-1]
mul = np.array([r.dot(e) for r in R])
ROT = np.diag([1, -1, -1])

""" with np.load(metad) as meta_colmap:
    intrinsics = meta_colmap["intrinsics"]
    extrinsics = meta_colmap["extrinsics"]

e = np.array([0,0,1])
t2 = extrinsics[:,0:3,-1]
R = extrinsics[:,0:3,:-1]
mul2 = np.array([-r.dot(e)*1 for r in R])

images = []
for i in range(4,len(img), 2):
    items = img[i].split(" ")
    name = items[-1]
    
    quat = np.array([float(s) for s in items[1:5]])
    tmp = np.array([float(s) for s in items[5:8]]).reshape(1,-1)
    Rtmp, tmp = load_colmap.quat_to_rot(quat, tmp)
    ori = -Rtmp.T.dot(e).reshape(1,-1)*-1
    print(tmp.shape, ori.shape)
    images.append(np.hstack((tmp.T, ori)))
images = np.vstack(images)
print(images.shape) """
#x, y, z = zip(*t)
#u, v, w = zip(*R)
#print(len(x))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(t[:,0], t[:,1],t[:,2])
ax.quiver(t[:,0],t[:,1],t[:,2],mul[:,0],mul[:,1],mul[:,2], length=0.5)
#ax.plot(t2[:,0], t2[:,1],t2[:,2])
#ax.quiver(t2[:,0],t2[:,1],t2[:,2],mul2[:,0],mul2[:,1],mul2[:,2], length=2.0)
#ax.scatter(images[:,0], images[:,1],images[:,2])
#ax.quiver(images[:,0],images[:,1],images[:,2],images[:,3],images[:,4],images[:,5], length=0.5)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
# plt.show()
#exit()
#print(tmp)
intr = o3d.camera.PinholeCameraIntrinsic(384, 224, tmp[0], tmp[1], tmp[2], tmp[3])
extraRow = [0,0,0,1]
extrinsics[:,0:3,-1:] *= -1
depth = o3d.io.read_image(depth_dir+fmt.format(0))
color = o3d.io.read_image(color_dir+fmt.format(0))
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, convert_rgb_to_intensity=False)

volume.integrate(rgbd, intr, np.vstack((extrinsics[0], extraRow)))
single = volume.extract_triangle_mesh()
single.compute_vertex_normals()
o3d.io.write_triangle_mesh("single.ply", single)
#ptc = volume.extract_voxel_point_cloud()
#o3d.io.write_point_cloud("single_cld.pcd", ptc)
volume.reset()

for i, ext in enumerate(extrinsics):
    ext = np.vstack((ext, extraRow))
    
    #color = o3d.geometry.Image(load_raw_float32_image(color_dir+fmt.format(i)))
    #depth = o3d.geometry.Image(load_raw_float32_image(depth_dir+fmt.format(i)))
    #print(color)
    depth = o3d.io.read_image(depth_dir+fmt.format(i))
    color = o3d.io.read_image(color_dir+fmt.format(i))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, convert_rgb_to_intensity=False)
    volume.integrate(rgbd, intr, ext)

    # if i >= 43:
    #     break

print("Extract a triangle mesh from the volume and visualize it.")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("mesh.ply", mesh)
#ptc = volume.extract_voxel_point_cloud()
#o3d.io.write_point_cloud("cld.pcd", ptc)
#o3d.visualization.draw_geometries([mesh], front=[0.5297, -0.1873, -0.8272], lookat=[2.0712, 2.0312, 1.7251], up=[-0.0558, -0.9809, 0.1864], zoom=0.47)