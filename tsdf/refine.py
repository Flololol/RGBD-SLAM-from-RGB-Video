from scipy.optimize import minimize
import numpy as np
from PIL import Image
import struct
import matplotlib.pyplot as plt

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

color_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/color_down_png/"
depth_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/depth/"
metadata = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/metadata_scaled.npz"
metad = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/colmap_dense/metadata.npz"

fmt = "frame_{:06d}.png"
fmt_raw = "frame_{:06d}.raw"
size_new = (1280, 720)
size_old = (384, 224)

with np.load(metadata) as meta_colmap:
    intrinsics = meta_colmap["intrinsics"]
    extrinsics = meta_colmap["extrinsics"]
    scales = meta_colmap["scales"]
intr = intrinsics[0]
intr = np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1]])

extr_shape = (extrinsics.shape[0], 4, 4)
extra_row = np.zeros((extrinsics.shape[0],1,4))
extra_row[:,0,3] = 1
extrinsics = np.concatenate((extrinsics, extra_row), axis=1)

img1 = np.array(Image.open(color_dir + fmt.format(0)))
dpt1 = load_raw_float32_image(depth_dir+fmt_raw.format(0))
T1 = extrinsics[0]
print(T1.shape)
#comparing frame 0 to 10 for testing
img2 = np.array(Image.open(color_dir + fmt.format(10)))
dpt2 = load_raw_float32_image(depth_dir+fmt_raw.format(10))
T2 = extrinsics[10]
transformed = np.zeros_like(img1)
print(transformed.shape)
print(dpt1.shape)

for x in range(size_old[0]):
    for y in range(size_old[1]):
        curDepth = dpt1[y, x]
        curRGB = img1[y, x]
        # print(curRGB)
        pos = np.array([x, y, 1])
        # print(pos)
        dik = np.linalg.inv(intr).dot(pos) * curDepth
        dik = np.append(dik, 1)
        # print(dik)
        tgt = np.linalg.inv(T2).dot(T1.dot(dik))
        tgt = np.delete(tgt, 3)
        # print(tgt)
        imgPos = intr.dot(tgt)
        # print(imgPos)
        imgX = int(imgPos[0])
        imgY = int(imgPos[1])
        if imgX > 0 and imgX < img2.shape[1]:
            if imgY > 0 and imgY < img2.shape[0]:
                transformed[y, x] = curRGB
                img2[y, x] = curRGB
        # exit()

plt.imshow(transformed)
plt.show()
plt.imshow(img2)
plt.show()
exit()

def filter_framepairs():
    pass

def photo_energy(Ti, Tj, intr, imgI, imgJ):
    photo = 1
    return photo

def geo_energy():
    return 1

def total_energy(extr, intr, pairs):
    wgeo = wphoto = 0.5
    for pair in pairs:
        Ti = extr[pair[0]]
        Tj = extr[pair[1]]
    total = wphoto * photo_energy() + wgeo * geo_energy()
    return total