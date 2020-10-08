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

class pose_refiner:
    def __init__(self, color_dir, depth_dir, metadata):
        self.color_dir = color_dir
        self.depth_dir = depth_dir
        self.metadata = metadata

        with np.load(self.metadata) as meta_colmap:
            intrinsics = meta_colmap["intrinsics"]
            self.extrinsics = meta_colmap["extrinsics"]
            scales = meta_colmap["scales"]
        intr = intrinsics[0]
        self.intrinsics = np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1]])
        # self.intrinsics = np.linalg.inv(self.intrinsics)
        self.scale = scales[:,1].mean()
        print("mean scale: {}".format(self.scale))

        COL = np.diag([1, -1, -1])
        extra_row = np.zeros((self.extrinsics.shape[0],1,4))
        extra_row[:,0,3] = 1
        self.extrinsics = np.concatenate((self.extrinsics, extra_row), axis=1)
        for i in range(self.extrinsics.shape[0]):
            self.extrinsics[i,:3,:3] = COL.dot(self.extrinsics[i,:3,:3]).dot(COL.T)
            self.extrinsics[i,:3,3] = COL.dot(self.extrinsics[i,:3,3])

            self.extrinsics[i,:3,3] = self.extrinsics[i,:3,3]/self.scale

            # self.extrinsics[i] = np.linalg.inv(self.extrinsics[i]) # DONT DO THIS PART! not needed here!

        self.N = self.extrinsics.shape[0]

        self.pair_mat = None
        self.depth = None
        self.RGB = None
        self.luminance = None
        self.normals = None

    def sizes(self, size_old, size_new):
        self.size_old = size_old
        self.size_new = size_new

    def filter_framepairs(self):
        self.pair_mat = np.zeros((self.N, self.N))
        for y in range(self.N):
            for x in range(self.N):
                if y == x+stride:
                    self.pair_mat[y,x] = 1

    def photo_energy(self, y, x, py, px, Ti, Tj, dik):
        left = self.luminance[y, py, px]

        dim3 = np.linalg.inv(Tj).dot(Ti.dot(dik))
        print(dim3[3])
        dim3 = dim3 / dim3[-1]
        dim3 = dim3[:-1]
        dim2 = self.intrinsics.dot(dim3)
        dim2 = dim2 / dim2[-1]
        dim2 = dim2[:-1]

        right = self.luminance(x, dim2[1], dim2[0])

        energy = left - right
        energy = np.linalg.norm(energy)
        return energy

    def geo_energy(self, y, x, py, px, Ti, Tj, dik):
        return 1

    def total_energy(self, extr):
        wgeo = wphoto = 0.5
        egeo = ephoto = 0
        for y in range(self.N):
            for x in range(self.N):
                if self.pair_mat[y,x] != 1:
                    continue
                Ti = extr[y]
                Tj = extr[x]
                for px in range(self.size_new.shape[0]):
                    for py in range(self.size_new.shape[1]):
                        dik = np.linalg.inv(self.intrinsics).dot(np.array([px, py, 1])) * self.depth[y, py, px]
                        dik = np.append(dik, 1)
                        egeo += self.geo_energy(y, x, py, px, Ti, Tj, dik)
                        ephoto += self.photo_energy(y, x, py, px, Ti, Tj, dik)

        total = wphoto * ephoto + wgeo * egeo
        return total

    def load_data(self):
        self.depth = 0
        self.RGB = 0

    def preprocess_data(self):
        self.luminance = np.zeros((self.N, self.size_new[1], self.size_new[0], 2))

    def prepare(self):
        self.load_data()
        self.preprocess_data()
        self.filter_framepairs()

    def optim(self):
        minimize(self.total_energy, self.extrinsics, method='Newton-CG')

stride = 10
if __name__ == "__main__":
    peter = True

    color_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/color_down_png/"
    # color_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/color_full/"
    depth_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/depth/"
    metadata = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/metadata_scaled.npz"
    size_new = (384, 224)

    if peter:
        color_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/color_down_png/"
        # color_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/color_full/"
        depth_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/depth/"
        metadata = "/home/noxx/Documents/projects/consistent_depth/results/debug03/R_hierarchical2_mc/metadata_scaled.npz"
        size_new = (384, 224)
        # size_new = (1920, 1080)

    size_old = (384, 224)

    refiner = pose_refiner(color_dir, depth_dir, metadata)
    refiner.sizes(size_old, size_new)
    
    # refiner.prepare()

    result = refiner.extrinsics

    fmt = "frame_{:06d}.png"
    fmt_raw = "frame_{:06d}.raw"

    img1 = np.array(Image.open(color_dir + fmt.format(0)))
    dpt1 = load_raw_float32_image(depth_dir + fmt_raw.format(0))
    dpt1 = abs(dpt1-1)
    T1 = refiner.extrinsics[0]
    print(T1.shape)
    #comparing frame 0 to stride for testing
    img2 = np.array(Image.open(color_dir + fmt.format(stride)))
    dpt2 = load_raw_float32_image(depth_dir+fmt_raw.format(stride))
    T2 = refiner.extrinsics[stride]
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
            dik = np.linalg.inv(refiner.intrinsics).dot(pos) * curDepth
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
                    img2[imgY, imgX] = curRGB
            # exit()

    plt.imshow(transformed)
    plt.show()
    plt.imshow(img2)
    plt.show()
    exit()