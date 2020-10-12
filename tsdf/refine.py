from scipy.optimize import minimize
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
from multiprocessing import Pool
import open3d as o3d

def resize_intrinsics(intrinsics, size_old, size_new):
    fx, fy, cx, cy = intrinsics[0]
    ratio = np.array(size_new) / np.array(size_old)
    fx *= ratio[0]
    fy *= ratio[1]
    cx *= ratio[0]
    cy *= ratio[1]
    return fx, fy, cx, cy

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
    def __init__(self, color_dir, depth_dir, metadata, size=(80,60)):
        self.ang_thresh = 50/180 * np.pi

        self.color_dir = color_dir
        self.depth_dir = depth_dir
        self.metadata = metadata
        self.size = size

        with np.load(self.metadata) as meta_colmap:
            intrinsics = meta_colmap["intrinsics"]
            self.extrinsics = meta_colmap["extrinsics"]
            scales = meta_colmap["scales"]
        intr = resize_intrinsics(intrinsics, (intrinsics[0,2]*2, intrinsics[0,3]*2), self.size)
        self.intrinsics = np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1]])
        # self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # self.intrinsics = np.linalg.inv(self.intrinsics)
        self.scale = scales[:,1].mean()
        print("mean scale: {}".format(self.scale))

        self.N = self.extrinsics.shape[0]
        COL = np.diag([1, -1, -1])
        for i in range(self.N):
            self.extrinsics[i,:3,:3] = COL.dot(self.extrinsics[i,:,:3]).dot(COL.T)
            self.extrinsics[i,:,3] = COL.dot(self.extrinsics[i,:,3])

            self.extrinsics[i,:,3] = self.extrinsics[i,:,3]/self.scale

            # self.extrinsics[i] = np.linalg.inv(self.extrinsics[i]) # DONT DO THIS PART! not needed here!
        self.extra_row = np.array([0,0,0,1])
        

        self.pair_mat = None
        self.depth = None
        self.RGB = None
        self.luminance = None
        self.normals = None
        self.fresh = True
        self.extrinsics_opt = None

        self.iter = 0

    def filter_framepairs(self):
        if not self.fresh and os.path.isfile("framepairs.npz"):
            framepairs = np.load('framepairs.npz')
            self.pair_mat = framepairs['pair_mat']
            return
        self.pair_mat = np.zeros((self.N, self.N))
        # iterate over all possible pairs
        print("finding valid frame pairs..")
        for i in tqdm(range(self.N)):
            Ti = np.vstack((self.extrinsics[i], self.extra_row))
            for j in range(self.N):
                if i == j:
                    continue
                # check this pair for angle
                Tj = np.vstack((self.extrinsics[j], self.extra_row))
                ez = np.array([0,0,1])
                viewi = Ti[:3,:3].dot(ez)
                viewj = Tj[:3,:3].dot(ez)
                dotprod = viewi.T.dot(viewj)
                angle = np.arccos(dotprod)
                if angle > self.ang_thresh:
                    continue
                # check this pair for overlap
                diks = self.diks * self.depth[i, :, :, np.newaxis]
                overlap = False
                for px in range(self.size[0]):
                    for py in range(self.size[1]):
                        dik = np.append(diks[py, px], 1)

                        dim3_2 = np.linalg.inv(Tj).dot(Ti.dot(dik))
                        dim3_2 = dim3_2[:-1]
                        dim2 = self.intrinsics.dot(dim3_2)
                        dim2 = dim2 / dim2[-1]
                        dim2 = dim2[:-1]

                        if (dim2[0] > 0 and dim2[0] < self.size[0]) and (dim2[1] > 0 and dim2[1] < self.size[1]):
                            overlap = True
                            break
                if not overlap:
                    continue
                
                self.pair_mat[i,j] = 1
        
        np.savez('framepairs', pair_mat=self.pair_mat)

    def load_data(self):
        rgbs = []
        depths = []
        fmt = "frame_{:06d}.png"
        fmt_raw = "frame_{:06d}.raw"
        print("loading data..")
        for i in range(self.N):
            rgbs.append(np.array(Image.open(self.color_dir + fmt.format(i)).resize(self.size)))
            dpt = load_raw_float32_image(self.depth_dir + fmt_raw.format(i))
            dpt = abs(np.array(Image.fromarray(dpt).resize(self.size))-1)
            depths.append(dpt)

        self.RGB = np.array(rgbs)
        self.depth = np.array(depths)

        px = np.repeat(np.arange(self.size[0])[np.newaxis, :], self.size[1], axis=0)
        py = np.repeat(np.arange(self.size[1])[:, np.newaxis], self.size[0], axis=1)
        pz = np.ones_like(px)
        pxyz = np.stack([px, py, pz], axis=2)
        self.diks = np.zeros_like(pxyz).astype(float)
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self.diks[y, x] = np.linalg.inv(self.intrinsics).dot(pxyz[y,x])

    def preprocess_data(self):
        if not self.fresh and os.path.isfile("preprocessed_data.npz"):
            preprocessed_data = np.load('preprocessed_data.npz')
            self.luminance = preprocessed_data["luminance"]
            self.normals = preprocessed_data["normals"]
            return
        lumConst = np.array([0.2126,0.7152,0.0722])
        lumi = []
        nrmls = []
        print("preprocessing data..")
        for i in tqdm(range(self.N)):
            lum = np.matmul(self.RGB[i], lumConst)
            lum = np.stack(np.gradient(lum)[::-1], axis=2) #inverting list order because we want x(width),y(heigth) gradient but images come in y(height),x(width)
            lumi.append(lum)
            # norm = np.linalg.norm(lum, axis=2)
            # plt.imshow(norm)
            # plt.show()
            # exit()

            # dpt = self.depth[i]
            # print(dpt.shape)
            diks = self.diks * self.depth[i,:,:,np.newaxis]

            ptcld = o3d.geometry.PointCloud()
            ptcld.points = o3d.utility.Vector3dVector(diks.reshape(-1,3))

            ptcld.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=25))
            ptcld.orient_normals_towards_camera_location()

            nrml = np.asarray(ptcld.normals).reshape(self.size[1], self.size[0],3)
            nrmls.append(nrml)

            # o3d.visualization.draw_geometries([ptcld], point_show_normal=True)
            # exit()
            
            # dks = diks.reshape(-1,3)
            # nrm = nrml.reshape(-1,3)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.plot(dks[:,0], dks[:,1], dks[:,2], 'r', markersize=1)
            # ax.quiver(dks[:,0], dks[:,1], dks[:,2], nrm[:,0], nrm[:,1], nrm[:,2], length=0.04)
            # plt.show()
            # exit()

        self.luminance = np.array(lumi)
        self.normals = np.array(nrmls)

        np.savez('preprocessed_data', luminance=self.luminance, normals=self.normals)

    def photo_energy(self, i, j, py, px, Ti, Tj, dik):
        left = self.luminance[i, py, px]

        dim3 = np.linalg.inv(Tj).dot(Ti.dot(dik))
        dim3 = dim3[:-1]
        dim2 = self.intrinsics.dot(dim3)
        dim2 = dim2 / dim2[-1]
        dim2 = dim2[:-1]

        dim2_int = np.rint(dim2).astype(int)
        if (dim2_int[0] < 0 or dim2_int[1] < 0) or (dim2_int[0] >= self.size[0] or dim2_int[1] >= self.size[1]):
            return 0

        right = self.luminance[j, dim2_int[1], dim2_int[0]]

        energy = np.sum((left - right)**2)
        return energy

    def geo_energy(self, i, j, py, px, Ti, Tj, dik):
        normal = self.normals[i, py, px]

        dim3_2 = np.linalg.inv(Tj).dot(Ti.dot(dik))
        dim3_2 = dim3_2[:-1]
        dim2 = self.intrinsics.dot(dim3_2)
        dim2 = dim2 / dim2[-1]
        dim2 = dim2[:-1]

        dim2_int = np.rint(dim2).astype(int)
        if (dim2_int[0] < 0 or dim2_int[1] < 0) or (dim2_int[0] >= self.size[0] or dim2_int[1] >= self.size[1]):
            return 0

        depth_probed_j = np.array([dim2[0], dim2[1], self.depth[j, dim2_int[1], dim2_int[0]]])
        dim3_2 = np.linalg.inv(self.intrinsics).dot(depth_probed_j)
        dim3_2 = np.append(dim3_2, 1)

        depth_diff = dik - np.linalg.inv(Ti).dot(Tj.dot(dim3_2))
        energy = ((normal.T).dot(depth_diff[:3]))**2
        return energy

    def total_energy_pair(self, params):
        i, j, Ti, Tj = params
        diks = self.diks * self.depth[i, :, :, np.newaxis]
        egeo = ephoto = 0
        for px in range(self.size[0]):
            for py in range(self.size[1]):
                dik = np.append(diks[py, px], 1)
                egeo += self.geo_energy(i, j, py, px, Ti, Tj, dik)
                ephoto += self.photo_energy(i, j, py, px, Ti, Tj, dik)
        return egeo, ephoto

    def total_energy(self, extr):
        print("function call!")
        extr = extr.reshape(self.extrinsics.shape)
        wgeo = wphoto = 0.5
        egeo = ephoto = 0
        for i in tqdm(range(self.N)):
            Ti = np.vstack((extr[i], self.extra_row))
            # Ti = extr[i]
            for j in range(self.N):
                if self.pair_mat[i,j] != 1:
                    continue
                # Tj = extr[j]
                Tj = np.vstack((extr[j], self.extra_row))
                egeo_n, ephoto_n = self.total_energy_pair([i, j, Ti, Tj])
                egeo += egeo_n
                ephoto += ephoto_n

        total = wphoto * ephoto + wgeo * egeo
        return total
    
    def total_energy_mt(self, extr):
        self.iter += 1
        print("function call #{}".format(self.iter))
        extr = extr.reshape(self.extrinsics.shape)
        wgeo = wphoto = 0.5
        pool = Pool(12)
        params = []
        for i in range(self.N):
            Ti = np.vstack((extr[i], self.extra_row))
            # Ti = extr[i]
            for j in range(self.N):
                if self.pair_mat[i,j] != 1:
                    continue
                # Tj = extr[j]
                Tj = np.vstack((extr[j], self.extra_row))
                params.append([i, j, Ti, Tj])

        energies = pool.map(self.total_energy_pair, params)
        pool.close()
        pool.join()
        energies = np.sum(np.array(energies), axis=1)
        egeo = energies[0]
        ephoto = energies[1]

        total = wphoto * ephoto + wgeo * egeo
        return total

    def resize_stride(self, stride):
        self.extrinsics = self.extrinsics[::stride]
        
        self.RGB = self.RGB[::stride]
        self.depth = self.depth[::stride]

        self.luminance = self.luminance[::stride]
        self.normals = self.normals[::stride]

        self.fresh = True
        self.N = self.extrinsics.shape[0]
        self.filter_framepairs()

    def prepare(self):
        self.load_data()
        self.preprocess_data()
        self.filter_framepairs()

    def optim(self, maxIter=1):
        self.minimizer = minimize(self.total_energy_mt, self.extrinsics, method=None, options={"maxiter":maxIter})
        self.extrinsics_opt = self.minimizer.x

        return self.extrinsics_opt

stride = 10
fresh = False
if __name__ == "__main__":
    peter = False

    color_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/color_down_png/"
    # color_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/color_full/"
    depth_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/depth/"
    metadata = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/R_hierarchical2_mc/metadata_scaled.npz"

    if peter:
        color_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/color_down_png/"
        # color_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/color_full/"
        depth_dir = "/home/noxx/Documents/projects/consistent_depth/results/debug03/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/depth/"
        metadata = "/home/noxx/Documents/projects/consistent_depth/results/debug03/R_hierarchical2_mc/metadata_scaled.npz"

    refiner = pose_refiner(color_dir, depth_dir, metadata)
    refiner.fresh = fresh
    refiner.prepare()
    refiner.resize_stride(5)
    extrinsics_new = refiner.optim()

    COL = np.diag([1, -1, -1])
    for i in range(refiner.N):
        extrinsics_new[i,:3,3] = extrinsics_new[i,:3,3]*refiner.scale

        extrinsics_new[i,:3,:3] = COL.dot(extrinsics_new[i,:3,:3]).dot(COL.T)
        extrinsics_new[i,:3,3] = COL.dot(extrinsics_new[i,:3,3])

    np.savez('extrinsics_new', extrinsics_new)

    import pdb
    pdb.set_trace()

    fmt = "frame_{:06d}.png"
    fmt_raw = "frame_{:06d}.raw"

    img1 = np.array(Image.open(color_dir + fmt.format(0)))
    dpt1 = load_raw_float32_image(depth_dir + fmt_raw.format(0))
    dpt1 = abs(dpt1-1)
    T1 = refiner.extrinsics[0]
    #comparing frame 0 to stride for testing
    img2 = np.array(Image.open(color_dir + fmt.format(stride)))
    dpt2 = load_raw_float32_image(depth_dir+fmt_raw.format(stride))
    T2 = refiner.extrinsics[stride]
    transformed = np.zeros_like(img1)

    # size_old = (6,4)
    size_old = (384, 224)
    px = np.repeat(np.arange(size_old[0])[np.newaxis,:], size_old[1], axis=0)
    py = np.repeat(np.arange(size_old[1])[:, np.newaxis], size_old[0], axis=1)
    pz = np.ones_like(px)
    pxyz = np.stack([px, py, pz], axis=2)
    print(px.shape, py.shape, pxyz.shape)
    diks = np.zeros_like(pxyz).astype(float)
    for x in range(size_old[0]):
        for y in range(size_old[1]):
            # print(pxyz[y,x])
            tmp = np.linalg.inv(refiner.intrinsics).dot(pxyz[y,x])
            # print(tmp)
            diks[y, x,:] = tmp
            # print(diks[y,x])
            # print(np.linalg.inv(refiner.intrinsics).dot(pxyz[y,x]))
    # for i, xyz in enumerate(pxyz.reshape(-1,3)):
    #     print(xyz)
    #     diks[i] = np.linalg.inv(refiner.intrinsics).dot(xyz)
    # diks = diks.reshape(pxyz.shape)
    # diks = np.tensordot(np.linalg.inv(refiner.intrinsics), pxyz, axes=([0,1],[2]))
    print("transform")
    for x in range(size_old[0]):
        for y in range(size_old[1]):
            curDepth = dpt1[y, x]
            curRGB = img1[y, x]
            # print(curRGB)
            pos = np.array([x, y, 1])
            # print(pos)
            # print(pxyz[y,x])
            # dik = np.linalg.inv(refiner.intrinsics).dot(pos)
            # print(dik)
            # dik *= curDepth
            # print(dik)
            dik = diks[y, x] * curDepth
            # print(dik)
            # dik *= curDepth
            # print(dik)
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