from scipy.optimize import minimize
import numpy as np
from PIL import Image
import struct
import os
from tqdm import tqdm
from multiprocessing import Pool
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class pose_refiner:
    def __init__(self, color_dir, depth_dir, metadata, size=(80,60), GRND_TRTH=False):
        self.ang_thresh = 50/180 * np.pi

        self.color_dir = color_dir
        self.size = size

        self.depth_dir = depth_dir + "depth/"
        self.depth_truth_dir = None
        
        self.metadata = metadata + "metadata_scaled.npz"
        self.metadata_truth = None

        with np.load(self.metadata) as meta_colmap:
            intrinsics = meta_colmap["intrinsics"]
            self.extrinsics = meta_colmap["extrinsics"]
            scales = meta_colmap["scales"]
        intr = resize_intrinsics(intrinsics, (intrinsics[0,2]*2, intrinsics[0,3]*2), self.size)
        self.intrinsics = np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1]])
        self.scale = scales[:,1].mean()

        self.N = self.extrinsics.shape[0]
        self.extrinsics_euler = np.empty((self.N, 6))
        COL = np.diag([1, -1, -1])
        for i in range(self.N):
            rotation = COL.dot(self.extrinsics[i,:,:3]).dot(COL.T)
            euler = R.from_matrix(rotation).as_euler('xyz', degrees=True)
            self.extrinsics[i,:3,:3] = rotation
            translation = COL.dot(self.extrinsics[i,:,3])/self.scale
            self.extrinsics[i,:,3] = translation
            self.extrinsics_euler[i] = np.append(euler, translation)
            # self.extrinsics[i] = np.linalg.inv(self.extrinsics[i]) # DONT DO THIS PART! not needed here!

        self.iter = 0
        self.pair_mat = None
        self.depth = None
        self.depth_truth = None
        self.RGB = None
        self.luminance = None
        self.normals = None
        self.fresh = True
        self.extrinsics_opt = None

        self.GRND_TRTH = GRND_TRTH
        if self.GRND_TRTH:
            self.depth_truth_dir = depth_dir + "depth_truth/"
            self.metadata_truth = metadata + "livingRoom2n.gt.sim"
            extr_truth = []
            f = open(self.metadata_truth, "r")
            for l in f:
                if len(l) != 1:
                    split = l.split(sep=' ')
                    for s in split:
                        extr_truth.append(float(s))
            
            extr_truth = np.array(extr_truth)
            extr_truth = extr_truth.reshape((int(extr_truth.shape[0]/12), 3, 4))

            self.N = np.min([extr_truth.shape[0], self.extrinsics.shape[0]]) #somehow we're missing 1 extrinsics.. we'll just assume it is the last one that is missing

            ROT = np.diag([1, -1, 1])
            for i in range(self.N):
                extr_truth[i,:3,:3] = ROT.dot(extr_truth[i,:3,:3]).dot(ROT.T)
                extr_truth[i,:3,3] = ROT.dot(extr_truth[i,:3,3])
            
            self.extrinsics_truth = extr_truth[:self.N]
            self.extrinsics = self.extrinsics[:self.N]
            self.extrinsics_euler = self.extrinsics_euler[:self.N]

    def filter_framepairs(self, overlap=0.2, save=True):
        if not self.fresh and os.path.isfile("framepairs.npz"):
            framepairs = np.load('framepairs.npz')
            self.pair_mat = framepairs['pair_mat']
            if self.pair_mat.shape[0] == self.N:
                return
        self.pair_mat = np.zeros((self.N, self.N))
        # iterate over all possible pairs
        print("finding valid frame pairs..")
        for i in tqdm(range(self.N)):
            Ti = np.vstack((self.extrinsics[i], np.array([0,0,0,1])))
            for j in range(self.N):
                if i == j:
                    continue
                # check this pair for angle
                Tj = np.vstack((self.extrinsics[j], np.array([0,0,0,1])))
                Tj_inv = np.linalg.inv(Tj)
                Tij = Tj_inv.dot(Ti)
                Tij[:3,:4] = self.intrinsics.dot(Tij[:3,:4])

                ez = np.array([0,0,1])
                viewi = Ti[:3,:3].dot(ez)
                viewj = Tj[:3,:3].dot(ez)
                dotprod = viewi.T.dot(viewj)
                angle = np.arccos(dotprod)
                if angle > self.ang_thresh:
                    continue

                # check this pair for overlap vectorized
                diks = (self.diks * self.depth[i, :, :, np.newaxis]).reshape(-1,3)
                diks = np.concatenate((diks, np.ones((diks.shape[0],1))), axis=1)

                diks = np.tensordot(Tij, diks, axes=([1],[1])).T
                diks = (diks[:,:3] / diks[:,2][:,np.newaxis]).reshape(self.size[1],self.size[0],3)

                xLim = np.logical_and(diks[:,:,0]>0, diks[:,:,0]<self.size[0])
                yLim = np.logical_and(diks[:,:,1]>0, diks[:,:,1]<self.size[1])
                lim = np.logical_and(xLim, yLim)
                frac = np.sum(lim) / np.multiply(*self.size)

                if frac < overlap:
                    continue
                
                self.pair_mat[i,j] = 1
        
        if save:
            np.savez('framepairs', pair_mat=self.pair_mat)
        print('found {} framepairs'.format(int(np.sum(self.pair_mat))))

    def load_data(self):
        rgbs = []
        depths = []
        depth_truth = []
        fmt = "frame_{:06d}.png"
        fmt_raw = "frame_{:06d}.raw"
        fmt_truth = "frame_{:06d}.depth"
        fmt_numpy = "frame_{:06d}.npy"
        print("loading data..")
        fx, fy, cx, cy = self.intrinsics[0,0], self.intrinsics[1,1], self.intrinsics[0,2], self.intrinsics[1,2]
        for i in tqdm(range(self.N)):
            rgbs.append(np.array(Image.open(self.color_dir + fmt.format(i)).resize(self.size)))
            dpt = load_raw_float32_image(self.depth_dir + fmt_raw.format(i))
            dpt = abs(np.array(Image.fromarray(dpt).resize(self.size))-1)
            depths.append(dpt)
            
            if self.GRND_TRTH:
                npy = self.depth_truth_dir + fmt_numpy.format(i)
                if os.path.isfile(npy):
                # if False:
                    dpt_trth = np.load(npy)
                else:
                    dpt_trth = np.loadtxt(self.depth_truth_dir + fmt_truth.format(i)).reshape(self.size[1],self.size[0])
                    for y in range(dpt_trth.shape[0]):
                        for x in range(dpt_trth.shape[1]):
                            dpt_trth[y, x] = dpt_trth[y, x] / np.sqrt(((x-cx)/fx)**2 + ((y-cy)/fy)**2 + 1)
                    np.save(npy, dpt_trth)

                depth_truth.append(dpt_trth)

        self.RGB = np.array(rgbs)
        self.depth = np.array(depths)
        if self.GRND_TRTH:
            self.depth_truth = np.array(depth_truth)

        #prepare array 'diks' that, when multiplied with the depth, yields the 3d point 'dik' in camera coordinates (for image i and pixel k)
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
            if self.luminance.shape[0] == self.N:
                return
        lumConst = np.array([0.2126,0.7152,0.0722])
        lumi = []
        nrmls = []
        print("preprocessing data..")
        for i in tqdm(range(self.N)):
            lum = np.matmul(self.RGB[i], lumConst)
            lum = np.stack(np.gradient(lum)[::-1], axis=2) #inverting list order because we want x(width),y(heigth) gradient but images come in y(height),x(width)
            lumi.append(lum)


            diks = self.diks * self.depth[i,:,:,np.newaxis]
            ptcld = o3d.geometry.PointCloud()
            ptcld.points = o3d.utility.Vector3dVector(diks.reshape(-1,3))
            ptcld.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=25))
            ptcld.orient_normals_towards_camera_location()

            nrml = np.asarray(ptcld.normals).reshape(self.size[1], self.size[0],3)
            nrmls.append(nrml)

        self.luminance = np.array(lumi)
        self.normals = np.array(nrmls)

        np.savez('preprocessed_data', luminance=self.luminance, normals=self.normals)

    def photo_energy(self, i, j, py, px, dim2, dim2_int):
        left = self.luminance[i, py, px]

        # dim3 = np.linalg.inv(Tj).dot(Ti.dot(dik))
        # dim3 = dim3[:-1]
        # dim2 = self.intrinsics.dot(dim3)
        # dim2 = dim2 / dim2[-1]
        # dim2 = dim2[:-1]

        right = self.luminance[j, dim2_int[1], dim2_int[0]]

        energy = np.sum((left - right)**2)
        return energy

    def geo_energy(self, i, j, py, px, Ti, Tj, dik, dim2, dim2_int):
        normal = self.normals[i, py, px]

        # dim3_2 = np.linalg.inv(Tj).dot(Ti.dot(dik))
        # dim3_2 = dim3_2[:-1]
        # dim2 = self.intrinsics.dot(dim3_2)
        # dim2 = dim2 / dim2[-1]
        # dim2 = dim2[:-1]

        depth_probed_j = np.array([dim2_int[0], dim2_int[1], self.depth[j, dim2_int[1], dim2_int[0]]])
        dim3_2 = np.linalg.inv(self.intrinsics).dot(depth_probed_j)
        dim3_2 = np.append(dim3_2, 1)
        depth_diff = dik - np.linalg.inv(Ti).dot(Tj.dot(dim3_2))
        energy = ((normal.T).dot(depth_diff[:3]))**2
        return energy

    def total_energy_pair(self, params):
        i, j, Ti, Tj, diks, djks = params
        # diks = self.diks * self.depth[i, :, :, np.newaxis]
        egeo = np.zeros((self.size[0]*self.size[1]))
        ephoto = np.zeros((self.size[0]*self.size[1]))
        valid = np.zeros((self.size[0]*self.size[1]))
        # valid = self.size[0]*self.size[1]
        for px in range(self.size[0]):
            for py in range(self.size[1]):
                djk = djks[py, px]
                dim2 = djk[:-1]
                dim2_int = np.rint(dim2).astype(int)
                if (dim2_int[0] < 0 or dim2_int[1] < 0) or (dim2_int[0] >= self.size[0] or dim2_int[1] >= self.size[1]):
                    continue
                cut = 5
                if (px < cut or py < cut) or (px >= self.size[0]-cut or py >= self.size[1]-cut):
                    continue
                valid[self.size[0]*py+px] = 1

                dik = np.append(diks[py, px],1)
                ephoto[self.size[0]*py+px] = self.photo_energy(i, j, py, px, dim2, dim2_int)
                egeo[self.size[0]*py+px] = self.geo_energy(i, j, py, px, Ti, Tj, dik, dim2, dim2_int)

        return (egeo, ephoto, valid)
    
    # def total_energy(self, extr):
    #     print("function call!")
    #     extr = extr.reshape(self.extrinsics_euler.shape)
    #     wgeo = wphoto = 0.5
    #     egeo = ephoto = valid = 0
    #     for i in tqdm(range(self.N)):
    #         Ti = np.vstack((extr[i], np.array([0,0,0,1])))

    #         diks = (self.diks * self.depth[i, :, :, np.newaxis])
    #         Tiks = diks.reshape(-1,3)
    #         tmp = np.ones((Tiks.shape[0]))
    #         Tiks = np.concatenate((Tiks, tmp[:,np.newaxis]), axis=1)
    #         Tiks = np.tensordot(Ti,Tiks,axes=([1],[1])).T

    #         for j in range(self.N):
    #             if self.pair_mat[i,j] != 1:
    #                 continue
    #             Tj = np.vstack((extr[j], np.array([0,0,0,1])))
    #             Tj_inv = np.linalg.inv(Tj)
                
    #             djks = np.tensordot(Tj_inv, Tiks, axes=([1],[1])).T
    #             djks = np.tensordot(self.intrinsics, djks[:,:3], axes=([1],[1])).T
    #             djks = (djks / djks[:,2][:,np.newaxis]).reshape(self.size[1],self.size[0],3) #pixel coordinates
    #             egeo_n, ephoto_n, valid_n = self.total_energy_pair([i, j, Ti, Tj, diks, djks])
    #             egeo += egeo_n
    #             ephoto += ephoto_n
    #             valid += valid_n

    #     egeo /= valid
    #     ephoto /= valid
    #     total = wphoto * ephoto + wgeo * egeo
    #     return total
    
    def transformation_mt(self, inp):
        extr, i = inp

        roti = R.from_euler('xyz', extr[i,:3], degrees=True).as_matrix()
        Ti = np.identity(4)
        Ti[:3,:] = np.hstack((roti, extr[i,3:,np.newaxis]))

        diks = self.diks * self.depth[i, :, :, np.newaxis]
        diks1 = diks.reshape(-1,3)
        diks1 = np.concatenate((diks1, np.ones((diks1.shape[0],1))), axis=1)
    
        params = []
        for j in range(self.N):
            if self.pair_mat[i,j] != 1:
                continue
            rotj = R.from_euler('xyz', extr[j,:3], degrees=True).as_matrix()
            Tj = np.identity(4)
            Tj[:3,:] = np.hstack((rotj, extr[j,3:,np.newaxis]))

            Tij = np.linalg.inv(Tj).dot(Ti)
            Tij[:3,:] = self.intrinsics.dot(Tij[:3,:])

            djks = np.tensordot(Tij, diks1, axes=([1],[1])).T
            djks = (djks[:,:3] / djks[:,2][:,np.newaxis]).reshape(self.size[1],self.size[0],3) #pixel coordinates

            params.append([i, j, Ti, Tj, diks, djks])
        return params

    def total_energy_mt(self, extr):
        wgeo = 1
        wphoto = 0.01

        self.iter += 1
        print("call #{}".format(self.iter), end='\r')
        extr = extr.reshape(self.extrinsics_euler.shape)

        rng = list(np.arange(self.N))
        rng = [(extr, i) for i in rng]

        pool = Pool(12)
        params = pool.map(self.transformation_mt, rng)
        params = [itm for lst in params for itm in lst]

        out = pool.map(self.total_energy_pair, params)
        pool.close()
        pool.join()
        valid = np.array(out)[:,2,:]
        energies = np.array(out)[:,:2,:]

        egeo = 0
        ephoto = 0
        use_median = False
        if use_median:
            for i in range(self.pair_mat.shape[0]):
                if np.sum(valid[i]) > 0:
                    # print(np.median(energies[i,0,np.where(valid[i]==1)]))
                    # print(np.median(energies[i,1,np.where(valid[i]==1)]))
                    egeo += np.median(energies[i,0,np.where(valid[i]==1)])
                    ephoto += np.median(energies[i,1,np.where(valid[i]==1)])
            egeo /= self.pair_mat.shape[0]
            ephoto /= self.pair_mat.shape[0]
        else:
            energies = np.sum(energies, axis=2) #sum over all pixels for each pair
            energies = np.sum(energies, axis=0) #sum over all pairs
            total_valid = np.sum(valid)
            egeo = energies[0] / total_valid
            ephoto = energies[1] / total_valid
        
        total = wphoto * ephoto + wgeo * egeo
        # print("E: {}, call #: {}".format(total, self.iter))
        return total

    def resize_stride(self, stride):
        self.extrinsics = self.extrinsics[::stride]
        self.N = self.extrinsics.shape[0]
        self.extrinsics_euler = self.extrinsics_euler[::stride]
        
        self.RGB = self.RGB[::stride]
        self.depth = self.depth[::stride]

        if self.luminance is not None:
            self.luminance = self.luminance[::stride]
            self.normals = self.normals[::stride]
            self.filter_framepairs(save=False)

        if self.GRND_TRTH:
            self.extrinsics_truth = self.extrinsics_truth[::stride]
            self.depth_truth = self.depth_truth[::stride]

    def cut_length(self, length):
        self.extrinsics = self.extrinsics[:length]
        self.N = self.extrinsics.shape[0]
        self.extrinsics_euler = self.extrinsics_euler[:length]
        
        self.RGB = self.RGB[:length]
        self.depth = self.depth[:length]

        if self.luminance is not None:
            self.luminance = self.luminance[:length]
            self.normals = self.normals[:length]
            self.filter_framepairs(save=False)

        if self.GRND_TRTH:
            self.extrinsics_truth = self.extrinsics_truth[:length]
            self.depth_truth = self.depth_truth[:length]

    def prepare(self):
        self.load_data()
        self.preprocess_data()
        self.filter_framepairs()

    def optim(self, eps_euler=.2, eps_translation=.005, maxIter=1):
        print('starting optim() with ee: {:.2f} & et: {:.4f}'.format(eps_euler, eps_translation))

        eps = np.empty_like(self.extrinsics_euler)
        eps[:,:3] = eps_euler
        eps[:,3:] = eps_translation
        eps = eps.reshape(-1)

        self.minimizer = minimize(self.total_energy_mt, self.extrinsics_euler.reshape(-1), method = 'Nelder-Mead', options={"disp":True,"maxiter":maxIter})
        # self.minimizer = minimize(self.total_energy_mt, self.extrinsics_euler.reshape(-1), options={"disp":True,"maxiter":maxIter, "eps":eps})

        self.extrinsics_euler_opt = self.minimizer.x.reshape(self.extrinsics_euler.shape)
        self.extrinsics_opt = np.empty_like(self.extrinsics)
        for i in range(self.extrinsics_euler_opt.shape[0]):
            rotation = R.from_euler('xyz', self.extrinsics_euler_opt[i,:3], degrees=True).as_matrix()
            self.extrinsics_opt[i] = np.hstack((rotation, self.extrinsics_euler_opt[i,3:,np.newaxis]))

        return self.extrinsics_opt


def resize_intrinsics(intrinsics, size_old, size_new):
    fx, fy, cx, cy = intrinsics[0]
    ratio = np.array(size_new) / np.array(size_old)
    fx *= ratio[0]
    fy *= ratio[1]
    cx *= ratio[0]
    cy *= ratio[1]
    print('resizing intrinsics from {} to {}'.format(size_old, size_new))
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