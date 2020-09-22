

import time

import cv2
import numpy as np

import fusion
import os
import transformations
import matplotlib.pyplot as plt
import struct

# testing just for image 1 and image 5 (arbitrary af)
def calibrate_scale_glob(img, points):
    # grab required data for 2 images
    items = img[4].split(" ")
    name1 = items[-1]

    quat = items[1:5]
    w = quat.pop(0)
    quat.append(w)
    R = np.array(transformations.quaternion_matrix(quat))
    
    tran = np.array([float(items[5]),float(items[6]),float(items[7])])
    Translation = np.eye(4)
    Translation[0:3,3] = tran
    Tvar1 = np.linalg.inv(R.dot(Translation))

    items = img[12].split(" ")
    name2 = items[-1]

    quat = items[1:5]
    w = quat.pop(0)
    quat.append(w)
    R = np.array(transformations.quaternion_matrix(quat))
    
    tran = np.array([float(items[5]),float(items[6]),float(items[7])])
    Translation = np.eye(4)
    Translation[0:3,3] = tran
    Tvar2 = np.linalg.inv(R.dot(Translation))

    pts3D = img[5].split(" ")
    x1 = pts3D[::3]
    y1 = pts3D[1::3]
    pID1 = pts3D[2::3]

    pts3D = img[13].split(" ")
    x2 = pts3D[::3]
    y2 = pts3D[1::3]
    pID2 = pts3D[2::3]

    # look for intersections between the two point sets

    matches = set(pID1).intersection(pID2)
    matches_idx = []
    for m in matches:
        id = np.argwhere(points[:,0] == int(m))
        if len(id) == 0:
            continue
        matches_idx.append(id.item())
    
    best = np.argmin(points[matches_idx,4]) # pick BESTEST feature point for pair (lowest error, probably wrong error) TODO:(that error is also not quality of the feature point, it's the error for colmap to place it in the 3d world?)
    best_pid = points[best,0]
    # find best point again in pixel coords for both images

    mask_1 = np.where(pID1 == best_pid)
    mask_2 = np.where(pID2 == best_pid)

    x1_b = x1[mask_1]
    y1_b = y1[mask_1]
    x2_b = x2[mask_2]
    y2_b = y2[mask_2]

    # now camera intrinsics? then we solve for the same depth parameter in 2 equations that place the same point along different axis into the world space -> the intersection (or almost... it's two lines in a 3d space. ez, just find point where they're closest.) gives us the hopefully proper global depth scale
    

    depth1 = load_raw_float32_image(save_dir+name1[:-4]+"raw")
    depth1 = resize_to_target(depth1, res, suppress_messages=True)
    depth2 = load_raw_float32_image(save_dir+name2[:-4]+"raw")
    depth2 = resize_to_target(depth2, res, suppress_messages=True)

    # now we have the rigid transform for both camera positions, both depths, and a feature point that is shared between both
    # now we EASILY compute the perfect global scale, such that both depth images are a perfect fit (maybe top 90% der feature points, sollten alle gleich sein..)
    # this will be the magical scale that makes everything fall into place!

    exit()

    return scale_factor


def calibrate_scale(x, y, pID, t):
    factors = []
    for i, pt in enumerate(pID):
        if int(pt) == -1: continue
        locX = int(float(x[i]))
        locY = int(float(y[i]))
        dep = depth[locY, locX]
        idx = np.argwhere(points[:,0]==int(pt))
        if points[idx.item(), 4] > 3: continue
        globX = points[idx.item(),1:4]
        dist = np.linalg.norm(t-globX)
        fac = dist/dep
        factors.append(fac)
    print(np.average(factors))
    print(np.median(factors))
    # exit()
    return np.average(factors)

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

if __name__ == "__main__":
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    print("Estimating voxel volume bounds...")
    #requires cameras.txt, images.txt and points3D.txt in data_dir folder
    #requires rgb pngs and corresponding raw depth maps in save_dir
    data_dir = "./colmap/"
    save_dir = "./data2/"
    
    cam = open(data_dir+"cameras.txt", "r")
    
    for line in cam:
        if line.startswith("#"): continue
        params = line.split(" ")
        res = (int(params[3]),int(params[2]))
        cam_intr = np.array([[float(params[4]),0,float(params[6])],[0, float(params[4]), float(params[5])],[0,0,1]])

    img = open(data_dir+"images.txt", "r").readlines()
    n_imgs = [int(s) for s in img[3].replace(",", "").split() if s.isdigit()][0]
    imgs = [s for s in os.listdir(save_dir) if s.endswith('npy')]
    print(n_imgs)

    points3D = [s.replace("\n", "") for s in open(data_dir+"points3D.txt", "r").readlines()[3:]]
    points = np.zeros((len(points3D), 5)).astype(float)
    for i, line in enumerate(points3D):
        items = line.split(" ")
        points[i] = [float(items[0]), float(items[1]), float(items[2]), float(items[3]), float(items[7])]
    
    scale = calibrate_scale_glob(img, points)
    
    #if len(imgs) < n_imgs:
    for i in range(4,len(img), 2):
        items = img[i].split(" ")
        name = items[-1]
        
        quat = items[1:5]
        w = quat.pop(0)
        quat.append(w)
        R = np.array(transformations.quaternion_matrix(quat))
        
        tran = np.array([float(items[5]),float(items[6]),float(items[7])])
        Translation = np.eye(4)
        Translation[0:3,3] = tran
        Tvar = np.linalg.inv(R.dot(Translation))
        savename = save_dir + name[:-4] + "txt"
        np.savetxt(savename, Tvar, delimiter= ' ')

        depth = load_raw_float32_image(save_dir+name[:-4]+"raw")
        depth = resize_to_target(depth, res, suppress_messages=True)

        pts3D = img[i+1].split(" ")
        x = pts3D[::3]
        y = pts3D[1::3]
        pID = pts3D[2::3]
        #calibrate scale
        # scale = calibrate_scale(x, y, pID, tran)
        # scale = 22
        depthname = save_dir + name[:-4] + "npy"
        np.save(depthname, depth * scale)
        
        #print(Tvar)
        exit()

    print("done convert")

    #volume = o3d.integration.ScalableTSDFVolume(0.04)
    vol_bnds = np.zeros((3,2))
    for i in range(n_imgs):
        # Read depth image and camera pose
        #depth_im = cv2.imread(save_dir+"depth_frame_%06d.png"%(i),-1).astype(float)
        """ depth_im = cv2.imread(save_dir+"depth_frame_%06d.png"%(i),0)
        depth_im = cv2.resize(depth_im, (1280,720))
        
        depth_im = np.abs(np.array(depth_im).astype(float)/256)*1.5 """

        depth_im = np.load(save_dir+"frame_%06d.npy"%(i))
        
        #cv2.imshow("gray", depth_im.astype(np.uint8))
        #cv2.waitKey(0)
        #exit()
        #depth_im /= 256.    # depth is saved in 16-bit PNG in millimeters
        #depth_im[depth_im == 65.535] = 0    # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = np.loadtxt(save_dir+"frame_%06d.txt"%(i))    # 4x4 rigid transformation matrix

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #
    #vol_bnds[:,1] = 1 
    print(vol_bnds)
    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.1)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d"%(i+1, n_imgs))

        # Read RGB-D image and camera pose
        color_image = cv2.imread(save_dir+"frame_%06d.png"%(i))

        #depth_im = cv2.resize(cv2.imread(save_dir+"depth_frame_%06d.png"%(i),0), (color_image.shape[1], color_image.shape[0])).astype(float)
        """ print(depth_im.dtype)
        print(depth_im.shape)
        print(depth_im)
        exit()
        depth_im = cv2.resize(depth_im, (color_image.shape[1], color_image.shape[0])) """
        
        #depth_im = np.abs(np.array(depth_im).astype(float)/256)*1.5
        depth_im = np.load(save_dir+"frame_%06d.npy"%(i))
        #depth_im = cv2.cvtColor(depth_im, cv2.COLOR_RGB2GRAY)
        #print(depth_im.dtype)
        #depth_im /= 1000.
        #depth_im[depth_im == 65.535] = 0
        #print(depth_im)
        cam_pose = np.loadtxt(save_dir+"frame_%06d.txt"%(i))

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("pc.ply", point_cloud)