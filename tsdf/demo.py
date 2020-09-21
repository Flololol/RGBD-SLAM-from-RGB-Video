"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion
import os
import transformations
import matplotlib.pyplot as plt


if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  data_dir = "./colmap/"
  save_dir = "./data2/"
  
  cam = open(data_dir+"cameras.txt", "r")
  for line in cam:
      if line.startswith("#"): continue
      params = line.split(" ")
      cam_intr = np.array([[float(params[4]),0,float(params[6])],[0, float(params[4]), float(params[5])],[0,0,1]])

  img = open(data_dir+"images.txt", "r").readlines()
  for i in range(4,len(img), 2):
    items = img[i].split(" ")
    quat = items[1:5]
    w = quat.pop(0)
    quat.append(w)
    R = transformations.quaternion_matrix(quat)
    tran = items[5:8]
    name = items[-1]
    Translation = np.eye(4)
    Translation[0][3] = float(tran[0])
    Translation[1][3] = float(tran[1])
    Translation[2][3] = float(tran[2])
    Tvar = np.matmul(R.transpose(),Translation)
    #print(Tvar)
    savename = save_dir + "pose_" + name[:-4] + "txt"
    np.savetxt(savename, Tvar, delimiter= ' ')
  vol_bnds = np.zeros((3,2))
  imgs = [strng for strng in os.listdir(save_dir) if strng.startswith('pose')]
  n_imgs = len(imgs)
  print(n_imgs)
  for i in range(n_imgs):
    # Read depth image and camera pose
    #depth_im = cv2.imread(save_dir+"depth_frame_%06d.png"%(i),-1).astype(float)
    depth_im = cv2.imread(save_dir+"depth_frame_%06d.png"%(i),0)
    depth_im = cv2.resize(depth_im, (1280,720))
    
    depth_im = np.abs(np.array(depth_im).astype(float)/256 -1)
    #print(depth_im)
    
    #cv2.imshow("gray", depth_im.astype(np.uint8))
    #cv2.waitKey(0)
    #exit()
    #depth_im /= 256.  # depth is saved in 16-bit PNG in millimeters
    #depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    cam_pose = np.loadtxt(save_dir+"pose_frame_%06d.txt"%(i))  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  for i in range(n_imgs):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(save_dir+"frame_%06d.png"%(i)), cv2.COLOR_BGR2RGB)
    color_image = cv2.imread(save_dir+"frame_%06d.png"%(i))
    depth_im = cv2.imread(save_dir+"depth_frame_%06d.png"%(i),0)
    depth_im = cv2.resize(depth_im, (color_image.shape[1], color_image.shape[0]))
    
    depth_im = np.abs(np.array(depth_im).astype(float)/256 -1)
    #depth_im = cv2.cvtColor(depth_im, cv2.COLOR_RGB2GRAY)
    #depth_im /= 1000.
    #depth_im[depth_im == 65.535] = 0
    cam_pose = np.loadtxt(save_dir+"pose_frame_%06d.txt"%(i))

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