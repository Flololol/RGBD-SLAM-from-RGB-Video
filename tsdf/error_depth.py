import numpy as np
from pose_refiner import pose_refiner
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

peter = True
size = (640, 480)
use_mean = False

single_img = False
img_id = 0

def depth_error(scale, depth, depth_truth):
    error = depth-(depth_truth/scale)
    error = error.reshape(-1)

    if use_mean:
        error = abs(np.mean(error))
    else:
        error = abs(np.median(error))

    return error

if __name__ == "__main__":
    base_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/debug/"
    depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/depth/"
    depth_truth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS2_Oadam/depth_truth/"
    if peter:
        base_dir = "/home/noxx/Documents/projects/consistent_depth/results/lr_kt2_flo/"
        depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/depth/"
        depth_truth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/depth_truth/"

    color_dir = base_dir+"color_full/"
    metadata = base_dir+"R_hierarchical2_mc/metadata_scaled.npz"

    refiner = pose_refiner(color_dir, depth_dir, metadata, size=size)
    refiner.load_data()

    depth_truth = np.empty_like(refiner.depth)
    fmt = "frame_{:06d}.png"
    print('loading ground truth of depth..')
    for i in tqdm(range(refiner.N)):
        depth_truth[i] = np.array(Image.open(depth_truth_dir + fmt.format(i)).resize(refiner.size))

    res = minimize(depth_error, 10, args=(refiner.depth[img_id], depth_truth[img_id]), method = 'Nelder-Mead', options={"disp":True}).x
    best_scale = res[0]
    print('found best scale for img {}: {}'.format(img_id, best_scale))

    # x = np.linspace(23000,23500,1000)
    # y = [depth_error(xs, refiner.depth[0], depth_truth[0]) for xs in x]
    # plt.plot(x, y)
    # plt.show()

    if single_img:
        plt.imshow(refiner.depth[img_id], cmap='plasma')
        plt.show()
        plt.imshow(depth_truth[img_id]/best_scale, cmap='plasma')
        plt.show()
        error_img = abs(refiner.depth[img_id] - depth_truth[img_id]/best_scale)
        plt.imshow(error_img, cmap='plasma')
        plt.show()
    else:
        fmt = "./error_depth/error_{:06d}.png"
        scales = []
        for i in tqdm(range(refiner.N)):
            cur_scale = minimize(depth_error, 10, args=(refiner.depth[i], depth_truth[i]), method = 'Nelder-Mead', options={"disp":False}).x[0]
            scales.append(cur_scale)
            error = refiner.depth[i] - depth_truth[i]/cur_scale
            plt.imsave(fmt.format(i), error, cmap='coolwarm', vmin=-0.25, vmax=0.25)
    