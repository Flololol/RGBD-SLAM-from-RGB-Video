import numpy as np
from pose_refiner import pose_refiner
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

peter = False
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

class DepthScale:
    def __init__(self, refiner):
        self.refiner = refiner
    
    def fit(self, img_id):
        res = minimize(depth_error, 10, args=(self.refiner.depth[img_id], self.refiner.depth_truth[img_id]), method = 'Nelder-Mead', options={"disp":True}).x
        best_scale = res[0]
        print('found best scale for img {}: {}'.format(img_id, best_scale))
        return best_scale
    
    def fit_all(self, mode='median', saveImg=False):
        fmt = "./error_depth/error_{:06d}.png"
        if mode == 'all':
            best_scale = minimize(depth_error, 10, args=(self.refiner.depth, self.refiner.depth_truth), method = 'Nelder-Mead', options={"disp":False}).x[0]
            if saveImg:
                for i in tqdm(range(self.refiner.N)):
                    error = self.refiner.depth[i] - self.refiner.depth_truth[i]/best_scale
                    plt.imsave(fmt.format(i), error, cmap='coolwarm', vmin=-0.25, vmax=0.25)
        else:
            scales = []
            for i in tqdm(range(self.refiner.N)):
                cur_scale = minimize(depth_error, 10, args=(self.refiner.depth[i], self.refiner.depth_truth[i]), method = 'Nelder-Mead', options={"disp":False}).x[0]
                scales.append(cur_scale)
                if saveImg:
                    error = refiner.depth[i] - refiner.depth_truth[i]/cur_scale
                    plt.imsave(fmt.format(i), error, cmap='coolwarm', vmin=-0.25, vmax=0.25)
                
            if mode == 'median': best_scale = np.median(scales)
            if mode == 'mean': best_scale = np.mean(scales)
            
        print("Scale over all images: {}".format(best_scale))
        return best_scale



if __name__ == "__main__":
    base_dir = "/home/flo/Documents/3DCVProject/RGBD-SLAM/room/"
    depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/"
    if peter:
        base_dir = "/home/noxx/Documents/projects/consistent_depth/results/lr_kt2_flo/"
        depth_dir = base_dir+"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS3_Oadam/"

    color_dir = base_dir+"color_full/"
    metadata = base_dir+"R_hierarchical2_mc/"

    refiner = pose_refiner(color_dir, depth_dir, metadata, size=size, GRND_TRTH=True)
    refiner.load_data()

    dptScale = DepthScale(refiner)

    # x = np.linspace(23000,23500,1000)
    # y = [depth_error(xs, refiner.depth[0], refiner.depth_truth[0]) for xs in x]
    # plt.plot(x, y)
    # plt.show()

    if single_img:
        best_scale = dptScale.fit(img_id)
        plt.imshow(refiner.depth[img_id], cmap='plasma')
        plt.show()
        plt.imshow(refiner.depth_truth[img_id]/best_scale, cmap='plasma')
        plt.show()
        error_img = abs(refiner.depth[img_id] - refiner.depth_truth[img_id]/best_scale)
        plt.imshow(error_img, cmap='plasma')
        plt.show()
    else:
        best_scale = dptScale.fit_all(saveImg=True)

       
    