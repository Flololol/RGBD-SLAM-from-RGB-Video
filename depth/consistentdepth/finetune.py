import torch
import numpy as np
from models import pix2pix_model
from options.train_options import TrainOptions
from loaders import aligned_data_loader

BATCH_SIZE = 1

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

video_list = 'test_data/test_davis_video_list.txt'

eval_num_threads = 2
video_data_loader = aligned_data_loader.DAVISDataLoader(video_list, BATCH_SIZE)
video_dataset = video_data_loader.load_data()
print('========================= Video dataset #images = %d =========' %
      len(video_data_loader))

model = pix2pix_model.Pix2PixModel(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
global_step = 0

print(
    '=================================  BEGIN VALIDATION ====================================='
)

print('TESTING ON VIDEO')

model.switch_to_eval()
save_path = 'test_data/viz_predictions/'
print('save_path %s' % save_path)
for x in video_dataset:
    print(x[0].shape)
    print(x[1])
    exit()

for i, data in enumerate(video_dataset):
    print(i)
    stacked_img = data[0]
    targets = data[1]
    model.run_and_save_DAVIS(stacked_img, targets, save_path)