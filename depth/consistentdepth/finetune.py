import torch
import numpy as np
from PIL import Image
from models import pix2pix_model
from options.train_options import TrainOptions
from loaders import aligned_data_loader
import matplotlib.pyplot as plt
import torchvision.transforms as tf

BATCH_SIZE = 1

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
print(opt)

video_path = '../../video/snapshot.jpg'
toTens = tf.ToTensor()
res = tf.Resize((272,496))
#res = tf.Resize((144,256))
toPIL = tf.ToPILImage()
img = Image.open(video_path)
#print(np.array(img).shape)
#plt.imshow(img)
#plt.show()
#plt.clf()

img = res(img)
img = toTens(img).unsqueeze(0)
#img = toPIL(img)
#print(np.array(img).shape)
#plt.imshow(img)
#plt.show()
#plt.clf()

vid = []
vid.append((img, [video_path]))

model = pix2pix_model.Pix2PixModel(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

model.switch_to_eval()
save_path = 'test_data/results/'
print('save_path %s' % save_path)

for i, data in enumerate(vid):
    print(i)
    stacked_img = data[0]
    targets = data[1]
    print(stacked_img.shape)
    model.run_and_save_RGBD(stacked_img, targets, save_path)