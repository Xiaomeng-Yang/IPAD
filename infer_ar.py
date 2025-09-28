import torch
import argparse
from PIL import Image
from strhub.models.utils import load_from_checkpoint, parse_model_args
from strhub.data.module import SceneTextDataModule

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Load model and image transforms
# parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
parser.add_argument('image_path', help="Path of the Image")
parser.add_argument('save_path', help="Path to save the images")
args, unknown = parser.parse_known_args()
kwargs = parse_model_args(unknown)

parseq = load_from_checkpoint(args.checkpoint, **kwargs).eval()

img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

ori_img = Image.open(args.image_path).convert('RGB')
# Preprocess. Model expects a batch of images with shape: (B, C, H, W)
img = img_transform(ori_img).unsqueeze(0)

length_pred, logits, sa_weights_list = parseq(img)
T = len(sa_weights_list) - 1

attention_mask = np.zeros((T+1, 2*(T+2)))
for i in range(T+1):
    attention_mask[i][:i+1] = sa_weights_list[i][:i+1]
    attention_mask[i][(T+2):] = sa_weights_list[i][(i+1):(i+T+3)]
        
        
print(T)
print(sa_weights_list)
for i in range(len(sa_weights_list)):
    print(sa_weights_list[i].shape)
print(attention_mask)


# logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol
# Greedy decoding
pred = logits.softmax(-1)
label, confidence = parseq.tokenizer.decode(pred)
print('Decoded label = {}'.format(label[0]))

# print(attention_mask)

fig, ax = plt.subplots(figsize=(20, 9)) # set figure size
heatmap = ax.pcolor(attention_mask, cmap=plt.cm.Blues, alpha=0.9)
# print(list(label[0]))

Y_label = list(label[0]) + ['[E]']
X_label = ['[B]'] + list(label[0]) + ['[E]', '[B]'] + ['[M]']*T + ['[E]']

xticks = range(0,len(X_label))
ax.set_xticks(xticks, minor=False) # major ticks

ax.set_xticklabels(X_label, minor = False, fontsize=30)   # labels should be 'unicode'

# print(Y_label)
yticks = range(0, len(Y_label))
ax.set_yticks(yticks, minor=False)
ax.set_yticklabels(Y_label, minor = False, fontsize=30)   # labels should be 'unicode'

ax.grid(True)
plt.savefig(args.save_path + 'new' + '.svg')
