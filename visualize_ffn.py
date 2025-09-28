#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import numpy as np
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


def heatmap_visualize(img, alpha, pred, vis_dir, img_path):
    # assert len(img.shape) == 3
    H, W = 32, 128
    # print(alpha.shape)
    alpha = alpha.reshape([-1, 16, 8]).cpu().numpy()
    # alpha = alpha.permute(0, 2, 1).cpu().numpy() 
    # print(alpha.shape)
    for i, att_map in enumerate(alpha):
        # print(i)
        # print(len(pred))
        if i >= len(pred):
            break
        # print(att_map)
        att_map = cv2.resize(att_map, (W, H))
        att_max = att_map.max()
        att_map /= att_max
        att_map *= 255
        # print(att_map)
        att_map = att_map.astype(np.uint8)
        heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)

        show_attention = img.copy()
        show_attention = cv2.addWeighted(heatmap, 0.5, show_attention, 0.5, 0)
        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])), show_attention)

    return True


def ffn_visualize(ffn, pred, img_dir, name):
    ffn = ffn.squeeze(0)[:len(pred), :]
    # calculate the cosine similarities
    ffn_map = np.zeros((len(pred), len(pred)))
    for i in range(len(pred)):
        for j in range(len(pred)):
            temp_sim = F.cosine_similarity(ffn[i], ffn[j], dim=0).cpu().numpy()
            ffn_map[i][j] = temp_sim

    ffn_map = cv2.resize(ffn_map, (50, 50))
    ffn_max = ffn_map.max()
    ffn_map /= ffn_max
    ffn_map *= 255

    ffn_map = ffn_map.astype(np.uint8)
    heatmap = cv2.applyColorMap(ffn_map, cv2.COLORMAP_SUMMER)
    # Y_label, X_label = list(pred), list(pred)
    cv2.imwrite(os.path.join(img_dir, "{}_{}.jpg".format(os.path.basename(name).split('.')[0], pred)), heatmap)
    return True


def ffn_visualize_new(ffn, pred, img_dir, name):
    ffn = ffn.squeeze(0)[:len(pred), :]
    # Normalize each row to make them unit vectors
    ffn_norm = ffn / ffn.norm(dim=1, keepdim=True)
    # Calculate the cosine similarity matrix
    cosine_similarity_matrix = torch.mm(ffn_norm, ffn_norm.t())
    # Convert to numpy for visualization
    ffn_map = cosine_similarity_matrix.cpu().detach().numpy()
    # ffn_map /= ffn_map.max()
    
    x_labels = list(pred)
    y_labels = list(pred)
    # Visualize the cosine similarity matrix as a heatmap with a color bar
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 10))
    plt.tick_params(labelsize=20)
    ax = sns.heatmap(ffn_map, annot=False, cmap="summer", cbar=False, square=True,
                     xticklabels=x_labels, yticklabels=y_labels, linewidths=0)
    # Show the color bar with label
    # cbar = ax.collections[0].colorbar
    # cbar.set_label('Cosine Similarity Value', labelpad=15, rotation=270)
    # Save the heatmap as a PNG file
    plt.savefig(os.path.join(img_dir, "{}.svg".format(os.path.basename(name).split('.')[0])), bbox_inches='tight')
    plt.close()
    return True


def ffn_visualize_multi(ffns, preds, titles, img_dir, name):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(30, 10))  # Adjust the figure size
    
    for idx, (ffn, pred, title) in enumerate(zip(ffns, preds, titles)):
        ffn = ffn.squeeze(0)[:len(pred), :]
        ffn_norm = ffn / ffn.norm(dim=1, keepdim=True)
        cosine_similarity_matrix = torch.mm(ffn_norm, ffn_norm.t())
        ffn_map = cosine_similarity_matrix.cpu().detach().numpy()
        ffn_map /= ffn_map.max()
        
        x_labels = list(pred)
        y_labels = list(pred)
        
        plt.subplot(1, 3, idx + 1)  # 1 row, 3 columns, index starting from 1
        plt.tick_params(labelsize=13)
        
        ax = sns.heatmap(ffn_map, annot=False, cmap="summer", cbar=False, square=True,
                         xticklabels=x_labels, yticklabels=y_labels)
        
        plt.title(title)  # Set the title for this subplot
        
        # Show the color bar with label, if you need it
        # cbar = ax.collections[0].colorbar
        # cbar.set_label('Cosine Similarity Value', labelpad=15, rotation=270)

    # Save the heatmap as a PNG file
    plt.savefig(os.path.join(img_dir, "{}.png".format(os.path.basename(name).split('.')[0])), bbox_inches='tight')
    
    plt.close()

    return True


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--womimic', default="/scratch/yang.xiaome/code/diffpimnet/outputs/pimnet_womimic/2025-02-01_22-50-07/checkpoints/epoch=8-step=88292-val_accuracy=88.3060-val_NED=95.0490.ckpt")
    # parser.add_argument('pimnet', help="Model checkpoint (or 'pretrained=<model_id>')")
    # parser.add_argument('diffusion', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images', default=["visualize/visual_1.jpg", "visualize/visual_2.jpg", "visualize/visual_3.jpg"])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--multi', action='store_true', help='whether store multiple ffn')
    parser.add_argument('--output_path', default='visualize/', help='where to store the visualization')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    # model_diffusion = load_from_checkpoint(args.diffusion, **kwargs).eval().to(args.device)
    model_womimic = load_from_checkpoint(args.womimic, **kwargs).eval().to(args.device)
    # model_pimnet = load_from_checkpoint(args.pimnet, **kwargs).eval().to(args.device)

    img_transform = SceneTextDataModule.get_transform(model_womimic.hparams.img_size)

    for fname in args.images:
        # Load image and prepare for input
        image = Image.open(fname).convert('RGB')
        # ori_img = np.array(image)
        # ori_img = cv2.resize(ori_img, (128, 32))
        image = img_transform(image).unsqueeze(0).to(args.device)
        # p_diffusion, alphas_diffusion = model_diffusion(image)
        # p_diffusion = p_diffusion.softmax(-1)
        # pred_diffusion, p_diffusion = model_diffusion.tokenizer.decode(p_diffusion)

        p_womimic, alphas_womimic = model_womimic(image)
        p_womimic = p_womimic.softmax(-1)
        pred_womimic, p_womimic = model_womimic.tokenizer.decode(p_womimic)

        # p_pimnet, alphas_pimnet = model_pimnet(image)
        # p_pimnet = p_pimnet.softmax(-1)
        # pred_pimnet, p_pimnet = model_pimnet.tokenizer.decode(p_pimnet)
        
        if args.multi:
            alpha_list = [alphas_womimic, alphas_pimnet, alphas_diffusion]
            pred_list = [pred_womimic[0], pred_pimnet[0], pred_diffusion[0]]
            titles_list = ['womimic', 'pimnet', 'diffusion']
            ffn_visualize_multi(alpha_list, pred_list, titles_list, args.output_path, fname)
        else:
            # ffn_visualize_new(alphas_diffusion, pred_diffusion[0], os.path.join(args.output_path, 'diffusion'), fname)
            # ffn_visualize_new(alphas_pimnet, pred_pimnet[0], os.path.join(args.output_path, 'pimnet'), fname)
            ffn_visualize_new(alphas_womimic, pred_womimic[0], os.path.join(args.output_path, 'womimic'), fname)
            # heatmap_visualize(ori_img, alphas, pred[0], 'visualize/test/', fname)

if __name__ == '__main__':
    main()
