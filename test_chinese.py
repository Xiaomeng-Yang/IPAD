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
import string
import sys
import os
import zhconv
from dataclasses import dataclass
from nltk import edit_distance
from typing import List

import torch

from tqdm import tqdm
from torchvision import utils as vutils

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


# conver the full_width to half_width
def Q2B(uchar):
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    return chr(inside_code)


def strQ2B(ustring):
    return "".join([Q2B(uchar) for uchar in ustring])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='/home/test13/yxm/data/chinese_benchmark_dataset/scene')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--store_result', action='store_true', default=False, help='Whether store the recognition results')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    
    with open('configs/charset/chinese_charset.txt', 'r') as file:
        chinese_charset = file.readline().rstrip('\n')
        
    charset_test = chinese_charset
    kwargs.update({'charset_test': charset_test})
    # print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation,
                                     train = False)

    dataloader = datamodule.test_dataloader_chinese()
    # print('num_samples:', dataloader.__len__())
    total = 0
    correct = 0
    correct_convert = 0
    ned = 0
    ned_convert = 0
    label_length = 0
    
    if args.store_result:
        output_dir = os.path.join('./visualize', 'womimic_iter5')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = open(os.path.join(output_dir, 'result.txt'), 'w', encoding='utf-8')
        output_false_file = open(os.path.join(output_dir, 'result_false.txt'), 'w', encoding='utf-8')
        output_file.write('Idx\tGT\tPredict\n')
        output_false_file.write('Idx\tGT\tPredict\n')
        
    idx = 0

    for imgs, labels in tqdm(iter(dataloader)):
        res = model.test_step((imgs.to(model.device), labels), -1)['output']
        total += res.num_samples
        correct += res.correct
        ned += res.ned
        label_length += res.label_length

        for i in range(len(res.pred_list)):
            # convert the ground truth and predictions
            gt = zhconv.convert(strQ2B(res.gt_list[i]), 'zh-cn')
            pred = zhconv.convert(strQ2B(res.pred_list[i]), 'zh-cn')
            gt = gt.lower()
            pred = pred.lower()
            ned_convert += edit_distance(pred, gt) / max(len(pred), len(gt))
            if gt == pred:
                correct_convert += 1

            if args.store_result:
                # print(res.gt_list[i] + '\t' + res.pred_list[i] + '\n')
                output_file.write(str(idx) + '\t' + gt + '\t' + pred + '\n')
                if gt != pred:
                    output_false_file.write(str(idx) + '\t' + gt + '\t' + pred + '\n')
                    img_name = str(idx) + '.png'
                    img_path = os.path.join(output_dir, img_name)
                    cur_image = imgs[i].to(torch.device('cpu'))
                    vutils.save_image(cur_image, img_path)                        
                    
            idx += 1

    accuracy = 100 * correct / total
    acc_convert = 100 * correct_convert / total
    mean_ned = 100 * (1 - ned / total)
    mean_ned_convert = 100 * (1 - ned_convert / total)
    mean_label_length = label_length / total

    with open(args.checkpoint + '.log.txt', 'w') as f:
        f.write('Accuracy: '+ str(accuracy) + ' | Acc_convert:' + str(acc_convert))

    print('Accuracy:', accuracy)
    print('Accuracy_convert:', acc_convert)
    print('NED:', mean_ned)
    print('NED_convert:', mean_ned_convert)
    print('Label_Length:', mean_label_length)
        
if __name__ == '__main__':
    main()
