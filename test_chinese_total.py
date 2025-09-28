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
import time
from dataclasses import dataclass
from nltk import edit_distance
from typing import List

import torch
import numpy as np

from tqdm import tqdm
from torchvision import utils as vutils

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float
    infer_time: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        c.infer_time += res.infer_time
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    c.infer_time /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)
    print('infer_time:', c.infer_time)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='/home/yangxiaomeng5/data/chinese_benchmark_dataset/scene/test')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--store_result', action='store_true', default=False, help='Whether store the recognition results')
    parser.add_argument('--output_file', type=str, default='test.txt', help='The file to store the predictions')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    
    with open('configs/charset/chinese_charset.txt', 'r') as file:
        chinese_charset = file.readline().rstrip('\n')
        
    charset_test = chinese_charset
    kwargs.update({'charset_test': charset_test})
    # print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    params_num = sum([np.prod(p.size()) for p in model.parameters()])
    print("parameters:", params_num/1e6)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation,
                                     train=False)

    test_set = SceneTextDataModule.TEST_CHINESE
    test_set = sorted(set(test_set))
    results={}
    max_width =max(map(len, test_set))
    
    if args.store_result:
        # output_dir = os.path.join('./visualize', 'chinese_iter5')
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        output_file = open(args.output_file, 'w', encoding='utf-8')
        # output_file.write('GT\tPredict\tT/F\n')
        
    for name, dataloader in datamodule.test_dataloaders_chinese(test_set).items():
        total = 0
        correct = 0
        infer_time = 0
        ned = 0
        confidence = 0
        label_length = 0

        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            start_time = time.time()
            res = model.test_step((imgs.to(model.device), labels), -1, Chinese=True)['output']
            forward_time = time.time() - start_time
            infer_time += forward_time
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length

            '''
            if args.store_result:
                for i in range(len(res.pred_list)):
                    # print(res.gt_list[i] + '\t' + res.pred_list[i] + '\n')
                    # output_file.write(res.gt_list[i] + '\t' + res.pred_list[i] + '\n')
                    output_file.write(str(idx)+'\t'+ res.gt_list[i] + '\t' + res.pred_list[i] + '\n')
                    # else:
                    #     output_file.write(res.gt_list[i] + '\t' + res.pred_list[i] + '\tF\n')
            '''        

        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length, infer_time)

    result_groups = {
        'Chinese': SceneTextDataModule.TEST_CHINESE
    }

    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)
        

if __name__ == '__main__':
    main()
