import lmdb
import os
import re
import io
import argparse
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', type=str, required=True, help='path to lmdb')
    parser.add_argument('--img_dir', type=str, required=True, help='path to store the images')
    args = parser.parse_args()

    # create a target directory
    root = args.img_dir
    output_dir = os.path.join(root, 'image')
    if not os.path.exists(root):
        os.makedirs(root)
        # then make the subdirectory to store the imgs
        os.makedirs(output_dir)

    # read the lmdb dataset
    env = lmdb.open(args.lmdb_path, readonly=True)
    txn = env.begin()

    # the file to store the labels
    label_file = open(os.path.join(root, 'label.txt'), 'w', encoding='utf-8')
    clean_file = open(os.path.join(root, 'label_36char.txt'), 'w', encoding='utf-8')

    num_samples = int(txn.get('num-samples'.encode()))
    charset = "0123456789abcdefghijklmnopqrstuvwxyz"
    unsupport = f'[^{re.escape(charset)}]'

    for index in range(num_samples):
        index += 1     # lmdb starts with 1
        label_key = f'label-{index:09d}'.encode()
        label = txn.get(label_key).decode()

        # process the labels for 36-char
        clean_label = label.lower()
        # Remove the unsupported characters
        clean_label = re.sub(unsupport, '', clean_label)

        img_key = f'image-{index:09d}'.encode()
        buf = io.BytesIO(txn.get(img_key))
        img = Image.open(buf)
        img_name = str(index) + '.png'
        img_path = os.path.join(output_dir, img_name)
        img.save(img_path)

        label_file.write(img_name+'\t'+label+'\n')
        clean_file.write(img_name+'\t'+clean_label+'\n')


