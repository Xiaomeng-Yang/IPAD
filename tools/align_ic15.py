import lmdb
import os
import re
import io
import argparse
import hashlib
from PIL import Image


def ImageID(img):
    return hashlib.md5(img.tobytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parseq_path', type=str, required=True, help='path to parseq lmdb')
    parser.add_argument('--abinet_path', type=str, required=True, help='path to abinet lmdb')
    parser.add_argument('--img_dir', type=str, required=True, help='path to store the images')
    args = parser.parse_args()

    # create a target directory
    root = args.img_dir
    output_dir = os.path.join(root, 'image')
    if not os.path.exists(root):
        os.makedirs(root)
        # then make the subdirectory to store the imgs
        os.makedirs(output_dir)

    # read the lmdb datasets
    parseq_env = lmdb.open(args.parseq_path, readonly=True)
    parseq_txn = parseq_env.begin()
    abinet_env = lmdb.open(args.abinet_path, readonly=True)
    abinet_txn = abinet_env.begin()

    # the files to store the labels
    parseq_label_file = open(os.path.join(root, 'label_parseq.txt'), 'w', encoding='utf-8')
    abinet_label_file = open(os.path.join(root, 'label_abinet.txt'), 'w', encoding='utf-8')
    # the files to store the labels for 36-char
    parseq_clean_file = open(os.path.join(root, 'label_parseq_36char.txt'), 'w', encoding='utf-8')
    abinet_clean_file = open(os.path.join(root, 'label_abinet_36char.txt'), 'w', encoding='utf-8')

    num_samples = int(abinet_txn.get('num-samples'.encode()))
    charset = "0123456789abcdefghijklmnopqrstuvwxyz"
    unsupport = f'[^{re.escape(charset)}]'

    # process the images and labels for parseq, and store them to a dictionary
    parseq_imgs = {}
    for index in range(num_samples):
        index += 1     # lmdb starts with 1
        label_key = f'label-{index:09d}'.encode()
        label = parseq_txn.get(label_key).decode()
        img_key = f'image-{index:09d}'.encode()
        buf = io.BytesIO(parseq_txn.get(img_key))
        img = Image.open(buf)
        parseq_imgs[ImageID(img)] = label

    for index in range(num_samples):
        # get the image from the abinet lmdb
        index += 1     # lmdb starts with 1
        label_key = f'label-{index:09d}'.encode()
        label = abinet_txn.get(label_key).decode()
        # process the labels for 36-char
        clean_label = label.lower()
        # Remove the unsupported characters
        clean_label = re.sub(unsupport, '', clean_label)
        img_key = f'image-{index:09d}'.encode()
        buf = io.BytesIO(abinet_txn.get(img_key))
        img = Image.open(buf)
        img_name = str(index) + '.png'
        img_path = os.path.join(output_dir, img_name)
        img.save(img_path)

        abinet_label_file.write(img_name+'\t'+label+'\n')
        abinet_clean_file.write(img_name+'\t'+clean_label+'\n')

        # Find the corresponding parseq image and label
        parseq_label = parseq_imgs[ImageID(img)]
        parseq_clean_label = parseq_label.lower()
        parseq_clean_label = re.sub(unsupport, '', parseq_clean_label)
        parseq_label_file.write(img_name+'\t'+parseq_label+'\n')
        parseq_clean_file.write(img_name+'\t'+parseq_clean_label+'\n')

