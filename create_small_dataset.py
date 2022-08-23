import os
import sys
import shutil
from glob import glob
import random

train_origin = '../imagenet/train_v2/'
val_origin = '../imagenet/valid_v2/'

train_destination= '../imagenet/train_v2_small/'
val_destination = '../imagenet/valid_v2_small/'

train_origin_dirs = os.listdir(train_origin)
val_origin_dirs = os.listdir(val_origin)

for d in train_origin_dirs:
    images = glob(os.path.join(os.path.join(train_origin, d), '*.JPEG'))
    assert images
    picked_imgs = random.sample(images, 100)
    for img in picked_imgs:
        name = os.path.basename(img)
        dest = shutil.copyfile(img, os.path.join(os.path.join(train_destination, d), name))
        print(dest)

for d in val_origin_dirs:
    images = glob(os.path.join(os.path.join(val_origin, d), '*.JPEG'))
    assert images
    picked_imgs = random.sample(images, 5)
    for img in picked_imgs:
        name = os.path.basename(img)
        dest = shutil.copyfile(img, os.path.join(os.path.join(val_destination, d), name))
        print(dest)

