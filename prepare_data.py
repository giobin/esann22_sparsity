# import os
# from glob import glob
# import shutil
# from tqdm import tqdm
#
# def main(args):
#     train_data_path = args.train_data_path
#     valid_data_path = args.valid_data_path
#     train_label_path = args.train_label_path
#     valid_label_path = args.valid_label_path
#
#     #valid_images_path = glob(os.path.join(valid_data_path, '*'))
#
#     # organize valid
#     # with open(valid_label_path, 'r', encoding='utf-8') as vl:
#     #     v_lines = vl.readlines()
#     #     v_lines = [v.strip().split() for v in v_lines]
#     #     names = [os.path.splitext(n)[0] for n, i in v_lines]
#     #     labels = [i for n, i in v_lines]
#     #
#     # # create valid folder using labels
#     # labels_set = set(labels)
#     # for l in labels_set:
#     #     l = os.path.join(valid_data_path, l)
#     #     if not os.path.exists(l):
#     #         os.makedirs(l)
#     #
#     # # mv images inside label folder
#     # moved = 0
#     # for image in tqdm(valid_images_path):
#     #     image_name = os.path.splitext(os.path.basename(image))[0]
#     #     for n, l in zip(names, labels):
#     #         if n == image_name:
#     #             _ = shutil.move(image, os.path.join(valid_data_path, l))
#     #             moved += 1
#     #             break
#
#     # organize train
#     # crea cartelle da 0 a 999 in train/
#     for l in range(1000):
#         l = os.path.join(train_data_path, str(l))
#         if not os.path.exists(l):
#              os.makedirs(l)
#
#     # apri train.txt
#     with open(train_label_path, 'r', encoding='utf-8') as tl:
#         t_lines = tl.readlines()
#         t_lines = [t.strip().split() for t in t_lines]
#         names = [os.path.splitext(n)[0] for n, i in t_lines]
#         labels = [i for n, i in t_lines]
#
#         # trova le img by name
#     for n, i in zip(names, labels):
#         n = os.path.join(train_data_path, n + '.jpeg')
#         _ = shutil.move(n, os.path.join(train_data_path, i))
#         # usa il nome per spostare quel file in label (occhio a .jpeg, .JPEG)
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-trd','--train_data_path', type=str)
#     parser.add_argument('-vad', '--valid_data_path', type=str)
#     parser.add_argument('-trl','--train_label_path', type=str)
#     parser.add_argument('-val', '--valid_label_path', type=str)
#     args = parser.parse_args()
#     main(args)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import os.path
import sys
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-vad', '--valid_data_path', type=str)
  parser.add_argument('-val', '--valid_label_path', type=str)
  args = parser.parse_args()

  data_dir = args.valid_data_path
  validation_labels_file = args.valid_label_path

  # Read in the 50000 synsets associated with the validation data set.
  labels = [l.strip() for l in open(validation_labels_file).readlines()]
  unique_labels = set(labels)

  # Make all sub-directories in the validation data dir.
  for label in unique_labels:
    labeled_data_dir = os.path.join(data_dir, label)
    # Catch error if sub-directory exists
    try:
      os.makedirs(labeled_data_dir)
    except OSError as e:
      # Raise all errors but 'EEXIST'
      if e.errno != errno.EEXIST:
        raise

  # Move all of the image to the appropriate sub-directory.
  for i in range(len(labels)):
    basename = 'ILSVRC2012_val_000%.5d.JPEG' % (i + 1)
    original_filename = os.path.join(data_dir, basename)
    if not os.path.exists(original_filename):
      print('Failed to find: %s' % original_filename)
      sys.exit(-1)
    new_filename = os.path.join(data_dir, labels[i], basename)
    os.rename(original_filename, new_filename)