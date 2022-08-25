#!/usr/bin/env python3

# MODEL_DIR The directory of the exported ViP-DeepLab model.
MODEL_URL = 'https://storage.googleapis.com/gresearch/tf-deeplab/saved_model/resnet50_beta_os32_vip_deeplab_cityscapes_dvps_train_saved_model.tar.gz' #@param {type:"string"}

# SEQUENCE_PATTERN The file name pattern for the input sequence.
SEQUENCE_PATTERN = 'input/*.jpg' #@param {type:"string"}

# LABEL_DIVISOR The label divisor for the dataset.
LABEL_DIVISOR = 1000 #@param {type:"integer"}

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy
import collections
import typing
import tempfile
import urllib

from vipdeeplab import ViPDeepLab

#@title Visualization Utilities

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    'num_classes, label_divisor, thing_list, colormap, class_names')


def _cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.

  See more about CITYSCAPES dataset at https://www.cityscapes-dataset.com/
  M. Cordts, et al. "The Cityscapes Dataset for Semantic Urban Scene Understanding." CVPR. 2016.

  Returns:
    A 2-D numpy array with each row being mapped RGB color (in uint8 range).
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap


def _cityscapes_class_names():
  return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
          'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
          'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
          'bicycle')


def cityscapes_dataset_information():
  return DatasetInfo(
      num_classes=19,
      label_divisor=1000,
      thing_list=tuple(range(11, 19)),
      colormap=_cityscapes_label_colormap(),
      class_names=_cityscapes_class_names())


def perturb_color(color, noise, used_colors, max_trials=50, random_state=None):
  """Pertrubs the color with some noise.

  If `used_colors` is not None, we will return the color that has
  not appeared before in it.

  Args:
    color: A numpy array with three elements [R, G, B].
    noise: Integer, specifying the amount of perturbing noise (in uint8 range).
    used_colors: A set, used to keep track of used colors.
    max_trials: An integer, maximum trials to generate random color.
    random_state: An optional np.random.RandomState. If passed, will be used to
      generate random numbers.

  Returns:
    A perturbed color that has not appeared in used_colors.
  """
  if random_state is None:
    random_state = np.random

  for _ in range(max_trials):
    random_color = color + random_state.randint(
        low=-noise, high=noise + 1, size=3)
    random_color = np.clip(random_color, 0, 255)

    if tuple(random_color) not in used_colors:
      used_colors.add(tuple(random_color))
      return random_color

  print('Max trial reached and duplicate color will be used. Please consider '
        'increase noise in `perturb_color()`.')
  return random_color


def color_panoptic_map(panoptic_prediction,
                       dataset_info,
                       perturb_noise,
                       used_colors,
                       color_mapping):
  """Helper method to colorize output panoptic map.

  Args:
    panoptic_prediction: A 2D numpy array, panoptic prediction from deeplab
      model.
    dataset_info: A DatasetInfo object, dataset associated to the model.
    perturb_noise: Integer, the amount of noise (in uint8 range) added to each
      instance of the same semantic class.
    used_colors: A set, used to keep track of used colors.
    color_mapping: A dict, used to map exisiting panoptic ids.

  Returns:
    colored_panoptic_map: A 3D numpy array with last dimension of 3, colored
      panoptic prediction map.
    used_colors: A dictionary mapping semantic_ids to a set of colors used
      in `colored_panoptic_map`.
  """
  if panoptic_prediction.ndim != 2:
    raise ValueError('Expect 2-D panoptic prediction. Got {}'.format(
        panoptic_prediction.shape))

  semantic_map = panoptic_prediction // dataset_info.label_divisor
  instance_map = panoptic_prediction % dataset_info.label_divisor
  height, width = panoptic_prediction.shape
  colored_panoptic_map = np.zeros((height, width, 3), dtype=np.uint8)

  # Use a fixed seed to reproduce the same visualization.
  random_state = np.random.RandomState(0)

  unique_semantic_ids = np.unique(semantic_map)
  for semantic_id in unique_semantic_ids:
    semantic_mask = semantic_map == semantic_id
    if semantic_id in dataset_info.thing_list:
      # For `thing` class, we will add a small amount of random noise to its
      # correspondingly predefined semantic segmentation colormap.
      unique_instance_ids = np.unique(instance_map[semantic_mask])
      for instance_id in unique_instance_ids:
        instance_mask = np.logical_and(semantic_mask,
                                       instance_map == instance_id)
        panoptic_id = semantic_id * dataset_info.label_divisor + instance_id
        if panoptic_id not in color_mapping:
          random_color = perturb_color(
              dataset_info.colormap[semantic_id],
              perturb_noise,
              used_colors[semantic_id],
              random_state=random_state)
          colored_panoptic_map[instance_mask] = random_color
          color_mapping[panoptic_id] = random_color
        else:
          colored_panoptic_map[instance_mask] = color_mapping[panoptic_id]
    else:
      # For `stuff` class, we use the defined semantic color.
      colored_panoptic_map[semantic_mask] = dataset_info.colormap[semantic_id]
      used_colors[semantic_id].add(tuple(dataset_info.colormap[semantic_id]))
  return colored_panoptic_map

if __name__ == "__main__":
    #@title Run Inference on Examples from Cityscapes-DVPS
    model_path = "resnet50_beta_os32_vip_deeplab_cityscapes_dvps_train_saved_model/exports"
    vip_deeplab = ViPDeepLab(model_path=model_path, label_divisor=LABEL_DIVISOR)
    filenames = sorted(tf.io.gfile.glob(SEQUENCE_PATTERN))[0:3]
    inputs = []
    for filename in filenames:
      inputs.append(tf.image.decode_png(tf.io.read_file(filename)))
    inputs.append(inputs[-1])
    vip_deeplab.infer(inputs)
    depth_preds, stitched_panoptic = vip_deeplab.results()

    #@title Visualize the Predictions
    used_colors = collections.defaultdict(set)
    color_mapping = dict()
    for i in range(len(filenames)):
      fig, ax = plt.subplots(1, 3, figsize=(18, 6))
      ax[0].title.set_text('Input Image')
      ax[0].imshow(np.squeeze(inputs[i]))
      ax[1].title.set_text('Depth')
      ax[1].imshow(np.squeeze(depth_preds[i]))
      panoptic = stitched_panoptic[i]
      ax[2].title.set_text('Video Panoptic Segmentation')
      panoptic_map = color_panoptic_map(
          np.squeeze(panoptic), cityscapes_dataset_information(), 60, used_colors,
          color_mapping)
      ax[2].imshow(panoptic_map)

    plt.show()

