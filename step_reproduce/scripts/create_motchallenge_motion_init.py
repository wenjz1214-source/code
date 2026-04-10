#!<STEP_PYTHON>
"""Create a MOTChallenge-STEP Motion-DeepLab init checkpoint.

This combines:
1. the 7-channel first-layer Motion-DeepLab checkpoint, and
2. a Cityscapes semantic last layer remapped to MOTChallenge-STEP classes.
"""

from pathlib import Path

import tensorflow as tf
from google.protobuf import text_format

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset as dataset_lib
from deeplab2.model import deeplab
from deeplab2.video.motion_deeplab import MotionDeepLab


CITYSCAPES_TO_MOT = (1, 2, 8, 10, 11, 12, 18)
SOURCE_PANOPTIC_CKPT = '<KITTI_STEP_ROOT>/checkpoints/resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine/ckpt-60000'
SOURCE_MOTION_FIRST_CKPT = '<KITTI_STEP_ROOT>/checkpoints/resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine_netsurgery_first_layer/pretrained-1'
OUTPUT_PREFIX = '<MOTCHALLENGE_STEP_ROOT>/checkpoints/motion_deeplab_motchallenge_first_and_last/pretrained'


def _parse_proto(path: str) -> config_pb2.ExperimentOptions:
  cfg = config_pb2.ExperimentOptions()
  text_format.Parse(Path(path).read_text(), cfg)
  return cfg


def _load_cityscapes_panoptic_weights():
  cfg = _parse_proto('deeplab2/configs/kitti/panoptic_deeplab/resnet50_os32.textproto')
  cfg.model_options.initial_checkpoint = SOURCE_PANOPTIC_CKPT
  info = dataset_lib.CITYSCAPES_PANOPTIC_INFORMATION
  model = deeplab.DeepLab(cfg, info)
  model(tf.keras.Input([385, 1249, 3]), training=False)
  ckpt = tf.train.Checkpoint(**model.checkpoint_items)
  ckpt.read(SOURCE_PANOPTIC_CKPT).expect_partial().assert_existing_objects_matched()
  kernel, bias = model._decoder._semantic_head.final_conv.get_weights()  # pylint: disable=protected-access
  kernel = kernel[:, :, :, CITYSCAPES_TO_MOT]
  bias = bias[list(CITYSCAPES_TO_MOT)]
  return kernel, bias


def _build_target_motion_model():
  cfg = _parse_proto('deeplab2/configs/motchallenge/motion_deeplab/resnet50_os32.textproto')
  cfg.model_options.initial_checkpoint = SOURCE_MOTION_FIRST_CKPT
  cfg.model_options.restore_semantic_last_layer_from_initial_checkpoint = False
  cfg.trainer_options.solver_options.use_sync_batchnorm = False
  cfg.trainer_options.solver_options.batchnorm_epsilon = 0.001
  cfg.train_dataset_options.crop_size[:] = [545, 961]
  cfg.eval_dataset_options.crop_size[:] = [545, 961]
  info = dataset_lib.MOTCHALLENGE_STEP_INFORMATION
  model = MotionDeepLab(cfg, info)
  model(tf.keras.Input([545, 961, 6]), training=False)
  restore_dict = dict(model.checkpoint_items)
  del restore_dict[common.CKPT_SEMANTIC_LAST_LAYER]
  ckpt = tf.train.Checkpoint(**restore_dict)
  ckpt.read(SOURCE_MOTION_FIRST_CKPT).expect_partial().assert_nontrivial_match()
  return model


def main():
  kernel, bias = _load_cityscapes_panoptic_weights()
  model = _build_target_motion_model()
  model._decoder._semantic_head.final_conv.set_weights([kernel, bias])  # pylint: disable=protected-access

  out_dir = Path(OUTPUT_PREFIX).parent
  out_dir.mkdir(parents=True, exist_ok=True)
  ckpt = tf.train.Checkpoint(**model.checkpoint_items)
  saved = ckpt.save(OUTPUT_PREFIX)
  print(saved)


if __name__ == '__main__':
  main()
