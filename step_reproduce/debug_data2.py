"""Debug: test with gradient clipping and TF32 disabled."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import tensorflow as tf
import numpy as np

tf.config.experimental.enable_tensor_float_32_execution(False)

from google.protobuf import text_format
from deeplab2 import config_pb2, common
from deeplab2.trainer import runner_utils
from deeplab2.data import dataset as dataset_lib
from deeplab2.model import deeplab
from deeplab2.model.loss import loss_builder

config = config_pb2.ExperimentOptions()
with open('deeplab2/configs/kitti/motion_deeplab/resnet50_os32.textproto', 'r') as f:
    text_format.Parse(f.read(), config)

dataset_info = dataset_lib.MAP_NAME_TO_DATASET_INFO['kitti_step']
train_dataset = runner_utils.create_dataset(
    config.train_dataset_options, is_training=True, only_semantic_annotations=False)
iterator = iter(train_dataset)

print("Building model (TF32 disabled)...")
model = deeplab.DeepLab(config, dataset_info)

loss_fn = loss_builder.DeepLabFamilyLoss(
    loss_options=config.trainer_options.loss_options,
    deeplab_options=config.model_options,
    num_classes=dataset_info.num_classes,
    ignore_label=dataset_info.ignore_label,
    ignore_depth=dataset_info.ignore_depth if hasattr(dataset_info, 'ignore_depth') else 0.0,
    thing_class_ids=dataset_info.class_has_instances_list,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1.25e-5)

print("Running training steps (TF32 disabled)...")
for step in range(10):
    batch = next(iterator)
    with tf.GradientTape() as tape:
        outputs = model(batch[common.IMAGE], training=True)
        loss_dict = loss_fn(batch, outputs)
        total_loss = tf.reduce_mean(loss_dict['total_loss'])

    grads = tape.gradient(total_loss, model.trainable_variables)
    valid_pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]

    clipped_grads, global_norm = tf.clip_by_global_norm([g for g, _ in valid_pairs], 10.0)
    clipped_pairs = list(zip(clipped_grads, [v for _, v in valid_pairs]))

    grad_norms = [tf.norm(g).numpy() for g in clipped_grads]
    nan_grads = sum(1 for gn in grad_norms if np.isnan(gn))

    optimizer.apply_gradients(clipped_pairs)

    print(f"Step {step}: loss={total_loss.numpy():.6f} nan={np.isnan(total_loss.numpy())}, "
          f"global_norm={global_norm.numpy():.4f}, nan_grads={nan_grads}/{len(grad_norms)}")

print("Done!")
