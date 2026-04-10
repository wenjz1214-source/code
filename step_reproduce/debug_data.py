"""Debug: test if gradient explosion causes NaN."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

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

print("Building model...")
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

print("Running training steps...")
for step in range(10):
    batch = next(iterator)
    with tf.GradientTape() as tape:
        outputs = model(batch[common.IMAGE], training=True)
        loss_dict = loss_fn(batch, outputs)
        avg_losses = {}
        for name, loss in loss_dict.items():
            v = tf.reduce_mean(loss)
            avg_losses[name] = tf.where(tf.math.is_nan(v), 0.0, v)
        total_loss = avg_losses['total_loss']

    grads = tape.gradient(total_loss, model.trainable_variables)
    valid_grads = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    grad_norms = [tf.norm(g).numpy() for g, _ in valid_grads]
    max_grad = max(grad_norms) if grad_norms else 0
    nan_grads = sum(1 for gn in grad_norms if np.isnan(gn))

    optimizer.apply_gradients(valid_grads)

    raw_total = tf.reduce_mean(loss_dict['total_loss']).numpy()
    print(f"Step {step}: raw_total={raw_total:.6f} nan={np.isnan(raw_total)}, "
          f"max_grad={max_grad:.4f}, nan_grads={nan_grads}/{len(grad_norms)}")

    for name, loss in loss_dict.items():
        v = tf.reduce_mean(loss).numpy()
        if np.isnan(v) or v > 100:
            print(f"  WARNING: {name} = {v}")

print("Done!")
