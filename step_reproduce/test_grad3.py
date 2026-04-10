import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
sys.stdout.reconfigure(line_buffering=True)

print('=== Conv2D gradient NaN test ===', flush=True)
for shape in [[2,32,32,3], [2,64,64,3], [2,128,128,3], [2,256,256,3]]:
    inp = tf.random.normal(shape)
    layer = tf.keras.layers.Conv2D(64, 7, padding='same')
    with tf.GradientTape() as tape:
        out = layer(inp)
        loss = tf.reduce_mean(out**2)
    grads = tape.gradient(loss, layer.trainable_variables)
    knan = tf.reduce_any(tf.math.is_nan(grads[0])).numpy()
    print(f'  {shape}: kernel_nan={knan}', flush=True)

print('\n=== DeepLab small crop ===', flush=True)
from deeplab2 import config_pb2, common
from deeplab2.data import dataset as dataset_lib
from deeplab2.model import deeplab
from deeplab2.model.loss import loss_builder
from google.protobuf import text_format

config = config_pb2.ExperimentOptions()
with open('deeplab2/configs/kitti/motion_deeplab/resnet50_os32.textproto', 'r') as f:
    text_format.Parse(f.read(), config)
dataset_info = dataset_lib.MAP_NAME_TO_DATASET_INFO['kitti_step']

model = deeplab.DeepLab(config, dataset_info)
loss_fn = loss_builder.DeepLabFamilyLoss(
    loss_options=config.trainer_options.loss_options,
    deeplab_options=config.model_options,
    num_classes=dataset_info.num_classes,
    ignore_label=dataset_info.ignore_label,
    ignore_depth=0.0,
    thing_class_ids=dataset_info.class_has_instances_list,
)

inp = tf.random.uniform([1, 385, 1249, 7], 0, 255)
with tf.GradientTape() as tape:
    outputs = model(inp, training=True)
    fake_batch = {
        common.IMAGE: inp,
        'semantic_weights': tf.ones([1, 385, 1249]),
        common.GT_SEMANTIC_KEY: tf.zeros([1, 385, 1249], dtype=tf.int32),
        common.GT_INSTANCE_CENTER_KEY: tf.zeros([1, 385, 1249]),
        common.GT_INSTANCE_REGRESSION_KEY: tf.zeros([1, 385, 1249, 2]),
        common.GT_FRAME_OFFSET_KEY: tf.zeros([1, 385, 1249, 2]),
        common.GT_SIZE_KEY: tf.constant([[385, 1249]]),
    }
    loss_dict = loss_fn(fake_batch, outputs)
    total = tf.reduce_mean(loss_dict['total_loss'])
grads = tape.gradient(total, model.trainable_variables)
nan_count = sum(1 for g in grads if g is not None and tf.reduce_any(tf.math.is_nan(g)).numpy())
total_g = len([g for g in grads if g is not None])
print(f'  DeepLab: loss={total.numpy():.4f}, nan_grads={nan_count}/{total_g}', flush=True)

if nan_count > 0:
    for g, v in zip(grads, model.trainable_variables):
        if g is not None and tf.reduce_any(tf.math.is_nan(g)).numpy():
            print(f'    NaN: {v.name}', flush=True)

print('Done!', flush=True)
