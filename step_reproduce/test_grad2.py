import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

print('=== Conv2D with different configs ===')
configs = [
    ([2, 64, 64, 3], 16, 3),
    ([2, 64, 64, 7], 16, 3),
    ([2, 64, 64, 3], 16, 7),
    ([2, 128, 128, 3], 64, 7),
    ([4, 385, 1249, 3], 64, 7),
    ([4, 385, 1249, 7], 64, 7),
]
for shape, filters, ks in configs:
    inp = tf.random.normal(shape)
    layer = tf.keras.layers.Conv2D(filters, ks, padding='same')
    with tf.GradientTape() as tape:
        out = layer(inp)
        loss = tf.reduce_mean(out**2)
    grads = tape.gradient(loss, layer.trainable_variables)
    kernel_nan = tf.reduce_any(tf.math.is_nan(grads[0])).numpy()
    print(f'  input={shape}, filters={filters}, ks={ks}: kernel_nan={kernel_nan}, loss={loss.numpy():.6f}')

print('\n=== Test DeepLab model with smaller crop ===')
from google.protobuf import text_format
from deeplab2 import config_pb2, common
from deeplab2.data import dataset as dataset_lib
from deeplab2.model import deeplab

config = config_pb2.ExperimentOptions()
with open('deeplab2/configs/kitti/motion_deeplab/resnet50_os32.textproto', 'r') as f:
    text_format.Parse(f.read(), config)

dataset_info = dataset_lib.MAP_NAME_TO_DATASET_INFO['kitti_step']
model = deeplab.DeepLab(config, dataset_info)

for h, w in [(129, 129), (193, 641), (385, 1249)]:
    inp = tf.random.normal([1, h, w, 7])
    with tf.GradientTape() as tape:
        outputs = model(inp, training=True)
        sem = outputs['semantic_logits']
        loss = tf.reduce_mean(sem**2)
    grads = tape.gradient(loss, model.trainable_variables)
    nan_count = sum(1 for g in grads if g is not None and tf.reduce_any(tf.math.is_nan(g)).numpy())
    total = len([g for g in grads if g is not None])
    print(f'  crop={h}x{w}: nan_grads={nan_count}/{total}, loss={loss.numpy():.6f}')

print('Done!')
