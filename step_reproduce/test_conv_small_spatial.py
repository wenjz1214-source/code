"""Reproduce Conv2D backward NaN on small HxW (H100 + TF2.6 PTX JIT)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

# Approximate ResNet-OS32 feature map: 385/32 x 1249/32
sizes = [
    (12, 39), (13, 40), (24, 78), (48, 157), (64, 64), (96, 96), (128, 128),
]
print('Conv2D(64, 7, same) backward — kernel NaN?')
for h, w in sizes:
    inp = tf.random.normal([2, h, w, 256])
    layer = tf.keras.layers.Conv2D(64, 7, padding='same')
    with tf.GradientTape() as tape:
        out = layer(inp)
        loss = tf.reduce_mean(out ** 2)
    g = tape.gradient(loss, layer.kernel)
    knan = bool(tf.reduce_any(tf.math.is_nan(g)).numpy())
    print(f'  ({h:3d},{w:3d}): kernel_nan={knan}')

print('\nConv2D(256, 3, same) — typical bottleneck')
for h, w in [(12, 39), (24, 78), (48, 157)]:
    inp = tf.random.normal([2, h, w, 512])
    layer = tf.keras.layers.Conv2D(256, 3, padding='same')
    with tf.GradientTape() as tape:
        out = layer(inp)
        loss = tf.reduce_mean(out ** 2)
    g = tape.gradient(loss, layer.kernel)
    knan = bool(tf.reduce_any(tf.math.is_nan(g)).numpy())
    print(f'  ({h:3d},{w:3d}): kernel_nan={knan}')
print('Done.')
