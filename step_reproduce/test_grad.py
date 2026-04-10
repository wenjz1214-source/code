import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

print('=== Basic gradient ===')
x = tf.Variable([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    y = tf.reduce_sum(x ** 2)
g = tape.gradient(y, x)
print(f'grad={g.numpy()} (expect [2,4,6])')

print('=== Conv2D ===')
inp = tf.random.normal([2, 64, 64, 3])
layer = tf.keras.layers.Conv2D(16, 3, padding='same')
with tf.GradientTape() as tape:
    out = layer(inp)
    loss = tf.reduce_mean(out**2)
grads = tape.gradient(loss, layer.trainable_variables)
for i, g in enumerate(grads):
    print(f'  var{i}: norm={tf.norm(g).numpy():.4f}, nan={tf.reduce_any(tf.math.is_nan(g)).numpy()}')

print('=== SyncBatchNorm ===')
sbn = tf.keras.layers.experimental.SyncBatchNormalization()
conv = tf.keras.layers.Conv2D(16, 3, padding='same')
inp2 = tf.random.normal([4, 32, 32, 3])
with tf.GradientTape() as tape:
    out = sbn(conv(inp2), training=True)
    loss = tf.reduce_mean(out**2)
all_vars = conv.trainable_variables + sbn.trainable_variables
grads = tape.gradient(loss, all_vars)
for i, g in enumerate(grads):
    if g is not None:
        print(f'  var{i}: norm={tf.norm(g).numpy():.6f}, nan={tf.reduce_any(tf.math.is_nan(g)).numpy()}')

print('=== ResNet50 ===')
resnet = tf.keras.applications.ResNet50(weights=None, input_shape=(128, 128, 3), classes=19)
inp3 = tf.random.normal([2, 128, 128, 3])
with tf.GradientTape() as tape:
    out = resnet(inp3, training=True)
    loss = tf.reduce_mean(out**2)
grads = tape.gradient(loss, resnet.trainable_variables)
nan_count = sum(1 for g in grads if g is not None and tf.reduce_any(tf.math.is_nan(g)).numpy())
total = len([g for g in grads if g is not None])
print(f'  nan_grads={nan_count}/{total}')
if nan_count > 0:
    for i, (g, v) in enumerate(zip(grads, resnet.trainable_variables)):
        if g is not None and tf.reduce_any(tf.math.is_nan(g)).numpy():
            print(f'  NaN in {v.name}: shape={v.shape}')
            if i > 5:
                print('  ...(more NaN grads)')
                break
print('Done!')
