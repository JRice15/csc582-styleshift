import numpy as np
import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """
  custom schedule from Vaswani paper
  """

  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    return lr

  def get_config(self):
    return {
      "d_model": int(self.d_model.numpy()),
      "warmup_steps": int(self.warmup_steps),
    }


### Loss and metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')

@tf.function
def loss_function(real, pred):
  # Since the target sequences are padded, it is important to apply a padding mask 
  # when calculating the loss.
  loss_ = loss_object(real, pred)

  mask = tf.math.not_equal(real, 0)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

@tf.function
def accuracy_metric(real, pred):
  real = tf.cast(real, tf.int32)
  accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.int32))

  mask = tf.math.not_equal(real, 0)
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


