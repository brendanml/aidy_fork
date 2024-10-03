import tensorflow as tf


def kl_divergence_loss(y_true, y_pred):
    y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1)
    return tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=-1)


def binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return tf.reduce_mean(-(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)))


def aidy_loss_v1(model, inputs, allele_probs, reconstructed):
    reconstruction_loss = binary_crossentropy(inputs, reconstructed)
    observed_snp_freq = tf.reduce_mean(inputs, axis=[1, 3])  # Average across reads and channel
    predicted_snp_freq = tf.matmul(allele_probs, model.allele_array)
    allele_loss = binary_crossentropy(observed_snp_freq, predicted_snp_freq)
    l1_reg = 0.01 * tf.reduce_sum(tf.abs(allele_probs))
    total_loss = reconstruction_loss + allele_loss + l1_reg
    return total_loss, reconstruction_loss, allele_loss, l1_reg
