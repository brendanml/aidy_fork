import tensorflow as tf


def kl_divergence_loss(y_true, y_pred):
    y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1)
    return tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=-1)


def binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return tf.reduce_mean(-(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)))


def aidy_loss_v1(inputs, allele_probs, reconstructed, allele_db, *args):
    reconstruction_loss = binary_crossentropy(inputs, reconstructed)
    observed_snp_freq = tf.reduce_mean(inputs, axis=[1, 3])  # Average across reads and channel
    predicted_snp_freq = tf.matmul(allele_probs, allele_db)
    allele_loss = binary_crossentropy(observed_snp_freq, predicted_snp_freq)
    l1_reg = 0.01 * tf.reduce_sum(tf.abs(allele_probs))
    total_loss = reconstruction_loss + allele_loss + l1_reg
    return total_loss, reconstruction_loss, allele_loss, l1_reg


def aidy_loss_v2(inputs, allele_probs, reconstructed, allele_db, exp_cov):
    observed_snp_counts = tf.reduce_sum(inputs, axis=[1, 3])  # Sum across reads and channel
    reconstructed_snp_counts = tf.reduce_sum(reconstructed, axis=[1, 3])
    reconstruction_loss = tf.reduce_mean(tf.square(observed_snp_counts - reconstructed_snp_counts))
    toy_allele_database_cov = tf.constant(allele_db * exp_cov, dtype=tf.float32)
    expected_snp_counts = tf.matmul(allele_probs, toy_allele_database_cov)
    # allele_loss = binary_crossentropy(observed_snp_freq, predicted_snp_freq)
    allele_loss = tf.reduce_mean(tf.square(expected_snp_counts - observed_snp_counts))
    l1_reg = 0.01 * tf.reduce_sum(tf.abs(allele_probs))
    # total_loss = reconstruction_loss + allele_loss + l1_reg
    total_loss = reconstruction_loss + allele_loss + l1_reg
    return total_loss, reconstruction_loss, allele_loss, l1_reg


def get_loss(loss_name, *args):
    if loss_name == "aidy_v1":
        return aidy_loss_v1(*args)
    if loss_name == "aidy_v2":
        return aidy_loss_v2(*args)
    raise ValueError("Invalid loss name:", loss_name)
