import tensorflow as tf


def kl_divergence_loss(y_true, y_pred):
    y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1)
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1)
    return tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=-1)


def binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return tf.reduce_mean(-(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)))


def mean_absolute_error(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def aidy_loss_v1(inputs, allele_probs, reconstructed, allele_db, *args):
    reconstruction_loss = binary_crossentropy(inputs, reconstructed)
    observed_snp_freq = tf.reduce_mean(inputs, axis=[1, 3])  # Average across reads and channel
    predicted_snp_freq = tf.matmul(allele_probs, allele_db)
    allele_loss = binary_crossentropy(observed_snp_freq, predicted_snp_freq)
    l1_reg = 0.01 * tf.reduce_sum(tf.abs(allele_probs))
    total_loss = reconstruction_loss + allele_loss + l1_reg
    return total_loss, reconstruction_loss, allele_loss, l1_reg


def aidy_loss_v2(inputs, allele_probs, reconstructed, allele_db, exp_cov, *args):
    observed_snp_counts = tf.reduce_sum(inputs, axis=[1, 3])  # Sum across reads and channel
    reconstructed_snp_counts = tf.reduce_sum(reconstructed, axis=[1, 3])
    reconstruction_loss = tf.reduce_mean(tf.square(observed_snp_counts - reconstructed_snp_counts))
    toy_allele_database_cov = tf.constant(allele_db * exp_cov, dtype=tf.float32)
    expected_snp_counts = tf.matmul(allele_probs, toy_allele_database_cov)
    allele_loss = tf.reduce_mean(tf.square(expected_snp_counts - observed_snp_counts))
    l1_reg = 0.01 * tf.reduce_sum(tf.abs(allele_probs))
    total_loss = reconstruction_loss + allele_loss + l1_reg
    return total_loss, reconstruction_loss, allele_loss, l1_reg


def aidy_loss_v3(reads, allele_probs, reconstructed, allele_db, coverage, *args):
    # For major allele calling. No copy number.
    reconstruction_loss = binary_crossentropy(reads, reconstructed)
    observed_snp_freq = tf.divide(tf.reduce_sum(reads, axis=[1, 3]), coverage)
    predicted_snp_freq = tf.matmul(allele_probs, allele_db)
    allele_loss = mean_absolute_error(observed_snp_freq, predicted_snp_freq)
    l1_reg = 0.01 * tf.reduce_sum(tf.abs(allele_probs))
    total_loss = 0.1 * reconstruction_loss + allele_loss + l1_reg
    return total_loss, reconstruction_loss, allele_loss, l1_reg


def aidy_loss_v4(reads, allele_probs, reconstructed, allele_db, coverage, minors_weights, *args):
    # For both major and minor allele calling. No copy number.
    reconstruction_loss = binary_crossentropy(reads, reconstructed)
    observed_snp_freq = tf.divide(tf.reduce_sum(reads, axis=[1, 3]), coverage)
    predicted_snp_freq = tf.matmul(allele_probs, allele_db)
    weighted_observed_snp_freq = tf.multiply(observed_snp_freq, minors_weights)
    weighted_predicted_snp_freq = tf.multiply(predicted_snp_freq, minors_weights)
    allele_loss = mean_absolute_error(weighted_observed_snp_freq, weighted_predicted_snp_freq)
    l1_reg = 0.01 * tf.reduce_sum(tf.abs(allele_probs))
    total_loss = 0.1 * reconstruction_loss + allele_loss + l1_reg
    return total_loss, reconstruction_loss, allele_loss, l1_reg


def aidy_loss_v5(reads, allele_probs, reconstructed, allele_db, coverage, minors_weights, num_expected_alleles, *args):
    # For both major and minor allele calling. With copy number and 2-dim latent variable.
    reconstruction_loss = binary_crossentropy(reads, reconstructed)
    observed_snp_freq = tf.divide(tf.reduce_sum(reads, axis=[1, 3]), coverage)
    predicted_snp_freq = tf.reduce_sum(tf.reshape(allele_probs, (num_expected_alleles, allele_db.shape[1])), axis=0)
    weighted_observed_snp_freq = tf.multiply(observed_snp_freq, minors_weights)
    weighted_predicted_snp_freq = tf.multiply(predicted_snp_freq, minors_weights)
    allele_loss = mean_absolute_error(weighted_observed_snp_freq, weighted_predicted_snp_freq)
    l1_reg = 0.01 * tf.reduce_sum(tf.abs(allele_probs))
    total_loss = 0.1 * reconstruction_loss + allele_loss + l1_reg
    return total_loss, reconstruction_loss, allele_loss, l1_reg


def aidy_loss_v6(reads, allele_vec, reconstructed, allele_db, coverage, minors_weights, num_expected_alleles, counter_matrix):
    # TODO: Not done yet. Reduces all calls to 0 for some reason.
    # For both major and minor allele calling. With a one-hot encoded copy number counter.
    reconstruction_loss = binary_crossentropy(reads, reconstructed)
    observed_snp_freq = tf.divide(tf.reduce_sum(reads, axis=[1, 3]), coverage)
    allele_mat = tf.reshape(allele_vec, (num_expected_alleles + 1, allele_db.shape[0]))
    allele_probs = allele_mat[0]
    allele_counts = tf.reduce_sum(tf.multiply(allele_mat[1:], counter_matrix), axis=0)
    predicted_snp_freq = tf.matmul(tf.expand_dims(tf.multiply(allele_probs, allele_counts), axis=0), allele_db)
    weighted_observed_snp_freq = tf.multiply(observed_snp_freq, minors_weights)
    weighted_predicted_snp_freq = tf.multiply(predicted_snp_freq, minors_weights)
    allele_loss = mean_absolute_error(weighted_observed_snp_freq, weighted_predicted_snp_freq)
    l1_reg = 0.01 * tf.reduce_sum(tf.abs(allele_vec))
    total_loss = 0.1 * reconstruction_loss + allele_loss + l1_reg
    return total_loss, reconstruction_loss, allele_loss, l1_reg


def get_loss(loss_name, *args):
    if loss_name == "aidy_v1":
        return aidy_loss_v1(*args)
    if loss_name == "aidy_v2":
        return aidy_loss_v2(*args)
    if loss_name == "aidy_v3":
        return aidy_loss_v3(*args)
    if loss_name == "aidy_v4":
        return aidy_loss_v4(*args)
    if loss_name == "aidy_v5":
        return aidy_loss_v5(*args)
    raise ValueError("Invalid loss name:", loss_name)
