import numpy as np
import tensorflow as tf


class CAE(tf.keras.Model):
    def __init__(self, num_alleles, num_snps, num_reads,
                 inner_act='relu', final_act='sigmoid',
                 num_expected_alleles=None, via_counter=False):
        super(CAE, self).__init__()
        self.num_alleles = num_alleles
        self.num_snps = num_snps
        self.num_reads = num_reads
        self.num_expected_alleles = num_expected_alleles

        latent_size = num_alleles
        if num_expected_alleles:
            latent_size = ((num_expected_alleles + 1) * num_alleles) if via_counter else (num_expected_alleles * num_snps) 
        
        # Encoder
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=inner_act, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=inner_act, padding='same')
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(128, activation=inner_act)
        self.dense2 = tf.keras.layers.Dense(latent_size, activation=final_act)
        
        # Decoder
        self.decode_dense1 = tf.keras.layers.Dense(128, activation=inner_act)
        self.decode_dense2 = tf.keras.layers.Dense(num_reads * num_snps * 4, activation=inner_act)
        self.decode_reshape = tf.keras.layers.Reshape((num_reads, num_snps, 4))
        self.decode_conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=inner_act, padding='same')
        self.decode_conv2 = tf.keras.layers.Conv2D(1, (3, 3), activation=final_act, padding='same')
        
        self.allele_array = None
        self.minors_weights = None
        self.counter_matrix = None
        
    def encode(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = self.dense1(x)
        return self.dense2(x)
    
    def decode(self, encoded):
        x = self.decode_dense1(encoded)
        x = self.decode_dense2(x)
        x = self.decode_reshape(x)
        x = self.decode_conv1(x)
        return self.decode_conv2(x)
    
    def call(self, inputs):
        allele_probs = self.encode(inputs)
        reconstructed = self.decode(allele_probs)
        return allele_probs, reconstructed

    def set_non_trainables(self, allele_array, minors_weights):
        self.allele_array = tf.constant(allele_array, dtype=tf.float32)
        self.minors_weights = tf.constant(minors_weights, dtype=tf.float32)

        if self.num_expected_alleles:
            counter_matrix = np.ones((self.num_expected_alleles, self.num_alleles)) + np.expand_dims(np.arange(self.num_expected_alleles), axis=1)
            self.counter_matrix = tf.constant(counter_matrix, dtype=tf.float32)
