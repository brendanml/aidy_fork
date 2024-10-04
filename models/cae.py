import tensorflow as tf


class CAE(tf.keras.Model):
    def __init__(self, num_alleles, num_snps, num_reads = None):
        super(CAE, self).__init__()
        self.num_alleles = num_alleles
        self.num_snps = num_snps
        
        # Encoder
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_alleles, activation='sigmoid')
        
        # Decoder
        self.decode_dense1 = tf.keras.layers.Dense(128, activation='relu')

        if num_reads:
            self.decode_dense2 = tf.keras.layers.Dense(num_reads * num_snps * 4, activation='relu')
            self.decode_reshape = tf.keras.layers.Reshape((num_reads, num_snps, 4))
            self.decode_conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
            self.decode_conv2 = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        else:
            self.decode_dense2 = tf.keras.layers.Dense(num_snps * 64, activation='relu')
            self.decode_reshape = tf.keras.layers.Reshape((1, num_snps, 64))
            self.decode_conv1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')
            self.decode_conv2 = tf.keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')
        
        self.allele_array = None
        
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

    def set_allele_array(self, allele_array):
        self.allele_array = tf.constant(allele_array, dtype=tf.float32)



class CAE2(tf.keras.Model):
    '''
        CAE with a prediciton layer, and the latent space is larger(not the number of alleles)
    '''
    def __init__(self, num_alleles, num_snps, num_reads, latent_dim=256):
        super(CAE2, self).__init__()
        self.num_alleles = num_alleles
        self.num_snps = num_snps
        self.num_reads = num_reads
        self.latent_dim = latent_dim
        
        # Encoder
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(latent_dim, activation='relu')
        
        # Allele Prediction Layer
        self.allele_predictor = tf.keras.layers.Dense(num_alleles, activation='sigmoid')
        
        # Decoder
        self.decode_dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.decode_dense2 = tf.keras.layers.Dense(num_reads * num_snps * 4, activation='relu')
        self.decode_reshape = tf.keras.layers.Reshape((num_reads, num_snps, 4))
        self.decode_conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.decode_conv2 = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        
        self.allele_array = None
        
    def encode(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = self.dense1(x)
        return self.dense2(x)
    
    def predict_alleles(self, latent):
        return self.allele_predictor(latent)
    
    def decode(self, latent):
        x = self.decode_dense1(latent)
        x = self.decode_dense2(x)
        x = self.decode_reshape(x)
        x = self.decode_conv1(x)
        return self.decode_conv2(x)
    
    def call(self, inputs):
        latent = self.encode(inputs)
        allele_probs = self.predict_alleles(latent)
        reconstructed = self.decode(latent)
        return latent, allele_probs, reconstructed

    def set_allele_array(self, allele_array):
        self.allele_array = tf.constant(allele_array, dtype=tf.float32)