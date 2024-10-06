import random

import numpy as np
import tensorflow as tf

from models.cae import CAE
from models.linear import Linear
from loss import get_loss


class Module:
    def __init__(self, model_name, etl, squeezed, inner_act, final_act,
                 epochs, loss_name, minor_allele_weight, verbose):
        self.etl = etl
        self.coverage = etl.coverage
        self.squeezed = squeezed
        self.allele_db = etl.squeezed_allele_db if squeezed else etl.allele_db
        self.minors_mask = etl.squeezed_minors_mask if squeezed else etl.minors_mask
        self.minors_weights = 1 - self.minors_mask.astype(float) * (1 - minor_allele_weight)
        self.model_name = model_name
        self.inner_act = inner_act
        self.final_act = final_act
        self.epochs = epochs
        self.loss_name = loss_name
        self.verbose = verbose

        self.reset_model()
    
    def reset_model(self, num_reads=0):
        if self.model_name.startswith('cae'):
            if self.model_name == 'cae_v1':
                self.model = CAE(*self.allele_db.shape,
                                 inner_act=self.inner_act, final_act=self.final_act)
            elif self.model_name == 'cae_v2':
                self.model = CAE(*self.allele_db.shape, num_reads,
                                 inner_act=self.inner_act, final_act=self.final_act)
            else:
                raise ValueError("Invalid model name", self.model_name)
            self.model.set_non_trainables(self.allele_db, self.minors_weights)
        elif self.model_name == 'linear':
            self.model = Linear(*self.allele_db.shape, num_reads, self.inner_act, self.final_act)
            self.model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
            
            self.model.fit(*self.etl.sample_feature_labels(self.squeezed), epochs=500)
        else:
            raise ValueError("Invalid model name", self.model_name)

    def loss(self, loss_name, *args):
        return get_loss(loss_name, *args)
    
    def unsupervised_run(self, reads, expected_alleles):
        # Convert reads to model input format (add batch and channel dimensions)
        input_reads = reads.reshape(1, -1, self.allele_db.shape[1], 1).astype(np.float32)

        # Initialize model (build)
        self.reset_model(len(reads))
        self.model(input_reads)

        # Training loop
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        clip_value = 1.0  # Gradient clipping threshold

        for epoch in range(self.epochs):
            with tf.GradientTape() as tape:
                allele_probs, reconstructed = self.model(input_reads)
                loss, rec_loss, allele_loss, l1_reg = self.loss(
                    self.loss_name, input_reads, allele_probs,
                    reconstructed, self.allele_db, self.coverage, self.minors_weights)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            # Gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
            
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            if self.verbose:
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Total Loss: {loss.numpy():.4f}, "
                        f"Rec Loss: {rec_loss.numpy():.4f}, Allele Loss: {allele_loss.numpy():.4f}, "
                        f"L1 Reg: {l1_reg.numpy():.4f}")

        # Final evaluation
        allele_probs, reconstructed = self.model(input_reads)
        probs = allele_probs[0].numpy()

        infered_alleles = self.allele_db[
            np.array(sorted(
                zip(probs, range(len(probs))),
                    reverse=True)[:len(expected_alleles)]).T[1].astype(int)]

        inf_all_sum = np.sum(infered_alleles, axis=0)
        exp_all_sum = np.sum(expected_alleles, axis=0)
        error = (inf_all_sum != exp_all_sum).astype(int).sum()
        
        if self.verbose:
            np.set_printoptions(suppress=True)
            print("Inferred all alleles prob:", probs)
            print("Inferred all alleles sum:", inf_all_sum)
            print("Expected all alleles sum:", exp_all_sum)
            print("Error all alleles:", error)

            majors_mask = (1 - self.minors_mask).astype(bool)
            inf_major_sum = inf_all_sum[majors_mask]
            exp_major_sum = exp_all_sum[majors_mask]
            print("Inferred major alleles sum:", inf_major_sum)
            print("Expected major alleles sum:", exp_major_sum)
            print("Error major alleles:", (inf_major_sum != exp_major_sum).astype(int).sum())

        # Calculate accuracy metrics
        return error
    
    def is_unsupervised(self):
        if isinstance(self.model, CAE):
            return True
        return False
    
    def evaluate(self, reads, expected_alleles):
        if self.is_unsupervised():
            return self.unsupervised_run(reads, expected_alleles)
