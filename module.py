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

        self.counter_matrix = None
        self.reset_model()
    
    def reset_model(self, num_reads=0, num_expected_alleles=None):
        if self.model_name.startswith('cae'):
            if self.model_name == 'cae_allele_probs':
                self.model = CAE(*self.allele_db.shape, num_reads,
                                inner_act=self.inner_act, final_act=self.final_act)
            elif self.model_name == 'cae_allele_calls':
                self.model = CAE(*self.allele_db.shape, num_reads,
                                 inner_act=self.inner_act, final_act=self.final_act,
                                 num_expected_alleles=num_expected_alleles)
            elif self.model_name == 'cae_allele_counts':
                self.model = CAE(*self.allele_db.shape, num_reads,
                                 inner_act=self.inner_act, final_act=self.final_act,
                                 num_expected_alleles=num_expected_alleles, via_counter=True)
            self.model.set_non_trainables(self.allele_db, self.minors_weights)
            if self.model.counter_matrix is not None:
                self.counter_matrix = self.model.counter_matrix.numpy()
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
        self.reset_model(len(reads), len(expected_alleles))
        self.model(input_reads)

        # Training loop
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        clip_value = 1.0  # Gradient clipping threshold

        for epoch in range(self.epochs):
            with tf.GradientTape() as tape:
                allele_probs, reconstructed = self.model(input_reads)
                loss, rec_loss, allele_loss, l1_reg = self.loss(
                    self.loss_name, input_reads, allele_probs,
                    reconstructed, self.allele_db, self.coverage,
                    self.minors_weights, len(expected_alleles),
                    self.counter_matrix)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            # Gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
            
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            if self.verbose:
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Total Loss: {loss.numpy():.4f}, "
                        f"Rec Loss: {rec_loss.numpy():.4f}, Allele Loss: {allele_loss.numpy():.4f}, "
                        f"L1 Reg: {(l1_reg.numpy() if l1_reg is not None else 0.0):.4f}")

        # Final evaluation
        if self.verbose:
            np.set_printoptions(suppress=True)

        if self.model_name == "cae_allele_counts":
            allele_matrix = self.model(input_reads)[0][0].numpy().reshape((
                len(expected_alleles) + 1, self.allele_db.shape[0]))
            allele_probs = allele_matrix[0]
            allele_counts = np.sum(allele_matrix[1:] * self.counter_matrix, axis=0)
            infered_alleles = np.expand_dims(allele_probs * allele_counts, axis=0) @ self.allele_db
        elif self.model_name == "cae_allele_calls":
            infered_alleles = np.round(self.model(input_reads)[0][0].numpy().reshape((
                len(expected_alleles), self.allele_db.shape[1])))
        elif self.model_name == "cae_allele_probs":
            probs = self.model(input_reads)[0][0].numpy()
            if self.verbose:
                print("Inferred all alleles probs:", probs)

            infered_alleles = []
            temp_probs = probs.copy()
            while len(infered_alleles) < len(expected_alleles):
                cp_num = round(temp_probs.max())
                infered_alleles.append(self.allele_db[temp_probs.argmax()].tolist())
                if cp_num > 1:
                    for _ in range(cp_num - 1):
                        infered_alleles.append(self.allele_db[temp_probs.argmax()].tolist())
                temp_probs[temp_probs.argmax()] = 0.0
            
            infered_alleles = np.array(infered_alleles)
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

        inf_all_sum = np.sum(infered_alleles, axis=0)
        exp_all_sum = np.sum(expected_alleles, axis=0)
        error_all = (inf_all_sum != exp_all_sum).astype(int).sum()

        majors_mask = (1 - self.minors_mask).astype(bool)
        inf_major_sum = inf_all_sum[majors_mask]
        exp_major_sum = exp_all_sum[majors_mask]
        error_major = (inf_major_sum != exp_major_sum).astype(int).sum()
        
        if self.verbose:
            print("\nInferred all alleles sum:", inf_all_sum)
            print("Expected all alleles sum:", exp_all_sum)
            print("Error all alleles:", error_all)

            print("\nInferred major alleles sum:", inf_major_sum)
            print("Expected major alleles sum:", exp_major_sum)
            print("Error major alleles:", error_major)

        # Calculate accuracy metrics
        return error_all, error_major 
    
    def accuracy_from_errors(self, errors):
        num_snps = len(self.minors_mask)
        num_major_snps = num_snps - self.minors_mask.sum()

        correct_calls = sum([(num_snps - err[0]) for err in errors])
        correct_major_calls = sum([(num_major_snps - err[1]) for err in errors])

        acc = correct_calls / (num_snps * len(errors))
        major_acc = correct_major_calls / (num_major_snps * len(errors))

        return acc, major_acc

    def is_unsupervised(self):
        if isinstance(self.model, CAE):
            return True
        return False
    
    def evaluate(self, reads, expected_alleles, filter_db=False):
        if filter_db:
            filtered_db = self.allele_db[
                np.any(
                    np.logical_and(
                        np.logical_xor(
                            self.allele_db.astype(bool),
                            np.sum(reads, axis=0) > 0),
                        self.allele_db.astype(bool)),
                    axis=1)]
            self.model.set_non_trainables(filtered_db, self.minors_weights)
        
        if self.is_unsupervised():
            return self.unsupervised_run(reads, expected_alleles)
