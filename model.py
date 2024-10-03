import numpy as np
import tensorflow as tf

from models.cae import CAE
from models.loss import aidy_loss_v1


class Model:
    def __init__(self, model_name, allele_db):
        if model_name == 'cae':
            self.allele_db = allele_db
            self.model = CAE(*allele_db.shape)
            self.model.set_allele_array(allele_db)
        else:
            raise ValueError("Invalid model name", model_name)

    def loss(self, loss_name, *args):
        if loss_name == "aidy_v1":
            return aidy_loss_v1(*args)
        raise ValueError("Invalid loss name:", loss_name)
    
    def unsupervised_run(self, reads, expected_alleles, epochs=500, loss_name='aidy_v1', verbose=False):
        # Convert reads to model input format (add batch and channel dimensions)
        input_reads = reads.reshape(1, -1, self.allele_db.shape[1], 1).astype(np.float32)

        # Initialize model (build)
        self.model(input_reads)

        # Training loop
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        clip_value = 1.0  # Gradient clipping threshold

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                allele_probs, reconstructed = self.model(input_reads)
                loss, rec_loss, allele_loss, l1_reg = self.loss(loss_name, self.model, input_reads, allele_probs, reconstructed)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            # Gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
            
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            if verbose:
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
        print("Inferred Allele indices (sum):", inf_all_sum)
        exp_all_sum = np.sum(expected_alleles, axis=0)
        print("Expected Alleles indices (sum):", exp_all_sum)

        error = (inf_all_sum != exp_all_sum).astype(int).sum()
        print("Error:", error)

        # Calculate accuracy metrics
        return error
    
    def is_unsupervised(self):
        if isinstance(self.model, CAE):
            return True
        return False
    
    def predict(self, *args, **kwargs):
        if self.is_unsupervised():
            return self.unsupervised_run(*args, **kwargs)
