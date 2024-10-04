import numpy as np
import tensorflow as tf

from models.cae import CAE
from models.loss import get_loss


class Model:
    def __init__(self, model_name, allele_db, coverage, num_reads):
        self.coverage = coverage
        self.allele_db = allele_db
        
        if model_name.startswith('cae'):
            if model_name == 'cae_v1':
                self.model = CAE(*allele_db.shape)
            elif model_name == 'cae_v2':
                self.model = CAE(*allele_db.shape, num_reads)
            elif model_name == 'cae_v3':
                self.model = CAE2(*allele_db.shape, num_reads)
            else:
                raise ValueError("Invalid model name", model_name)
            self.model.set_allele_array(allele_db)
        else:
            raise ValueError("Invalid model name", model_name) 

    def loss(self, loss_name, *args):
        return get_loss(loss_name, *args)
    
    def supervised_run(self, train_reads, train_labels, val_reads=None, val_labels=None, epochs=500, verbose=False):
        """
        Trains the model in a supervised manner using the provided training data and labels.
        Optionally uses validation data if provided.

        Args:
            train_reads (numpy.ndarray): Training input data.
            train_labels (numpy.ndarray): Training labels (allele vectors).
            val_reads (numpy.ndarray, optional): Validation input data.
            val_labels (numpy.ndarray, optional): Validation labels.
            epochs (int, optional): Number of training epochs.
            verbose (bool, optional): Verbosity mode.

        Returns:
            history: Keras History object containing training metrics.
        """
        # Preprocess inputs
        train_inputs = train_reads.astype(np.float32)
        train_labels = train_labels.astype(np.float32)

        # Preprocess validation data if provided
        if val_reads is not None and val_labels is not None:
            val_inputs = val_reads.astype(np.float32)
            val_labels = val_labels.astype(np.float32)
            validation_data = (
                val_inputs,
                {
                    'reconstructed': val_inputs,
                    'allele_probs': val_labels
                }
            )
        else:
            validation_data = None

        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss={
                'reconstructed': 'mse',
                'allele_probs': 'binary_crossentropy'
            },
            loss_weights={
                'reconstructed': 1.0,
                'allele_probs': 1.0
            },
            metrics={
                'allele_probs': 'accuracy'
            }
        )

        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )

        # Fit the model
        history = self.model.fit(
            x=train_inputs,
            y={
                'reconstructed': train_inputs,
                'allele_probs': train_labels
            },
            epochs=epochs,
            batch_size=32,
            shuffle=True,
            verbose=verbose,
            validation_data=validation_data,
            callbacks=[early_stopping]
        )

        return history
    
    def test(self, test_reads, test_labels, batch_size=32, verbose=False):
        """
        Evaluates the trained model on test data.

        Args:
            test_reads (numpy.ndarray): Test input data.
            test_labels (numpy.ndarray): Test labels (allele vectors).
            batch_size (int, optional): Batch size for evaluation.
            verbose (bool, optional): Verbosity mode.

        Returns:
            evaluation: A dictionary containing loss and metrics on test data.
            predictions: The model's predictions on test data.
        """
        # Preprocess inputs
        test_inputs = test_reads.astype(np.float32)
        test_labels = test_labels.astype(np.float32)

        # Evaluate the model
        evaluation = self.model.evaluate(
            x=test_inputs,
            y={
                'reconstructed': test_inputs,
                'allele_probs': test_labels
            },
            batch_size=batch_size,
            verbose=verbose
        )

        # Get predictions
        predictions = self.model.predict(
            x=test_inputs,
            batch_size=batch_size,
            verbose=verbose
        )

        # Threshold allele probabilities to get binary predictions
        predicted_allele_probs = predictions['allele_probs']
        predicted_alleles = (predicted_allele_probs > 0.5).astype(int)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        true_labels = test_labels.flatten()
        pred_labels = predicted_alleles.flatten()

        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

        if verbose:
            print(f"Test Loss: {evaluation[0]:.4f}")
            print(f"Reconstruction Loss: {evaluation[1]:.4f}")
            print(f"Allele Prediction Loss: {evaluation[2]:.4f}")
            print(f"Allele Prediction Accuracy: {evaluation[3]:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"Test F1 Score: {f1:.4f}")

        # Return evaluation metrics and predictions
        return {
            'evaluation': evaluation,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }, predictions
        
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
                loss, rec_loss, allele_loss, l1_reg = self.loss(
                    loss_name, input_reads, allele_probs,
                    reconstructed, self.allele_db, self.coverage)
            
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
        exp_all_sum = np.sum(expected_alleles, axis=0)
        error = (inf_all_sum != exp_all_sum).astype(int).sum()
        
        if verbose:
            print("Inferred Allele indices (sum):", inf_all_sum)
            print("Expected Alleles indices (sum):", exp_all_sum)
            print("Error:", error)

        # Calculate accuracy metrics
        return error
    
    
    def is_unsupervised(self):
        if isinstance(self.model, CAE):
            return True
        if isinstance(self.model, CAE2):
            return False
        return False
    
    def predict(self, *args, **kwargs):
        if self.is_unsupervised():
            return self.unsupervised_run(*args, **kwargs)
        else:
            return self.supervised_run(train_reads, train_labels, val_reads, val_labels, *args, **kwargs)
