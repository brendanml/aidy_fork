#%%
import random, os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import argparse

from etl import ETL
from model import Model
from util import dotdict

is_jupyter = "JPY_PARENT_PID" in os.environ

#%%
args = dotdict({
    "gene": 'cyp2c19',
    "seed": 42,
    "verbose": True,
    "runs": 3,
    "number_of_alleles": 1,
    "coverage": 10,
    "squeezed": True,
    "model": 'cae_v2',
    "epochs": 500,
    "loss": 'aidy_v2',
    "no_cache": True
})

if not is_jupyter:
    parser = argparse.ArgumentParser(
        prog='Aidy runtime')
    
    parser.add_argument('gene')
    parser.add_argument('-s', '--seed',
                        type=int, default=args.seed)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=args.verbose)
    parser.add_argument('-r', '--runs',
                        type=int, default=args.runs)
    parser.add_argument('-n', '--number-of-alleles',
                        type=int, default=args.number_of_alleles)
    parser.add_argument('-c', '--coverage',
                        type=int, default=args.coverage)
    parser.add_argument('-q', '--squeezed',
                        action='store_true',
                        default=args.squeezed)
    parser.add_argument('-m', '--model',
                        choices=['cae_v1', 'cae_v2'],
                        default=args.model)
    parser.add_argument('-e', '--epochs',
                        type=int, default=args.epochs)
    parser.add_argument('-l', '--loss',
                        default=args.loss)
    parser.add_argument('--no-cache',
                        action='store_true',
                        default=False)
    
    
    args = parser.parse_args()

#%%
if args.seed:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

etl = ETL(args.gene, args.coverage, args.no_cache)
runs = args.runs
population_size = args.number_of_alleles
model = Model(args.model, etl.squeezed_allele_db, args.coverage,
              etl.count_reads([random.choice(etl.allele_keys) for _ in range(population_size)]))
#%%
def load_test_bank(file_path):
    allele_names = list(a.name for a in etl.gene.alleles.values())
    
    allele_to_index = {name: index for index, name in enumerate(allele_names)}
    
    data = []
    labels = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            simulation = parts[0]
            alleles = parts[1:]
            
            # Create allele vector
            allele_vector = np.zeros(len(allele_names))
            for allele in alleles:
                allele = allele.split('_')[1]
                if allele in allele_to_index:
                    allele_vector[allele_to_index[allele]] = 1
                else:
                    assert False
            
            data.append(allele_vector)
            labels.append(alleles)
    
    return np.array(data), labels, allele_to_index


def prepare_data(data):
    return data.astype(np.float32)

def split_data(data, train_size=0.7, val_size=0.15, test_size=0.15):
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    train_data, val_data = train_test_split(train_val_data, test_size=val_size/(train_size+val_size), random_state=42)
    return train_data, val_data, test_data

def train_cae(model, train_data, val_data, epochs=100, batch_size=32):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        train_data, train_data, 
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(val_data, val_data),
        callbacks=[early_stopping]
    )
    
    return history

def evaluate_cae(model, test_data):
    test_loss = model.evaluate(test_data, test_data)
    print(f"Test Loss: {test_loss}")
    return test_loss

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

#%%
errors = []
for _ in range(runs):
    selected_alleles = [random.choice(etl.allele_keys) for _ in range(population_size)]
    reads = etl.sample(selected_alleles, squeezed=args.squeezed)

    expected_alleles = etl.filter_alleles(selected_alleles, squeezed=args.squeezed)
    errors.append(model.predict(
        reads, expected_alleles, epochs=args.epochs, loss_name=args.loss, verbose=args.verbose))

if args.verbose:
    print("Errors:", errors)

# %%
