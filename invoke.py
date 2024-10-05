#%%
import random, os

import numpy as np
import tensorflow as tf
import argparse

from etl import ETL
from model import Model
from util import dotdict

is_jupyter = "JPY_PARENT_PID" in os.environ

#%%
args = dotdict({
    "gene": 'cyp2c19',
    "seed": 0,
    "verbose": True,
    "runs": 3,
    "number_of_alleles": 1,
    "coverage": 10,
    "squeezed": True,
    "model": 'cae_v1',
    "epochs": 500,
    "loss": 'aidy_v1',
    "no_cache": False
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
                        default=args.no_cache)

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
