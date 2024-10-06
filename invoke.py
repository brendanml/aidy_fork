#%%
import os

import numpy as np
import tensorflow as tf
import argparse

from etl import ETL
from module import Module
from util import dotdict

is_jupyter = "JPY_PARENT_PID" in os.environ

#%%
args = dotdict({
    "gene": 'cyp2c19',
    "seed": 0,
    "verbose": True,
    "runs": 10,
    "number_of_alleles": 3,
    "coverage": 30,
    "squeezed": True,
    "model": 'cae_v2',
    "epochs": 1001,
    "loss": 'aidy_v3',
    "no_cache": True,
    "prune_zeros_reads": True,
    "inner_act": 'relu',
    "final_act": 'sigmoid'
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
                        choices=['cae_v1', 'cae_v2', 'linear'],
                        default=args.model)
    parser.add_argument('-e', '--epochs',
                        type=int, default=args.epochs)
    parser.add_argument('-l', '--loss',
                        default=args.loss)
    parser.add_argument('--prune-zeros-reads',
                        action='store_true',
                        default=args.prune_zeros_reads)
    parser.add_argument('--no-cache',
                        action='store_true',
                        default=args.no_cache)
    parser.add_argument('--inner-act',
                        default=args.inner_act)
    parser.add_argument('--final-act',
                        default=args.final_act)

    args = parser.parse_args()

#%%
if args.seed:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

etl = ETL(args.gene, args.coverage, args.number_of_alleles, args.no_cache)
module = Module(
    args.model, etl, args.squeezed, args.inner_act,
    args.final_act, args.epochs, args.loss, args.verbose)

#%%
errors = []
for _ in range(args.runs):
    selected_alleles = etl.get_random_alleles()
    reads = etl.sample(selected_alleles, squeezed=args.squeezed, non_zeros_only=args.prune_zeros_reads)

    if args.verbose:
        print("Input reads shape:", reads.shape)

    expected_alleles = etl.filter_alleles(selected_alleles, squeezed=args.squeezed)
    errors.append(module.evaluate(reads, expected_alleles))

if args.verbose:
    print("Errors:", errors)

# %%
