import random

import numpy as np
import tensorflow as tf
import argparse

from etl import ETL
from model import Model


parser = argparse.ArgumentParser(
    prog='Aidy runtime')

parser.add_argument('gene')
parser.add_argument('-s', '--seed', type=int)
parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=True)
parser.add_argument('-r', '--runs', type=int, default=3)
parser.add_argument('-n', '--number-of-alleles', type=int, default=3)
parser.add_argument('-c', '--coverage', type=int, default=10)
parser.add_argument('-q', '--squeezed',
                    action='store_true',
                    default=True)
parser.add_argument('-m', '--model',
                    choices=['cae_v1', 'cae_v2'],
                    default='cae_v1')
parser.add_argument('-e', '--epochs', type=int, default=500)
parser.add_argument('-l', '--loss', default='aidy_v2')

args = parser.parse_args()

if args.seed:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

etl = ETL(args.gene, args.coverage)
runs = args.runs
population_size = args.number_of_alleles
model = Model(args.model, etl.squeezed_allele_db, args.coverage,
              etl.count_reads([random.choice(etl.allele_keys) for _ in range(population_size)]))

errors = []
for _ in range(runs):
    selected_alleles = [random.choice(etl.allele_keys) for _ in range(population_size)]
    reads = etl.sample(selected_alleles, squeezed=args.squeezed)

    expected_alleles = etl.filter_alleles(selected_alleles, squeezed=args.squeezed)
    errors.append(model.predict(
        reads, expected_alleles, epochs=args.epochs, loss_name=args.loss, verbose=args.verbose))

if args.verbose:
    print("Errors:", errors)
