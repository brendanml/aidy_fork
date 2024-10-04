import random
from sklearn.model_selection import train_test_split
from etl import ETL
import argparse

parser = argparse.ArgumentParser(
        prog='Aidy simulation')
parser.add_argument('gene')
args = parser.parse_args()
def generate_simulation(alleles, copy_number):
    return random.sample(alleles, copy_number)

def write_simulations(file_path, simulations):
    with open(file_path, 'a') as file:
        for i, sim in enumerate(simulations):
            file.write(f"simulation_{i},{','.join(sim)}\n")

def verify_no_duplication(simulations):
    for i, sim in enumerate(simulations):
        if len(sim) != len(set(sim)):
            print(f"Error: Duplication found in simulation {i}")
            return False
    print("Verification complete: No duplications found in any simulation")
    return True

def split_simulations(simulations):
    train, temp = train_test_split(simulations, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    return train, val, test

# Read allele names
etl = ETL(args.gene, 10, False)
allele_names = list(a.name for a in etl.gene.alleles.values())

# Generate simulations
simulations = {
    1: [[allele] for allele in allele_names],  # 1 copy simulations (all alleles)
    2: [generate_simulation(allele_names, 2) for _ in range(800)],
    3: [generate_simulation(allele_names, 3) for _ in range(800)],
    4: [generate_simulation(allele_names, 4) for _ in range(800)]
}

# Verify no duplications
if all(verify_no_duplication(sims) for sims in simulations.values()):
    # Write 1-copy simulations to train.txt
    write_simulations("artifacts/train.txt", simulations[1])
    
    # Split and write 2, 3, 4 copy simulations
    for copy_num in [2, 3, 4]:
        train, val, test = split_simulations(simulations[copy_num])
        write_simulations("artifacts/train.txt", train)
        write_simulations("artifacts/validation.txt", val)
        write_simulations("artifacts/test.txt", test)
    
    print(f"Simulations generated and written to files:")
    print(f"- {len(simulations[1])} 1-copy simulations (all in train.txt)")
    print(f"- 800 2-copy simulations (split across train.txt, validation.txt, test.txt)")
    print(f"- 800 3-copy simulations (split across train.txt, validation.txt, test.txt)")
    print(f"- 800 4-copy simulations (split across train.txt, validation.txt, test.txt)")
else:
    print("Error: Duplications found. Simulations not written to files.")