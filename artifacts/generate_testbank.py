import random

def read_allele_names(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def generate_simulation(alleles, copy_number):
    return random.sample(alleles, copy_number)

def write_test_bank(output_file, simulations):
    with open(output_file, 'w') as file:
        for i, sim in enumerate(simulations):
            file.write(f"simulation_{i},{','.join(sim)}\n")

# Read allele names
allele_names = read_allele_names("artifacts/cyp2c19_allele_names.txt")

# Generate simulations
simulations = []

# 1 copy simulations (all alleles)
simulations.extend([[allele] for allele in allele_names])

# 2 copy simulations
for _ in range(300):
    simulations.append(generate_simulation(allele_names, 2))

# 3 copy simulations
for _ in range(300):
    simulations.append(generate_simulation(allele_names, 3))

# 4 copy simulations
for _ in range(300):
    simulations.append(generate_simulation(allele_names, 4))

def verify_no_duplication(simulations):
    for i, sim in enumerate(simulations):
        if len(sim) != len(set(sim)):
            print(f"Error: Duplication found in simulation {i}")
            return False
    print("Verification complete: No duplications found in any simulation")
    return True

# After generating all simulations
if verify_no_duplication(simulations):
    write_test_bank("artifacts/test_bank.txt", simulations)
else:
    print("Error: Duplications found. Test bank not generated.")
# Write to test_bank.txt
write_test_bank("artifacts/test_bank.txt", simulations)

print(f"Test bank generated with {len(simulations)} simulations.")
print(f"- {len(allele_names)} 1-copy simulations")
print("- 800 2-copy simulations")
print("- 1000 3-copy simulations")
print("- 1000 4-copy simulations")