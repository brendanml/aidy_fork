#!/bin/bash

output_file="artifacts/cyp2c19_allele_names.txt"

> "$output_file"


find temp/sim -name "*.fa" | while read -r file; do
    basename "$file" .fa >> "$output_file"
done

echo "Allele names have been written to $output_file"

