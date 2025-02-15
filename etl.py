import os, random

import numpy as np
import pysam
import aldy.gene
import aldy.common

from itertools import combinations
from tqdm import tqdm


class ETL:
    CACHE_PATH = 'cache'
    
    def __init__(self, gene_name, coverage, allele_count, no_cache, include_minor_alleles):
        self.coverage = coverage
        self.allele_count = allele_count
        gene_path = aldy.common.script_path(f"aldy.resources.genes/{gene_name}.yml")
        self.gene = aldy.gene.Gene(gene_path, genome="hg38")
        os.system('mkdir -p temp/sim')
        if not os.path.exists("temp/chr10.fa.bwt"):
            os.system("wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr10.fa.gz")
            os.system("gunzip chr10.fa.gz && mv chr10.fa temp/")
            os.system(f"{os.getenv('BWA_PATH', 'bwa')} index temp/chr10.fa")
        
        # Allele dictionary: name -> [list_of_mutations]
        alleles = {}
        minors = {}
        for ma_n, ma in self.gene.alleles.items():
            alleles[ma_n] = list(ma.func_muts)
            minors[ma_n] = set()

            if include_minor_alleles:
                for mi_n, mi in ma.minors.items():
                    neutral_muts = list(mi.neutral_muts)
                    alleles[mi_n] = alleles[ma_n] + neutral_muts
                    minors[mi_n] = set([mut.pos for mut in neutral_muts])

        # Clean up duplicate alleles
        seen = set()
        ap = {}
        for a_n, a_muts in alleles.items():
            if frozenset(a_muts) in seen: continue
            seen.add(frozenset(a_muts))
            ap[a_n] = a_muts
        alleles = ap

        # List of all mutations of interest
        muts = set(sorted(j for i in alleles.values() for j in i))
        # Mutation index: maps a mutation to a vector index
        indices = {}
        for pos, op in muts:
            for i, o in self.aldy2one(pos, op):
                indices.setdefault((i, '_'), len(indices))
                indices.setdefault((i, o), len(indices))
        self.indices = indices
        
        # Maps each allele to vector that has 0/1 for each mutation
        allele_vectors = {}
        for an, a in alleles.items():
            v = [0] * len(indices)
            for m in muts:
                ok = m in a  # We need to handle _ (no mutation) specially
                for i, o in self.aldy2one(*m):
                    v[indices[i, o if ok else '_']] = 1
            allele_vectors[an] = np.array(v)
        self.allele_vectors = allele_vectors

        allele_minors_masks = {}
        for an, a in alleles.items():
            allele_minors_masks[an] = [0] * len(indices)
            for m in muts:
                ok = m in a  # We need to handle _ (no mutation) specially
                for i, o in self.aldy2one(*m):
                    idx = indices[i, o if ok else '_']
                    allele_minors_masks[an][idx] = int(i in minors[an])
                    allele_minors_masks[an][idx + (-1 if ok else 1)] = int(i in minors[an])
        self.allele_minors_masks = allele_minors_masks

        # Do simulations
        # Simulate allele sequences with ART and align them to the reference
        for a_n, a_muts in alleles.items():
            if os.path.exists(f"temp/sim/{self.gene.name}_{a_n}_1.fq") and os.path.exists(f"temp/sim/{self.gene.name}_{a_n}_2.fq") and not no_cache:
                continue
            
            # Write FA file
            with open(f'temp/sim/{self.gene.name}_{a_n}.fa', 'w') as f:
                seq, mt = self.reconstruct_seq(self.gene, a_muts)
                print(f'>{a_n}', file=f)
                print(seq, file=f)
            # Simulate
            os.system(
                "art_bin_MountRainier/art_illumina "
                f"-ss HS20 -sam -p -l 100 -f {self.coverage} -m 500 -s 10 -ef -sam "
                f"-o temp/sim/{self.gene.name}_{a_n}_ -i temp/sim/{self.gene.name}_{a_n}.fa"
            )
            # Align
            os.system(
                f"{os.getenv('BWA_PATH', 'bwa')} mem -t 4 temp/chr10.fa "
                f"temp/sim/{self.gene.name}_{a_n}_1.fq temp/sim/{self.gene.name}_{a_n}_2.fq | "
                f"{os.getenv('SAMTOOLS_PATH', 'samtools')} sort --write-index -o temp/sim/{self.gene.name}_{a_n}.bwa.bam##idx##temp/sim/{self.gene.name}_{a_n}.bwa.bai")

        # This is database matrix (rows: alleles, columns: variants)
        self.allele_keys = list(allele_vectors.keys())
        
        allele_db = []
        squeezed_allele_db = []
        minors_masks = []
        squeezed_minors_masks = []
        for k in self.allele_keys:
            allele_db.append(allele_vectors[k])
            squeezed_allele_db.append(self.squeeze(allele_vectors[k]))
            minors_masks.append(allele_minors_masks[k])
            squeezed_minors_masks.append(self.squeeze(allele_minors_masks[k]))
        
        self.allele_db = np.array(allele_db)
        self.squeezed_allele_db = np.array(squeezed_allele_db)
        self.minors_masks = np.array(minors_masks)
        self.squeezed_minors_masks = np.array(squeezed_minors_masks)
        self.minors_mask = np.logical_or.reduce(self.minors_masks, axis=0)
        self.squeezed_minors_mask = np.logical_or.reduce(self.squeezed_minors_masks, axis=0)
    
    def count_reads(self, alleles):
        reads = {}
        for allele in alleles:
            with pysam.AlignmentFile(f"temp/sim/{self.gene.name}_{allele}.bwa.bam") as sam:
                for record in sam.fetch(region=self.gene.get_wide_region().samtools(prefix="chr")):
                    if not record.cigarstring or "H" in record.cigarstring:  # only valid alignments
                        continue
                    if record.is_supplementary:  # avoid supplementary alignments
                        continue
                    if not record.query_sequence:
                        continue
                    fragment = record.query_name
                    reads.setdefault(fragment, {})
        
        return len(reads)
    
    def get_random_alleles(self, unique=False):
        assert len(set(self.allele_keys)) >= self.allele_count, f"Cannot select {self.allele_count} distinct alleles"
        
        if unique:
            allele_set = set()
            while len(allele_set) < self.allele_count:
                allele_set.add(random.choice(self.allele_keys))
        
            return list(allele_set)
        
        return [random.choice(self.allele_keys) for _ in range(self.allele_count)]
    
    def get_allele_vector(self, selected_alleles):
        return np.array([(1 if key in selected_alleles else 0) for key in self.allele_keys])
    
    def get_allele_matrix(self, selected_alleles, squeezed, sorted):
        allele_matrix = [self.allele_vectors[k] for k in selected_alleles]

        if squeezed:
            allele_matrix = [self.squeeze(row) for row in allele_matrix]
        
        if sorted:
            allele_matrix.sort(key=lambda x: str(x), reverse=True)
        
        return np.array(allele_matrix)
    
    def sample_feature_labels(self, squeezed, multiplicity=1, shuffle=True, non_zeros_only=True, no_cache=False):
        features_cache_path, labels_cache_path = self.get_cache_paths(squeezed, non_zeros_only)
        
        if os.path.exists(features_cache_path) and os.path.exists(features_cache_path) and not no_cache:
            return np.fromfile(features_cache_path), np.fromfile(labels_cache_path)
        
        features, labels = [], []
        for allele_tuple in tqdm(list(combinations(self.allele_keys, self.allele_count)), "Generating feature/labels"):
            allele_vector = self.get_allele_vector(allele_tuple).tolist()
            for _ in range(multiplicity):
                features.append(self.sample(allele_tuple, squeezed, non_zeros_only).flatten().tolist())
                labels.append(allele_vector)
        
        max_len = max([len(row) for row in features])

        # Pad features
        np_features = np.zeros((len(features), max_len))
        for i in range(len(features)):
            for j in range(len(features[i])):
                np_features[i, j] = features[i][j]
        features = np_features.tolist()
        
        data = list(zip(features, labels))
        if shuffle:
            random.shuffle(data)
        
        features, labels = np.array([pair[0] for pair in data]), np.array([pair[1] for pair in data])

        np.savetxt(self.FEATURES_CACHE_PATH, features)
        np.savetxt(self.LABELS_CACHE_PATH, labels)
        
        return features, labels
    
    def sample(self, selected_alleles, squeezed, non_zeros_only=True):
        reads = []
        for allele in selected_alleles:
            allele_reads = self.get_sample(f"temp/sim/{self.gene.name}_{allele}.bwa.bam", self.gene, self.indices, squeezed).tolist()
            
            if not non_zeros_only:
                reads.extend(allele_reads.tolist())
                continue

            reads.extend([row for row in allele_reads if (np.array(row) != 0).any()])
        
        return np.array(reads)

    def filter_alleles(self, alleles, squeezed):
        filtered_alleles = [self.allele_vectors[allele].tolist() for allele in alleles]
        
        if squeezed:
            return np.array([self.squeeze(allele) for allele in filtered_alleles])
        return filtered_alleles

    # Convert Aldy mutation notation (delXXX, insXXX, X>Y) into point mutation
    # (ACGT, -, +ACGT)
    @staticmethod
    def aldy2one(pos, op):
        if op.startswith('ins'): return {(pos, "+" + op[3:])}
        elif op.startswith('del'): return {(pos + i, "-") for i in range(len(op) - 3)}
        elif len(op) == 3: return {(pos, op[2])}
        assert(f'bad {op}')
        return {}

    # This function reconstructs the allele sequence from the database.
    # Returns (sequence, list of changed mutations)
    # You can simulate from this sequence
    @staticmethod
    def reconstruct_seq(gene, muts):
        st, ed = gene._lookup_range
        seq = list(gene[st:ed])
        for pos, op in muts:
            if '>' in op: seq[pos - st] = op[2] # SNP
            if op.startswith('ins'): seq[pos - st] = op[3:] + seq[pos - st] # SNP
            if op.startswith('del'):
                for i in range(len(op) - 3):
                    seq[pos - st + i] = '' # SNP
        return (''.join(seq), [(pos - st, op) for pos, op in muts])
    
    # Load vectors from the SAM file
    @staticmethod
    def get_sample(file, gene, indices, squeezed):
        "Return a matrix: columns are mutations, rows are reads (fragments)"

        reads = {}
        with pysam.AlignmentFile(file) as sam:
            for record in sam.fetch(region=gene.get_wide_region().samtools(prefix="chr")):
                if not record.cigarstring or "H" in record.cigarstring:  # only valid alignments
                    continue
                if record.is_supplementary:  # avoid supplementary alignments
                    continue
                if not record.query_sequence:
                    continue
                fragment = record.query_name
                read = reads.setdefault(fragment, {})
                seq = record.query_sequence
                start, s_start = record.reference_start, 0
                for op, size in record.cigartuples:
                    if op == 2:  # Deletion
                        for i in range(size):
                            read[start + i] = "-"
                        start += size
                    elif op == 1:  # Insertion
                        read[start] = "+" + seq[s_start : s_start + size]
                        s_start += size
                    elif op == 4:  # Soft-clip
                        s_start += size
                    elif op in [0, 7, 8]:  # M, X and =
                        for i in range(size):
                            if start + i in gene and gene[start + i] != seq[s_start + i]:
                                read[start + i] = seq[s_start + i]
                            else:  # We ignore all mutations outside the RefSeq region
                                read[start + i] = "_"
                        start += size
                        s_start += size
        m = []
        for p in reads.values():  # iterate through reads
            v = [0] * len(indices)
            for i, o in p.items():
                if (i, o) in indices:
                    v[indices[i, o]] = 1
            
            m.append(ETL.squeeze(v) if squeezed else v)
        
        return np.array(m)

    @staticmethod
    def squeeze(allele):
        return [allele[2 * j + 1] for j in range(len(allele) // 2)]
