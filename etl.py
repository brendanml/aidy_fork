import numpy as np
import os
import pysam
import aldy.gene
import aldy.common


class ETL:
    def __init__(self, gene_name, coverage):
        self.coverage = coverage
        gene_path = aldy.common.script_path(f"aldy.resources.genes/{gene_name}.yml")
        self.gene = aldy.gene.Gene(gene_path, genome="hg38")
        os.system('mkdir -p temp/sim')
        if not os.path.exists("temp/chr10.fa.bwt"):
            os.system("wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr10.fa.gz")
            os.system("gunzip chr10.fa.gz && mv chr10.fa temp/")
            os.system("~/libs/bwa/bwa index temp/chr10.fa")
        
        # Allele dictionary: name -> [list_of_mutations]
        alleles = {}
        for ma_n, ma in self.gene.alleles.items():
            # for mi_n, mi in ma.minors.items():
            alleles[ma_n] = list(ma.func_muts)
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

        # Do simulations
        # Simulate allele sequences with ART and align them to the reference
        for a_n, a_muts in alleles.items():
            if os.path.exists(f"temp/sim/{self.gene.name}_{a_n}_1.fq") and os.path.exists(f"temp/sim/{self.gene.name}_{a_n}_2.fq"):
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
                "~/libs/bwa/bwa mem -t 4 temp/chr10.fa "
                f"temp/sim/{self.gene.name}_{a_n}_1.fq temp/sim/{self.gene.name}_{a_n}_2.fq | "
                f"/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx512/Core/samtools/1.17/bin/samtools sort --write-index -o temp/sim/{self.gene.name}_{a_n}.bwa.bam##idx##temp/sim/{self.gene.name}_{a_n}.bwa.bai")

        # This is database matrix (rows: alleles, columns: variants)
        self.allele_keys = list(allele_vectors.keys())
        self.allele_db = np.array([allele_vectors[k] for k in self.allele_keys])
        self.squeezed_allele_db = np.array([self.squeeze(allele_vectors[k]) for k in self.allele_keys])
    
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
    
    def sample(self, selected_alleles, squeezed):
        reads = []
        for allele in selected_alleles:
            reads.extend(self.get_sample(f"temp/sim/{self.gene.name}_{allele}.bwa.bam", self.gene, self.indices, squeezed).tolist())
        
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
