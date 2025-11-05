# graph_genomic_data
Genomic data as optimised graph data structures


A Phase Set (PS) is a value (often an integer) in the VCF format that tells us which variant positions belong to the same contiguous phase block. If two or more variants share the same PS value for a given sample, it implies that we know the relative ordering of alleles (which alleles are on the same haplotype) across those variants.

For instance:

Variant at position 1000: genotype = 0|1, PS=10
Variant at position 1010: genotype = 1|0, PS=10
If these two positions share the same PS=10 for that sample, it means the 0 and 1 at position 1000 lie on the same haplotypes as the 1 and 0 at position 1010, respectively. We can link the alleles across these two positions for each haplotype:

Haplotype #1 might have 0 at pos 1000 and 1 at pos 1010,
Haplotype #2 might have 1 at pos 1000 and 0 at pos 1010.
Linking Consecutive Variants in the Same PS
In the code, we maintain a structure—something like:

python
Copy
Edit
phase_paths[sample][ps][haplotype_index] = [list_of_node_ids]
sample: The individual/sample name.
ps: The phase set identifier.
haplotype_index: Which haplotype for that diploid sample (typically 0 or 1).
[list_of_node_ids]: The graph nodes (alleles) at each consecutive position that belong to this haplotype in the same phase set.
We do this by:

Reading each variant record (one genomic position).
Checking the genotype for a given sample. If it is phased (e.g. 0|1), we look at the PS value to see which phase set it belongs to.
We then append the “allele node” (e.g. "chr1:1000:A" for reference or "chr1:1000:G" for an alternate) to the appropriate list in phase_paths[sample][ps][haplotype_index].
After processing all variants, each phase_paths[sample][ps][haplotype_index] represents the ordered list of alleles that occur on that haplotype for all positions in that phase set. To build our graph:

We start at a special START node,
Connect to the first allele node in the haplotype list,
Then connect each consecutive pair of allele nodes,
Finally connect the last allele node to a special END node.
Hence, all consecutive variants sharing the same PS get connected in a continuous path, accurately reflecting the haplotype. This “linking” is the key to representing phased data in a graph: each path from variant to variant within the same phase set mirrors a single chromosome copy.
