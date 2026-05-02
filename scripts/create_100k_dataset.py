import argparse
import json 
import pandas as pd
import json
from Bio.Seq import Seq

# 1-indexed
nuc_start = 21563
nuc_end = 25384

nuc_root_seq_path = "../data/pathogen/sars_cov_2_spike/auspice_root-sequence.json"
tsv_path = "../data/metadata.tsv"
save_name = "100k_dataset.fasta"

str_to_mut = lambda x: (x[0], int(x[1:-1]), x[-1])
str_to_del = lambda x: tuple([int(y) for y in x.split("-")]) if len(x.split("-")) > 1 else (int(x), int(x) + 1)

with open(nuc_root_seq_path, "r") as f:
    auspice_json = json.load(f)

nuc_root_seq = auspice_json["S"]

seq_df = pd.read_csv(tsv_path, sep = '\t')
del_list = list(seq_df["deletions"].fillna(""))
sub_list = list(seq_df["aaSubstitutions"].fillna(""))

seqs = []
for idx,(dels,subs) in enumerate(zip(del_list, sub_list)):
    if idx % 100 == 0:
        print(".", end="", flush=True)

    wt_copy = list(nuc_root_seq)
    if len(subs) > 0:
        s_i_spike = [str_to_mut(x[2:]) for x in subs.split(",") if x[2:] == "S:"]
        for (aa1,loc,aa2) in s_i_spike:
            assert wt_copy[loc] == aa1, "wrong location for mut :("
            wt_copy[loc] = aa2

    if len(dels) > 0:
        d_i = [str_to_del(x) for x in dels.split(",")]
        d_i_spike = [(d1 - nuc_start, d2 - nuc_start) for (d1,d2) in d_i if (d1 >= nuc_start and d1 <= nuc_end)]

        for (d1,d2) in d_i_spike:
            gap_len = (d2 - d1 + 1) // 3
            d1 = d1 // 3
            wt_copy[d1:d1+gap_len] = ["-" for _ in range(gap_len)]

    seqs.append("".join(wt_copy))

print(set([len(x) for x in seqs]))
with open("../data/dset_100k.json", "w") as f:
    json.dump(seqs, f)
print("wrote dataset!")
