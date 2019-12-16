import os

import pandas as pd
def combine_tax_levels(levels_list):
    final_tax = ""
    for l in levels_list:
        if pd.isna(l):
            break
        if "/" in l:
            break
        final_tax += l + ";"
    return final_tax[:-1]


def adjust_otu_file(feature_path, tax_path, folder):
    otu_file = pd.read_csv(feature_path)
    seq_list = otu_file.columns
    tax_file = pd.read_csv(tax_path)
    seq_to_bact_df = pd.DataFrame(columns=["SEQ", "TAX"])
    seq_to_bact_map = {}
    for row in tax_file.values:
        t = combine_tax_levels(row[1:])
        seq_to_bact_df.loc[len(seq_to_bact_df)] = [row[0], t]
        seq_to_bact_map[row[0]] = t
    seq_to_bact_df.to_csv(os.path.join(folder, "seq_to_bact.csv"), index=False)
    tax_list = [seq_to_bact_map[s] for s in seq_list[1:]]
    otu_file.columns = ['#SampleID'] + tax_list
    otu_file.loc[len(otu_file)] = ['taxonomy'] + tax_list
    otu_file = otu_file.T
    otu_file.to_csv(os.path.join(folder, "dafna_proccessed_abundance.csv"))

if __name__ == "__main__":
    # df = pd.read_csv(os.path.join("MITRE_data_t1d", 'diabimmune_t1d_metaphlan_table.csv'))
    adjust_otu_file(os.path.join("MITRE_data_bokulich", 'abundance.csv'),
                    os.path.join("MITRE_data_bokulich", 'dada2_placements.csv'),
                    "MITRE_data_bokulich")

    adjust_otu_file(os.path.join("MITRE_data_david", 'abundance.csv'),
                    os.path.join("MITRE_data_david", 'mothur_placements.csv'),
                    "MITRE_data_david")

    adjust_otu_file(os.path.join("MITRE_data_digiulio", 'abundance.csv'),
                    os.path.join("MITRE_data_digiulio", 'dada2_placements.csv'),
                    "MITRE_data_digiulio")

    adjust_otu_file(os.path.join("MITRE_data_karelia", 'abundance.csv'),
                    os.path.join("MITRE_data_karelia", 'dada2_placements.csv'),
                    "MITRE_data_karelia")



