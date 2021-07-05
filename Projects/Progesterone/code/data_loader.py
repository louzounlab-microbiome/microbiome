import pandas as pd
from pathlib import Path
otu_path = Path('../data/exp1/original_data/otu.txt')
mapping_file_path = Path('../data/exp1/original_data/mapping_file.tsv')
otu_table = pd.read_csv(otu_path, index_col=0, sep='\t').T
mapping_table = pd.read_csv(mapping_file_path, index_col=0, sep='\t')
# Prepare the tables towards yoel's preprocess
otu_table.rename_axis('ID', inplace=True)
# The taxonomy is given by the columns of the otu table.
taxonomy = list(otu_table.columns)
otu_table.columns=range(otu_table.shape[1])

# add the taxonomy as the last row.
otu_table.loc['taxonomy'] = taxonomy

mapping_table.rename_axis('ID',inplace=True)
# Select a Tag column, In this case we are no trying to predict anything so I chose an arbitrary column.
mapping_table.rename({'MouseNumber':'Tag'},axis=1,inplace=True)

otu_table.to_csv(Path('../data/exp1/used_data/basic_data/otu.csv'))
mapping_table.to_csv(Path('../data/exp1/used_data/basic_data/mapping_table.csv'))
