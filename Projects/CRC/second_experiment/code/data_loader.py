from pathlib import Path
import pandas as pd
"Load ordinary project files "
path_to_otu_table = Path('../data/original_data/otu.txt')
path_to_taxonomy_table = Path('../data/original_data/taxonomy.tsv')
path_to_mapping_table = Path('../data/original_data/mapping_table.csv')

transposed_otu_table = pd.read_csv(path_to_otu_table, sep='\t', index_col=0)
# The confidence is not necessary to our analysis therefore, we remove it.
taxonomy_table = pd.read_csv(path_to_taxonomy_table, delimiter='\t', index_col=0).drop('Confidence', axis=1)
# The preprocess demands that the taxon column will be called taxonomy
taxonomy_table.rename({'Taxon': 'taxonomy'}, axis=1, inplace=True)

mapping_table = pd.read_csv(path_to_mapping_table, index_col=0)
# The preprocesses demands the taxonomy column to be attached to the otu table as a row, so first we add it as a column.
transposed_otu_table_with_taxonomy_col = pd.merge(transposed_otu_table, taxonomy_table, left_index=True,
                                                  right_index=True)
# And then we transpose th table.
otu_table = transposed_otu_table_with_taxonomy_col.transpose()

# The following renames are also according to the preprocess pipeline.
otu_table.index.rename('ID', inplace=True)

mapping_table.rename({'TumorLoad': 'Tag'}, inplace=True, axis=1)
mapping_table.index.rename('ID', inplace=True)
otu_table.to_csv(Path('../data/data_used/otu.csv'))
mapping_table.to_csv(Path('../data/data_used/mapping_table.csv'))
