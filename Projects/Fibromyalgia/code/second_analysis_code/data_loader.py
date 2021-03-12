import pandas as pd
from pathlib import Path
import numpy as np
otu_path = Path('../../data/original_data/otu.txt')
mapping_file_path = Path('../../data/original_data/mapping_file.tsv')
relevant_mapping_table_columns = ['age','sex','smoker','sport','Antibiotic_3_last_months','IBS_questionnaire']
transposed_otu_table=pd.read_csv(otu_path,index_col=0,sep='\t')
transposed_otu_table.set_index('taxonomy',inplace=True)
otu_table = transposed_otu_table.T

mapping_table = pd.read_csv(mapping_file_path, index_col=0, sep='\t')
mapping_table=mapping_table.replace('na',np.NAN)

# Prepare the data towards Yoel's preprocess

# The taxonomy is given by the columns of the otu table.
taxonomy = list(otu_table.columns)
otu_table.columns=range(otu_table.shape[1])
otu_table.rename_axis('ID',inplace=True)

# add the taxonomy as the last row.
otu_table.loc['taxonomy'] = taxonomy

mapping_table.rename_axis('ID',inplace=True)
# Select a Tag column, In this case we are no trying to predict anything so I chose an arbitrary column.
mapping_table.rename({'ReversePrimer':'Tag'},inplace=True,axis=1)

otu_table.to_csv(Path('../data/used_data/basic_data/otu.csv'))
mapping_table.to_csv(Path('../data/used_data/basic_data/mapping_table.csv'))

