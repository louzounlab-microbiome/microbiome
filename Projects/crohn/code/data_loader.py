from Preprocess.preprocess_grid import from_biom
from pathlib import Path

from_biom(Path('../data/original_data/IIRN_feature-table.biom'),Path('../data/original_data/IIRN_taxonomy.tsv'),Path('../data/original_data/otu.csv'))
