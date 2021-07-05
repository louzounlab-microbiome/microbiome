import pandas as pd
from pathlib import Path
def prepare_to_preprocess(df:pd.DataFrame)->pd.DataFrame:
    all_bacteria_taxonomy_form=df.index
    df=df.reset_index(drop=True)
    df=df.assign(taxonomy=all_bacteria_taxonomy_form)
    df=df.transpose()
    df=df.rename_axis('ID')

    return df




path_to_stool_otu_table = Path('../data/origninal_data/stool_OTU_table.tsv')
path_to_saliva_otu_table = Path('../data/origninal_data/Saliva_OTU_table_220620.txt')
path_to_stool_mapping_table = Path('../data/origninal_data/stool_mapping_file.tsv')
path_to_saliva_mapping_table = Path('../data/origninal_data/saliva_mapping file.tsv')

#Load the otu tables
transposed_stool_otu_table=pd.read_csv(path_to_stool_otu_table,sep='\t',index_col=0)
transposed_saliva_otu_table=pd.read_csv(path_to_saliva_otu_table,sep='\t',index_col=0)
# Transform them into a shape that can be interpreted by the preprocess pipeline
ready_to_preprocess_stool_otu_table=prepare_to_preprocess(transposed_stool_otu_table)
ready_to_preprocess_saliva_otu_table=prepare_to_preprocess(transposed_saliva_otu_table)
# save the tables
ready_to_preprocess_stool_otu_table.to_csv(Path('../data/data_used/basic_tables/stool_otu.csv'))
ready_to_preprocess_saliva_otu_table.to_csv(Path('../data/data_used/basic_tables/saliva_otu.csv'))
# Load the mapping tables
# make them interpretable
# Parse the dates in the DATE column into DateTime objects
stool_mapping_table=pd.read_csv(path_to_stool_mapping_table,sep='\t',index_col=0,parse_dates=['DATE'])
stool_mapping_table.rename({'fustatus': 'Tag'}, inplace=True, axis=1)
stool_mapping_table.index.rename('ID', inplace=True)

# Transform the data into a shape in which all patients samples are sorted based on their dates.
stool_mapping_table.sort_values(by=['subjid','DATE'],inplace=True)
# Create Time points for the samples
stool_mapping_table['TimePoint']= stool_mapping_table.groupby('subjid').cumcount()+1
# Please notice that the dates will be saved as strings
stool_mapping_table.to_csv(Path('../data/data_used/basic_tables/stool_mapping_table.csv'))

# Parse the dates in the DATE column into DateTime objects

saliva_mapping_table=pd.read_csv(path_to_saliva_mapping_table,sep='\t',index_col=0,parse_dates=['DATE'])
saliva_mapping_table.rename({'fustatus': 'Tag'}, inplace=True, axis=1)
saliva_mapping_table.index.rename('ID', inplace=True)

# Transform the data into a shape in which all patients samples are sorted based on their dates.
saliva_mapping_table.sort_values(by=['subjid','DATE'],inplace=True)
# Create Time points for the samples
saliva_mapping_table['TimePoint']= saliva_mapping_table.groupby('subjid').cumcount()+1

# Please notice that the dates will be saved as strings
saliva_mapping_table.to_csv(Path('../data/data_used/basic_tables/saliva_mapping_table.csv'))

