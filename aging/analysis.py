from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
from infra_functions.general import apply_pca

if __name__ == "__main__":
    OtuMf = OtuMfHandler('aging_otu_table.csv', 'mf.csv', from_QIIME=True)
    preproccessed_data = preprocess_data(OtuMf.otu_file_wo_taxonomy, visualize_data=True)
    otu_after_pca_wo_taxonomy, _ = apply_pca(preproccessed_data, n_components=10)
    # otu_after_pca = OtuMf.add_taxonomy_col_to_new_otu_data(otu_after_pca_wo_taxonomy)
    merged_data_after_pca = OtuMf.merge_mf_with_new_otu_data(otu_after_pca_wo_taxonomy)
    a=1