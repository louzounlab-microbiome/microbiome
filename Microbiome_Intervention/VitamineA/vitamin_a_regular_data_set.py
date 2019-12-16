from GVHD_BAR.anova_significant_correlation import get_significant_bact_for_col
from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
from infra_functions.general import apply_pca
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import pandas as pd
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
n_components = 20


class VitaminARegularDataLoader:
    def __init__(self, title, bactria_as_feature_file, samples_data_file, taxnomy_level, allow_printing, perform_anna_preprocess):
        OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, bactria_as_feature_file),
                             os.path.join(SCRIPT_DIR, samples_data_file),
                             from_QIIME=False, id_col='#OTU ID', taxonomy_col='Taxonomy')

        preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=taxnomy_level,
                                             taxonomy_col='Taxonomy',
                                             preform_taxnomy_group=True)
        self._preproccessed_data = preproccessed_data

        otu_after_pca_wo_taxonomy, pca_obj, _ = apply_pca(preproccessed_data, n_components=n_components,
                                                          visualize=False)
        self._pca_obj = pca_obj

        bacteria = preproccessed_data.columns
        with open(os.path.join("bacteria.txt"), "w") as file:
            for b in bacteria:
                file.write(b + '\n')

        index_to_id_map = {}
        id_to_features_map = {}
        for i, row in enumerate(otu_after_pca_wo_taxonomy.values):
            id_to_features_map[preproccessed_data.index[i]] = row
            index_to_id_map[i] = preproccessed_data.index[i]

        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map
        ids_list = preproccessed_data.index.tolist()
        self._ids_list = ids_list

        child_vaccines_column = 'ChildVaccines'
        id_to_child_vaccines_map = {}
        for sample in ids_list:
            child_num = OtuMf.mapping_file.loc[sample, child_vaccines_column]
            id_to_child_vaccines_map[sample] = child_num

        pcv_column = 'PCV'
        id_to_child_pcv_map = {}
        for sample in ids_list:
            child_num = OtuMf.mapping_file.loc[sample, pcv_column]
            id_to_child_pcv_map[sample] = child_num

        rota_column = 'Rota'
        id_to_child_rota_map = {}
        for sample in ids_list:
            child_num = OtuMf.mapping_file.loc[sample, rota_column]
            id_to_child_rota_map[sample] = child_num

        # one hots
        is_to_one_hot_map = {sample: [] for sample in ids_list}
        for sample in ids_list:
            is_to_one_hot_map[sample] = [id_to_child_pcv_map[sample],
                                         id_to_child_rota_map[sample]]

        # anova test
        p_val_df = pd.DataFrame(columns=["PCV", "Rota", "PCV*Rota"], index=preproccessed_data.columns)

        id_to_bacteria_map = {}
        for id in ids_list:
            id_to_bacteria_map[id] = preproccessed_data.loc[id]

        for bact in preproccessed_data.columns:
            bact_df = pd.DataFrame(columns=["bact", "PCV", "Rota"], index=ids_list)

            for i in ids_list:
                bact_val = id_to_bacteria_map[i][bact]
                pcv = id_to_child_pcv_map[i]
                rota = id_to_child_rota_map[i]
                bact_df.loc[i] = [bact_val, pcv, rota]

            # anova test
            bact_df["bact"] = pd.to_numeric(bact_df["bact"])
            model = ols('bact ~ C(PCV) + C(Rota) + C(PCV):C(Rota)', data=bact_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_val_df.loc[bact] = [anova_table["PR(>F)"]["C(PCV)"], anova_table["PR(>F)"]["C(Rota)"],
                                  anova_table["PR(>F)"]["C(PCV):C(Rota)"]]

        p_val_df.to_csv(os.path.join("p_val_df_taxonomy_level_" + str(taxnomy_level) + ".csv"))

        print("find significant bacteria")
        p_0 = 0.05
        pcv_significant_bact = get_significant_bact_for_col("PCV", p_val_df, bacteria, p_0=p_0)
        p_val_df.loc[pcv_significant_bact].to_csv("pcv_significant_bact_df_taxonomy_level_" + str(taxnomy_level) + ".csv")
        rota_significant_bact = get_significant_bact_for_col("Rota", p_val_df, bacteria, p_0=p_0)
        p_val_df.loc[rota_significant_bact].to_csv("rota_significant_bact_df_taxonomy_level_" + str(taxnomy_level) + ".csv")
        pcv_rota_significant_bact = get_significant_bact_for_col("PCV*Rota", p_val_df, bacteria, p_0=p_0)
        p_val_df.loc[pcv_rota_significant_bact].to_csv("pcv_rota_significant_bact_df_taxonomy_level_" + str(taxnomy_level) + ".csv")


if __name__ == "__main__":
    task = 'success task'
    bactria_as_feature_file = 'ok16_va_otu_table.csv'
    samples_data_file = 'metadata_ok16_va_for_dafna.csv'
    tax = 6

    allergy_dataset = VitaminARegularDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file,  taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)
    print(allergy_dataset)
