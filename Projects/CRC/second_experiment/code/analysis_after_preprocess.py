import pickle
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def transform_pvalue_to_astrix(pval: float):
    if pval <= 0.001:
        return '***'
    elif pval <= 0.01:
        return '**'
    elif pval <= 0.05:
        return '*'
    return ' '


with open(Path('../data/data_used/otuMF'), 'rb') as otu_file:
    otuMf = pickle.load(otu_file)
bacteria_names_from_previous_project_df = pd.read_csv(
    Path('../data/data_used/bacteria_names_from_previous_project.csv'))
bacteria_names_from_previous_project = bacteria_names_from_previous_project_df['bacteria_names'].values
bacteria_names_from_current_project = otuMf.otu_features_df.columns.values
mutal_bacteria = list(filter(lambda x: x in bacteria_names_from_previous_project, bacteria_names_from_current_project))

significant_bacteria_negative_correlation = [
    'k__Bacteria; p__Bacteroidetes; c__Bacteroidia; o__Bacteroidales; f__Prevotellaceae; g__Prevotella',
    'k__Bacteria; p__Bacteroidetes; c__Bacteroidia; o__Bacteroidales; f__Rikenellaceae; g__',
    'k__Bacteria; p__Firmicutes; c__Clostridia; o__Clostridiales; f__Clostridiaceae; g__Clostridium',
    'k__Bacteria; p__Actinobacteria; c__Coriobacteriia; o__Coriobacteriales; f__Coriobacteriaceae; g__',
    'k__Bacteria; p__Cyanobacteria; c__4C0d-2; o__YS2; f__; g__',
    'k__Bacteria; p__Actinobacteria; c__Actinobacteria; o__Bifidobacteriales; f__Bifidobacteriaceae; g__Bifidobacterium',
    'k__Bacteria; p__Proteobacteria; c__Alphaproteobacteria; o__RF32; f__; g__',
    'k__Bacteria; p__Bacteroidetes; c__Bacteroidia; o__Bacteroidales; f__Rikenellaceae; g__',
    'k__Bacteria; p__Bacteroidetes; c__Bacteroidia; o__Bacteroidales; f__[Odoribacteraceae]; g__Odoribacter',
    'k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; f__Staphylococcaceae; g__Staphylococcus'
]

significant_bacteria_positive_correlation = [
    'k__Bacteria; p__Firmicutes; c__Clostridia; o__Clostridiales; f__Lachnospiraceae; g__Dorea',
    'k__Bacteria; p__Proteobacteria; c__Deltaproteobacteria; o__Desulfovibrionales; f__Desulfovibrionaceae; g__Bilophila',
    'k__Bacteria; p__Firmicutes; c__Clostridia; o__Clostridiales; f__Lachnospiraceae; g__Roseburia',
    'k__Bacteria; p__Deferribacteres; c__Deferribacteres; o__Deferribacterales; f__Deferribacteraceae; g__Mucispirillum',
    'k__Bacteria; p__Firmicutes; c__Erysipelotrichi; o__Erysipelotrichales; f__Erysipelotrichaceae; g__',
    'k__Bacteria; p__Firmicutes; c__Clostridia; o__Clostridiales; f__Peptococcaceae; g__rc4-4',
    'k__Bacteria; p__Firmicutes; c__Clostridia; o__Clostridiales; f__Clostridiaceae; g__Clostridium'

]

otu_with_treatment_and_transplant = pd.merge(otuMf.otu_features_df,
                                             otuMf.extra_features_df[['Transplant', 'Treatment']],
                                             left_index=True, right_index=True)
treatment_transplant_otu_groups = otu_with_treatment_and_transplant.groupby(['Treatment', 'Transplant'])
all_bacteria_for_all_groups = [[group[bacteria] for _, group in treatment_transplant_otu_groups] for
                               bacteria in mutal_bacteria]
results_rows = []
for bacteria_within_different_groups, bacteria_name in zip(all_bacteria_for_all_groups, mutal_bacteria):
    anova_p_value = stats.f_oneway(*bacteria_within_different_groups)[1]
    if bacteria_name in significant_bacteria_negative_correlation:
        correlation_factor = -1
    elif bacteria_name in significant_bacteria_positive_correlation:
        correlation_factor = +1
    else:
        correlation_factor = 0
    row = [anova_p_value, correlation_factor]
    groups_average = []
    for group in bacteria_within_different_groups:
        groups_average.append(group.mean())
    row.extend(groups_average)
    results_rows.append(row)

columns_names = ['Pvalue', 'Correlation in original']
group_names = treatment_transplant_otu_groups.groups.keys()
modified_group_names = list(map(lambda x: str(x) + ' Average', group_names))
columns_names.extend(modified_group_names)
result_df = pd.DataFrame(results_rows, index=mutal_bacteria, columns=columns_names)
result_df['Pvalue'] = result_df['Pvalue'].apply(transform_pvalue_to_astrix)
result_df.to_csv(Path('../data/exported_data/Anova_results.csv'), index_label='Bacteria')
