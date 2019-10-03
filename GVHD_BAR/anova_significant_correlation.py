from os.path import join

from GVHD_BAR.prepare_data import prepare_data
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

from dafna.general_functions import shorten_bact_names
from dafna.plot_anove import plot_anove_significant_bacteria
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import seaborn as sns


def get_organize_data_for_anova(taxnomy_level=6):
    # get data
    x_for_deep, y_for_deep, x_for_deep_censored, y_for_deep_censored, censored_data, not_censored,\
    otu_after_pca_wo_taxonomy, OtuMf, preproccessed_data = prepare_data(taxnomy_level=taxnomy_level, preform_z_scoring=True)
    preproccessed_data = preproccessed_data.drop(['Unassigned'], axis=1)


    mapping_file = OtuMf.mapping_file
    samples = mapping_file.index
    bacteria = preproccessed_data.columns
    time = mapping_file['Day_Group']
    state = mapping_file['Mucositis_Grade']
    sample_to_time_map = {}
    sample_to_state_map = {}

    for sample, t, s in zip(samples, time, state):
        sample_to_time_map[sample] = t
        sample_to_state_map[sample] = s

    # get only samples from '0_6' and '7_13' time stamp
    ids = sub_set_samples(sample_to_time_map.items(), ['0_6', '7_13'])
    time_0_6_ids = ids[0]
    time_7_13_ids = ids[1]

    # get only samples from 0, 1, 3, 4 state stamp
    [state_0_ids, state_1_ids, state_3_ids, state_4_ids] = sub_set_samples(sample_to_state_map.items(), [0, 1, 3, 4])

    valid_ids = [s for s in samples if s in time_0_6_ids + time_7_13_ids
                 and s in state_0_ids + state_1_ids + state_3_ids + state_4_ids]
    valid_ids = [i for i in valid_ids if i in preproccessed_data.index]  # check if id has bacteria info

    # create maps
    id_to_bacteria_map = {}
    id_to_time_map = {}
    id_to_state_map = {}
    state_to_group_state_map = {0: "0-1", 1: "0-1", 3: "3-4", 4: "3-4"}

    for id in valid_ids:
        id_to_bacteria_map[id] = preproccessed_data.loc[id]
        id_to_time_map[id] = time.loc[id]
        id_to_state_map[id] = state_to_group_state_map[state.loc[id]]

    return id_to_bacteria_map, id_to_time_map, id_to_state_map, preproccessed_data, valid_ids, bacteria


def sub_set_samples(dict, vaild_options):   # get only samples from '0_6' and '7_13' time stamp
    ids = [[] for o in range(len(vaild_options))]

    for key, val in dict:
        if val in vaild_options:
            i = vaild_options.index(val)
            ids[i].append(key)
    return ids


def get_significant_bact_for_col(col_name, p_val_df, bacteria, p_0=0.05):
    significant_bact = []
    p_values = p_val_df[col_name]
    num_of_bact = len(p_values)
    bact_p = [[b, p] for b, p in zip(bacteria, p_values)]
    bact_p = sorted(bact_p, key=lambda b_p: b_p[1])
    # check for each bacteria if her p value is significant
    for i, p in enumerate(bact_p):
        threshold = p_0 * (i + 1) / num_of_bact
        if bact_p[i][1] < threshold:  # significant
            significant_bact.append(bact_p[i][0])

    return significant_bact


def get_sorted_significant_bacteria_colors(significant_bact, time_significant_bact, state_significant_bact,
                                           mutual_p_values, time_p_values, state_p_values):
    p_values = []
    p_values_colors = []
    for i, b in enumerate(significant_bact):
        """
        if b in time_significant_bact and b in state_significant_bact:  # or if mutual_p_values[b] < p_0  ????
            # p_values.append(mutual_p_values.loc[b]["time*state"])
            # p_values_colors.append(anove_labels_colors[2])
        
            # or add the bacteria twice for each time and state
            p_values.append(time_p_values.loc[b]["time"])
            p_values_colors.append(anove_labels_colors[0])
            p_values.append(state_p_values.loc[b]["state"])
            p_values_colors.append(anove_labels_colors[1])
        """
        if b in time_significant_bact and b not in significant_bact[:i]:  # b didn't apeare before
            p_values.append(time_p_values.loc[b]["time"])
            p_values_colors.append(anove_labels_colors[0])
        elif b in state_significant_bact:
            p_values.append(state_p_values.loc[b]["state"])
            p_values_colors.append(anove_labels_colors[1])

    # sort by p values
    bact_p_color = [[b, p, c] for b, p, c in zip(list(significant_bact), list(p_values), list(p_values_colors))]
    bact_p_color = sorted(bact_p_color, key=lambda b_p: b_p[1])
    significant_bact = [info[0] for info in bact_p_color]
    p_values = [info[1] for info in bact_p_color]
    p_values_colors = [info[2] for info in bact_p_color]
    return significant_bact, p_values, p_values_colors


def get_sub_set_bacteria_using_lefse(data):
    lefse_bacteria = ["k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.g__Moryella",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.g__Oribacterium",
                      "k__Bacteria.p__Bacteroidetes.c__Flavobacteriia.o__Flavobacteriales.f__Flavobacteriaceae",
                      "k__Bacteria.p__Firmicutes.c__Bacilli.o__Lactobacillales.f__Lactobacillaceae.g__Lactobacillus",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.g__Catonella",
                      "k__Bacteria.p__Bacteroidetes.c__Flavobacteriia.o__Flavobacteriales.f__Flavobacteriaceae.g__Capnocytophaga",
                      "k__Bacteria.p__Firmicutes.c__Bacilli.o__Lactobacillales.f__Lactobacillaceae",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.Unclassified_member_of_Clostridiales",
                      "k__Bacteria.p__Bacteroidetes.c__Flavobacteriia",
                      "k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales.f__Prevotellaceae",
                      "k__Bacteria.p__Bacteroidetes.c__Flavobacteriia.o__Flavobacteriales",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.Unclassified_member_of_Lachnospiraceae",
                      "k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales.f__Prevotellaceae.g__Prevotella",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.g__Butyrivibrio",
                      "k__Bacteria.p__Proteobacteria.c__Betaproteobacteria.o__Neisseriales.f__Neisseriaceae.g__Kingella",
                      "k__Bacteria.p__Bacteroidetes",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Peptostreptococcaceae.g__Filifactor",
                      "k__Bacteria.p__Proteobacteria.c__Betaproteobacteria.o__Neisseriales.f__Neisseriaceae.g__Eikenella",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Veillonellaceae.g__Schwartzia",
                      "k__Bacteria.p__Fusobacteria.c__Fusobacteriia.o__Fusobacteriales.f__Leptotrichiaceae.g__Leptotrichia",
                      "k__Bacteria.p__Proteobacteria.c__Alphaproteobacteria.o__Rhizobiales",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f___Mogibacteriaceae_",
                      "k__Bacteria.p__Proteobacteria.c__Betaproteobacteria.o__Neisseriales.f__Neisseriaceae.g__Neisseria",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f___Mogibacteriaceae_.Unclassified_member_of_Mogibacteriaceae",
                      "k__Bacteria.p__Firmicutes.c__Erysipelotrichi.o__Erysipelotrichales",
                      "k__Bacteria.p__Firmicutes.c__Bacilli.o__Lactobacillales.f__Carnobacteriaceae.g__Granulicatella",
                      "k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Actinomycetales.f__Actinomycetaceae",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.g__Moryella",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae",
                      "k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales",
                      "k__Bacteria.p__Firmicutes.c__Bacilli.o__Bacillales.f__Staphylococcaceae",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Ruminococcaceae",
                      "k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Actinomycetales.f__Actinomycetaceae.g__Actinomyces",
                      "k__Bacteria.p__Fusobacteria.c__Fusobacteriia",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.g__Oribacterium",
                      "k__Bacteria.p__Actinobacteria.c__Coriobacteriia",
                      "k__Bacteria.p__Bacteroidetes.c__Flavobacteriia.o__Flavobacteriales.f__Flavobacteriaceae",
                      "k__Bacteria.p__Firmicutes.c__Bacilli.o__Gemellales.f__Gemellaceae",
                      "k__Bacteria.p__Bacteroidetes.c__Bacteroidia",
                      "k__Bacteria.p__Proteobacteria.c__Alphaproteobacteria.o__Rhizobiales.f__Methylobacteriaceae.g__Methylobacterium",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.g__Catonella",
                      "k__Bacteria.p__Fusobacteria.c__Fusobacteriia.o__Fusobacteriales.f__Leptotrichiaceae",
                      "k__Bacteria.p__Firmicutes.c__Bacilli.o__Bacillales.f__Staphylococcaceae.g__Staphylococcus",
                      "k__Bacteria.p__Bacteroidetes.c__Flavobacteriia.o__Flavobacteriales.f__Flavobacteriaceae.g__Capnocytophaga",
                      "k__Bacteria.p__Proteobacteria.c__Epsilonproteobacteria.o__Campylobacterales.f__Campylobacteraceae",
                      "k__Bacteria.p__Actinobacteria.c__Actinobacteria",
                      "k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Actinomycetales.f__Micrococcaceae",
                      "k__Bacteria.p__Actinobacteria",
                      "k__Bacteria.p__Firmicutes.c__Erysipelotrichi.o__Erysipelotrichales.f__Erysipelotrichaceae",
                      "k__Bacteria.p__Proteobacteria",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f___Tissierellaceae_",
                      "k__Bacteria.p__TM7.c__TM7_3.Unclassified_member_of_TM7_3",
                      "k__Bacteria.p__Synergistetes.c__Synergistia.o__Synergistales",
                      "k__Bacteria.p__Firmicutes.c__Bacilli.o__Gemellales.f__Gemellaceae.Unclassified_member_of_Gemellaceae",
                      "k__Bacteria.p__Fusobacteria",
                      "k__Bacteria.p__Firmicutes.c__Erysipelotrichi.o__Erysipelotrichales.f__Erysipelotrichaceae.g__Bulleidia",
                      "k__Bacteria.p__Firmicutes",
                      "k__Bacteria.p__Proteobacteria.c__Gammaproteobacteria.o__Pasteurellales",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.Unclassified_member_of_Clostridiales",
                      "k__Bacteria.p__Proteobacteria.c__Betaproteobacteria",
                      "k__Bacteria.p__Actinobacteria.c__Coriobacteriia.o__Coriobacteriales",
                      "k__Bacteria.p__Firmicutes.c__Bacilli.o__Bacillales",
                      "k__Bacteria.p__Bacteroidetes.c__Flavobacteriia",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Veillonellaceae.Unclassified_member_of_Veillonellaceae",
                      "k__Bacteria.p__Tenericutes.c__Mollicutes",
                      "k__Bacteria.p__Fusobacteria.c__Fusobacteriia.o__Fusobacteriales",
                      "k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales.f__Prevotellaceae",
                      "k__Bacteria.p__Fusobacteria.c__Fusobacteriia.o__Fusobacteriales.f__Fusobacteriaceae",
                      "k__Bacteria.p__Synergistetes.c__Synergistia.o__Synergistales.f__Dethiosulfovibrionaceae",
                      "k__Bacteria.p__SR1",
                      "k__Bacteria.p__Bacteroidetes.c__Flavobacteriia.o__Flavobacteriales",
                      "k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales.f__Prevotellaceae.g__Prevotella",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.Unclassified_member_of_Lachnospiraceae",
                      "k__Bacteria.p__Firmicutes.c__Erysipelotrichi",
                      "k__Bacteria.p__Proteobacteria.c__Epsilonproteobacteria.o__Campylobacterales",
                      "k__Bacteria.p__Actinobacteria.c__Coriobacteriia.o__Coriobacteriales.f__Coriobacteriaceae",
                      "k__Bacteria.p__Proteobacteria.c__Epsilonproteobacteria.o__Campylobacterales.f__Campylobacteraceae.g__Campylobacter",
                      "k__Bacteria.p__Proteobacteria.c__Alphaproteobacteria.o__Rhizobiales.f__Methylobacteriaceae",
                      "k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales.f___Paraprevotellaceae_.g___Prevotella_",
                      "k__Bacteria.p__Proteobacteria.c__Betaproteobacteria.o__Neisseriales.f__Neisseriaceae.g__Kingella",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f___Mogibacteriaceae_.g__Mogibacterium",
                      "k__Bacteria.p__Tenericutes",
                      "k__Bacteria.p__Fusobacteria.c__Fusobacteriia.o__Fusobacteriales.f__Fusobacteriaceae.g__Fusobacterium",
                      "k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales.f___Paraprevotellaceae_",
                      "k__Bacteria.p__Bacteroidetes",
                      "k__Bacteria.p__Proteobacteria.c__Betaproteobacteria.o__Neisseriales",
                      "k__Bacteria.p__Proteobacteria.c__Gammaproteobacteria",
                      "k__Bacteria.p__Firmicutes.c__Bacilli.o__Lactobacillales.f__Carnobacteriaceae",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Veillonellaceae.g__Selenomonas",
                      "k__Bacteria.p__Firmicutes.c__Bacilli.o__Gemellales",
                      "k__Bacteria.p__Synergistetes.c__Synergistia.o__Synergistales.f__Dethiosulfovibrionaceae.g__TG5",
                      "k__Bacteria.p__Tenericutes.c__Mollicutes.o__Mycoplasmatales",
                      "k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Actinomycetales.f__Micrococcaceae.g__Rothia",
                      "k__Bacteria.p__Proteobacteria.c__Gammaproteobacteria.o__Pasteurellales.f__Pasteurellaceae",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Veillonellaceae.g__Megasphaera",
                      "k__Bacteria.p__SR1.Unclassified_member_of_SR1",
                      "k__Bacteria.p__Tenericutes.c__Mollicutes.o__Mycoplasmatales.f__Mycoplasmataceae",
                      "k__Bacteria.p__Synergistetes",
                      "k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f___Tissierellaceae_.g__Parvimonas",
                      "k__Bacteria.p__Proteobacteria.c__Alphaproteobacteria",
                      "k__Bacteria.p__Proteobacteria.c__Betaproteobacteria.o__Neisseriales.f__Neisseriaceae",
                      "k__Bacteria.p__Proteobacteria.c__Epsilonproteobacteria",
                      "k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Actinomycetales",
                      "k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Actinomycetales.f__Corynebacteriaceae",
                      "k__Bacteria.p__Actinobacteria.c__Coriobacteriia.o__Coriobacteriales.f__Coriobacteriaceae.g__Atopobium",
                      "k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Actinomycetales.f__Corynebacteriaceae.g__Corynebacterium",
                      "k__Bacteria.p__Synergistetes.c__Synergistia",
                      "k__Bacteria.p__Proteobacteria.c__Gammaproteobacteria.o__Pasteurellales.f__Pasteurellaceae.g__Haemophilus",
                      "k__Bacteria.p__Tenericutes.c__Mollicutes.o__Mycoplasmatales.f__Mycoplasmataceae.g__Mycoplasma"]

    for i, b in enumerate(lefse_bacteria):
        lefse_bacteria[i] = b.replace(".", "; ")

    all_bacteria = list(data.columns)
    chosen_bacteria = []
    for bact in all_bacteria:
        if bact in lefse_bacteria:
            chosen_bacteria.append(bact)
    print("from " + str(len(all_bacteria)) + " bacterias, " + str(len(chosen_bacteria)) +
          " were chosen using lefce analysis.\n")
    return data[chosen_bacteria]


def plot_heat_map(id_to_bacteria_map, id_to_time_map, id_to_state_map, preproccessed_data, valid_ids, bacteria, folder, title):
    data = pd.DataFrame(index=bacteria, columns=["state:0-1 time:0-6",  "state:3-4 time:0-6", "state:0-1 time:7-13", "state:3-4 time:7-13"])
    categories = {"state:0-1 time:0-6": [], "state:0-1 time:7-13": [], "state:3-4 time:0-6": [], "state:3-4 time:7-13": []}
    for id in valid_ids:
        # four groups
        category = "state:" + id_to_state_map[id] + " time:" + id_to_time_map[id].replace("_", "-")
        categories[category].append(id)
    # bulid data - y=bacteria, x=[s1_t1, s1_t2, s2_t1, s2_t2]
    for key, val_list in categories.items():
        sub_data = preproccessed_data.loc[val_list].mean(axis=0)
        data.loc[:, key] = sub_data

    short_bacterias_names, bacteria = shorten_bact_names(bacteria)
    x = [c.replace(" ", "\n") for c in data.columns]
    data = data.to_numpy()

    # plot
    fig, ax = plt.subplots(figsize=(7, 6))
    font_size = 8
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size + 2)
    ax = sns.heatmap(data, xticklabels=x, yticklabels=short_bacterias_names,
                     cmap='Blues', ax=ax)  #  annot=True, see the numbers
    ax.tick_params(axis='x', rotation=0)
    plt.title(title, fontsize=font_size + 5)

    plt.savefig(join(folder, title.replace(" ", "_") + ".svg"), bbox_inches='tight', format='svg')
    plt.show()


if __name__ == "__main__":
    tax = 6
    sub_set_bacteria_using_lefse = True
    folder = "ANOVA"
    p_val_df_file = True

    id_to_bacteria_map, id_to_time_map, id_to_state_map, preproccessed_data, valid_ids, bacteria = \
        get_organize_data_for_anova(tax)

    if sub_set_bacteria_using_lefse:
        preproccessed_data = get_sub_set_bacteria_using_lefse(preproccessed_data)
        bacteria = preproccessed_data.columns

    if not p_val_df_file:
        # run anova test for each bacteria
        print("run anova test for each bacteria")
        num_of_bact = preproccessed_data.shape[1]
        p_val_df = pd.DataFrame(columns=["time", "state", "time*state"], index=preproccessed_data.columns)
        for bact in preproccessed_data.columns:
            bact_df = pd.DataFrame(columns=["bact", "time", "state"], index=valid_ids)

            for i in valid_ids:
                bact_val = id_to_bacteria_map[i][bact]
                time = id_to_time_map[i]
                state = id_to_state_map[i]
                bact_df.loc[i] = [bact_val, time, state]

            # anova test
            bact_df["bact"] = pd.to_numeric(bact_df["bact"])
            model = ols('bact ~ C(time) + C(state) + C(time):C(state)', data=bact_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_val_df.loc[bact] = [anova_table["PR(>F)"]["C(time)"], anova_table["PR(>F)"]["C(state)"],
                                  anova_table["PR(>F)"]["C(time):C(state)"]]
            del(bact_df)

        if not sub_set_bacteria_using_lefse:
            p_val_df.to_csv(join(folder, "p_val_df_taxonomy_level_" + str(tax) + ".csv"))
        else:
            p_val_df.to_csv(join(folder, "p_val_df_lefse_taxonomy_level_" + str(tax) + ".csv"))

    else:
        if not sub_set_bacteria_using_lefse:
            p_val_df = pd.read_csv(join(folder, "p_val_df_taxonomy_level_" + str(tax) + ".csv"))
        else:
            p_val_df = pd.read_csv(join(folder, "p_val_df_lefse_taxonomy_level_" + str(tax) + ".csv"))

        # ------------------------------------------------------------------------------------------- #

    # plot hit map divided by state and time
    title = "Heat map of the amount of bacterium\nFor each state and time"
    folder = "heat maps"
    plot_heat_map(id_to_bacteria_map, id_to_time_map, id_to_state_map, preproccessed_data, valid_ids, bacteria, title=title, folder=folder)

    print("find significant bacteria")
    p_0 = 0.05
    time_significant_bact = get_significant_bact_for_col("time", p_val_df, bacteria, p_0=p_0)
    state_significant_bact = get_significant_bact_for_col("state", p_val_df, bacteria, p_0=p_0)

    # for each significant bacteria, assign type of influence
    try:
        mutual_p_values = pd.DataFrame(p_val_df["time*state"]).set_index(p_val_df['taxonomy'])
        time_p_values = pd.DataFrame(p_val_df["time"]).set_index(p_val_df['taxonomy'])
        state_p_values = pd.DataFrame(p_val_df["state"]).set_index(p_val_df['taxonomy'])
    except Exception:
        mutual_p_values = pd.DataFrame(p_val_df["time*state"]).set_index(p_val_df.index)
        time_p_values = pd.DataFrame(p_val_df["time"]).set_index(p_val_df.index)
        state_p_values = pd.DataFrame(p_val_df["state"]).set_index(p_val_df.index)

    anove_labels = ["time", "state", "time*state"]
    anove_labels_colors = ["b", "g", "r"]
    label_to_color_dict = {key: val for key, val in zip(anove_labels, anove_labels_colors)}

    significant_bact, p_values, p_values_colors =\
        get_sorted_significant_bacteria_colors(sorted(time_significant_bact + state_significant_bact),
                                               time_significant_bact, state_significant_bact,
                                               mutual_p_values, time_p_values, state_p_values)

    # plot
    plot_anove_significant_bacteria(label_to_color_dict, significant_bact, p_values, p_values_colors, p_0,
                                    taxnomy_level=tax, folder=folder)
    """
    max_idx = len(significant_bact) - 1
    for round in range(int(len(significant_bact)/12) + 1):
        start_idx = round
        end_idx = min(round + 10, max_idx)
        plot_anove_significant_bacteria(label_to_color_dict, significant_bact[start_idx:end_idx],
                                        p_values[start_idx:end_idx], p_values_colors[start_idx:end_idx], p_0,
                                        taxnomy_level=tax, folder=folder, round=round + 1)
    """
    print("!")
    """
    old 
    lefse_bacteria = ["g__Prevotella", "f__Prevotellaceae", "p_Bacteroidetes", "c__Bacteroidia", "o__Bacteroidales",
                          "f__Lactobacillaceae", "g__lactobacillus", "o__Actinomycetales", "c__Actinobacteria",
                          "f__Lachnospiraceae", "p__Actinobacteria", "f__Actinomycetaceae", "g__Actinomyces",
                          "g__Enterococcus", "f__Enterococcaceae", "p__SR1", "Unclassified_member_of_SR1",
                          "p__Proteobacteria", "p__Actinomyces", "g__Selenomonas", "c__Betaproteobacteria",
                          "g__Neisseria", "f__Neisseriaceae", "o__Neisseriales", "g__Capnocytophaga",
                          "f__Flavobacteriaceae", "o__Flavobacteriales", "c__Flavobacteriia", "f__Carnobacteriaceae",
                          "g__Granulicatella", "o__Fusobacteriales", "p__Fusobacteria", "c__Fusobacteriia",
                          "c__Alpaproteobacteria", "o__Rhizobiales", "f__Staphylococcaceae", "o__Bacillales",
                          "g__Staphylococcus"]
                              all_bacteria = data.columns
        chosen_bacteria = []
        for bact in all_bacteria:
            for lefse_bact in lefse_bacteria:
                if lefse_bact in bact:
                    chosen_bacteria.append(bact)
                    break
    
        return data[chosen_bacteria]
    """