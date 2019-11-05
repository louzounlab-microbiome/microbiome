import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy


def print_confusion_matrix(confusion_matrix, class_names, acc, algorithm, title, folder, figsize=(10, 7), fontsize=17):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 20})  # , fmt="d"
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    ac = str(round(acc, 3))
    plt.title(title.capitalize() + "\n" + algorithm + " Confusion Matrix Heat Map\n" + "Accuracy = " + ac, fontsize=17)
    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    # plt.show()
    plt.savefig(os.path.join(folder, title + "_" + algorithm + "_confusion_matrix_heat_map_" + ac + ".svg"),
                bbox_inches='tight', format='svg')
    return fig


def edit_confusion_matrix(title, confusion_matrixes, data_loader, algo, names, BINARY=False):
    if BINARY:
        if algo == "NN":
            c = confusion_matrixes.tolist()
            x1 = c[0][0]
            x2 = c[0][1]
            x3 = c[1][0]
            x4 = c[1][1]
        else:
            x1 = np.mean([c[0][0] for c in list(confusion_matrixes)])
            x2 = np.mean([c[0][1] for c in list(confusion_matrixes)])
            x3 = np.mean([c[1][0] for c in list(confusion_matrixes)])
            x4 = np.mean([c[1][1] for c in list(confusion_matrixes)])

        # calc_acc
        sum = x1 + x2 + x3 + x4
        x1 = x1 / sum
        x2 = x2 / sum
        x3 = x3 / sum
        x4 = x4 / sum
        acc = x1 + x4
        print("acc = " + str(acc))
        mat = [[x1, x2], [x3, x4]]
        mat.append([acc])
        df = pd.DataFrame(data=mat)
        df.columns = names
        df.index = names + ["acc"]
        confusion_matrix_average = df  # "[[" + str(x1) + ", " + str(x2) + "], [" + str(x3) + ", " + str(x4) + "]]"

        # random classification and acc calculation in order to validate the results.
        #TODO

        return confusion_matrix_average, acc

    else:  # MULTI CLASS
        if algo == "NN":
            sum = 0
            c = confusion_matrixes.tolist()
            for row in c:
                for num in row:
                    sum += num
            for i in range(len(c)):
                for j in range(len(c)):
                    c[i][j] = c[i][j] / sum

            acc = 0
            for i in range(len(c)):
                acc = acc + c[i][i]
            reg_df = pd.DataFrame(c)
            reg_df.columns = names
            reg_df.index = names
            print("acc = " + str(acc))

            types_w_acc = names + ["acc"]
            c.append([str(acc)])
            df = pd.DataFrame(c)
            df.columns = names
            df.index = types_w_acc

            return reg_df, acc

        else:
            final_matrix = []
            matrix = list(copy.deepcopy(confusion_matrixes[0]))
            for l in matrix:
                final_matrix.append(list(l))

            # set a final empty matrix
            for i in range(len(final_matrix)):
                for j in range(len(final_matrix)):
                    final_matrix[i][j] = 0
            # fill it with the sum of all matrixes
            for mat in confusion_matrixes:
                for i in range(len(final_matrix)):
                    for j in range(len(final_matrix)):
                        final_matrix[i][j] = final_matrix[i][j] + mat[i][j]
            # divide to get the avarege
            for i in range(len(final_matrix)):
                for j in range(len(final_matrix)):
                    final_matrix[i][j] = float(final_matrix[i][j]) / float(len(confusion_matrixes))

            # calc_acc
            sum = 0
            for i in range(len(final_matrix)):
                for j in range(len(final_matrix)):
                    sum = sum + final_matrix[i][j]

            reg_final_matrix = copy.deepcopy(final_matrix)
            for i in range(len(final_matrix)):
                for j in range(len(final_matrix)):
                    reg_final_matrix[i][j] = reg_final_matrix[i][j] / sum


            acc = 0
            for i in range(len(final_matrix)):
                acc = acc + reg_final_matrix[i][i]

            types_w_acc = names + ["acc"]
            final_matrix.append([str(acc)])
            df = pd.DataFrame(final_matrix)
            df.columns = names
            df.index = types_w_acc

            # reg_final_matrix.append([str(acc)])
            reg_df = pd.DataFrame(reg_final_matrix)
            reg_df.columns = names
            reg_df.index = names
            print("acc = " + str(acc))
            return reg_df, acc

        return None



"""
def print_confusion_matrix(confusion_matrix, class_names, acc, algorithm, TITLE, figsize=(10, 7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 20})  # , fmt="d"
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    ac = str(round(acc, 3))
    plt.title(TITLE.capitalize() + "\n" + algorithm + " Confusion Matrix Heat Map\n" + "Accuracy = " + ac, fontsize=17)
    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    # plt.show()
    plt.savefig(os.path.join(TITLE, TITLE + "_" + algorithm + "_confusion_matrix_heat_map_" + ac + ".svg"),
                bbox_inches='tight', format='svg')
    return fig

def edit_confusion_matrix(title, confusion_matrixes, data_loader, algo, names):
    if title in ["Success_task",  "Health_task", "Prognostic_task", "Milk_allergy_task"]:
        if title == "Milk_allergy_task":
            names = ['Other', 'Milk']
        elif title == "Health_task":
            names = ['Allergic', 'Healthy']
        elif title in ["Success_task", "Prognostic_task"]:
            names = ['No', 'Yes']

        if algo == "NN":
            c = confusion_matrixes.tolist()
            x1 = c[0][0]
            x2 = c[0][1]
            x3 = c[1][0]
            x4 = c[1][1]
        else:
            x1 = np.mean([c[0][0] for c in list(confusion_matrixes)])
            x2 = np.mean([c[0][1] for c in list(confusion_matrixes)])
            x3 = np.mean([c[1][0] for c in list(confusion_matrixes)])
            x4 = np.mean([c[1][1] for c in list(confusion_matrixes)])

        # calc_acc
        sum = x1 + x2 + x3 + x4
        x1 = x1 / sum
        x2 = x2 / sum
        x3 = x3 / sum
        x4 = x4 / sum
        acc = x1 + x4
        print("acc = " + str(acc))
        mat = [[x1, x2], [x3, x4]]
        mat.append([acc])
        df = pd.DataFrame(data=mat)
        df.columns = names
        df.index = names + ["acc"]
        confusion_matrix_average = df  # "[[" + str(x1) + ", " + str(x2) + "], [" + str(x3) + ", " + str(x4) + "]]"

        # random classification and acc calculation in order to validate the results.
        #TODO

        return confusion_matrix_average, acc, names

    elif title in ["Allergy_type_task"]:
        tag_to_allergy_type_map = data_loader.get_tag_to_allergy_type_map
        allergy_type_to_instances_map = data_loader.get_allergy_type_to_instances_map
        allergy_type_to_weight_map = data_loader.get_allergy_type_to_weight_map
        allergy_type_weights = list(allergy_type_to_weight_map.values())
        types = []
        for key in range(len(tag_to_allergy_type_map.keys())):
            types.append(tag_to_allergy_type_map.get(key))

        if algo == "NN":
            sum = 0
            c = confusion_matrixes.tolist()
            for row in c:
                for num in row:
                    sum += num
            for i in range(len(c)):
                for j in range(len(c)):
                    c[i][j] = c[i][j] / sum

            acc = 0
            for i in range(len(c)):
                acc = acc + c[i][i]
            reg_df = pd.DataFrame(c)
            reg_df.columns = types
            reg_df.index = types
            print("acc = " + str(acc))

            types_w_acc = types + ["acc"]
            c.append([str(acc)])
            df = pd.DataFrame(c)
            df.columns = types
            df.index = types_w_acc

            return reg_df, acc, types

        else:
            final_matrix = []
            matrix = list(copy.deepcopy(confusion_matrixes[0]))
            for l in matrix:
                final_matrix.append(list(l))

            # set a final empty matrix
            for i in range(len(final_matrix)):
                for j in range(len(final_matrix)):
                    final_matrix[i][j] = 0
            # fill it with the sum of all matrixes
            for mat in confusion_matrixes:
                for i in range(len(final_matrix)):
                    for j in range(len(final_matrix)):
                        final_matrix[i][j] = final_matrix[i][j] + mat[i][j]
            # divide to get the avarege
            for i in range(len(final_matrix)):
                for j in range(len(final_matrix)):
                    final_matrix[i][j] = float(final_matrix[i][j]) / float(len(confusion_matrixes))

            # calc_acc
            sum = 0
            for i in range(len(final_matrix)):
                for j in range(len(final_matrix)):
                    sum = sum + final_matrix[i][j]

            reg_final_matrix = copy.deepcopy(final_matrix)
            for i in range(len(final_matrix)):
                for j in range(len(final_matrix)):
                    reg_final_matrix[i][j] = reg_final_matrix[i][j] / sum


            acc = 0
            for i in range(len(final_matrix)):
                acc = acc + reg_final_matrix[i][i]

            types_w_acc = types + ["acc"]
            final_matrix.append([str(acc)])
            df = pd.DataFrame(final_matrix)
            df.columns = types
            df.index = types_w_acc

            # reg_final_matrix.append([str(acc)])
            reg_df = pd.DataFrame(reg_final_matrix)
            reg_df.columns = types
            reg_df.index = types
            print("acc = " + str(acc))
            return reg_df, acc, types
        return None

"""