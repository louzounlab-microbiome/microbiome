import matplotlib.pyplot as plt
import itertools
import os
from os.path import join

"""
             Iterate through all n chose two sub groups of columns in the data, and plot the relationship between them.
             colored by the categorical column inserted.
             Example-https://github.com/sharon200102/Initial_project/blob/ICA_on_the_whole_data/Graphs/All%20timepoints/Graphs%20after%20Log/Data_after_ICA_relationship_between_features_.png
             
            Parameters:
            dataframe - An ordinary DataFrame that its columns will be plotted.
            folder - A  String that represents the name of the folder where the plot will be stored (creates one if the folder doesn't exist) 
            separator - Series of discrete values which will be used to color each and every plot (Defult None).
            Title - String title for the plot (Default Generic title).
            labels_dict - A dictionary which maps every discrete color value into a string, for more informal legend (Default None).   
            figure size -  A tuple represents the plot figure size (default generic size).
            other size variables- Its possible to change x/y/legend/title sizes.
            kwargs - Parameters inserted  to each scatter plot. 
        Returns The created figure.
        
"""


def relationship_between_features(dataframe, folder, separator=None, title="Relationship between features", title_size=30,
                                  labels_dict=None, color_dict = None, figure_size=(18, 18), axis_labels_size=15, legend_size=25, **kwargs):
    number_of_columns = dataframe.shape[1]
    fig, axes = plt.subplots(number_of_columns, number_of_columns, squeeze=False, figsize=figure_size)
    if separator is not None:
        groups = dataframe.groupby(separator)
    # Iterate through all n chose two subgroups where n is the number of columns.

    for row, col in list(itertools.combinations(list(range(0, number_of_columns)), 2)):
        if separator is not None:
            for name, group in groups:
                if labels_dict is not None:
                    if color_dict is not None:
                        color = color_dict[name]
                        axes[row][col].scatter(group.iloc[:, row], group.iloc[:, col],color=color,label=labels_dict[name], **kwargs)
                    else:
                        axes[row][col].scatter(group.iloc[:, row], group.iloc[:, col],label=labels_dict[name], **kwargs)

                else:
                    if color_dict is not None:
                        color = color_dict[name]
                        axes[row][col].scatter(group.iloc[:, row], group.iloc[:, col],color = color,label=name, **kwargs)
                    else:
                        axes[row][col].scatter(group.iloc[:, row], group.iloc[:, col], label=name, **kwargs)

        else:
            axes[row][col].scatter(dataframe.iloc[:, row], dataframe.iloc[:, col], **kwargs)

            # Shape the subplot to be in a  triangular form and set titles.
        fig.delaxes(axes[col][row])
        axes[row][col].set_xlabel(dataframe.columns[row], fontsize=axis_labels_size)
        axes[row][col].set_ylabel(dataframe.columns[col],fontsize=axis_labels_size)
    for diag in range(0, number_of_columns):
        fig.delaxes(axes[diag][diag])

    # Set a main title and save the figure.
    handles, labels = axes[row][col].get_legend_handles_labels()
    fig.legend(handles, labels,fontsize=legend_size,loc = "upper left")
    fig.suptitle(title,size= title_size)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(join(folder, title.replace(" ", "_").replace("\n", "_") + ".png"))

    plt.close()
    return fig
