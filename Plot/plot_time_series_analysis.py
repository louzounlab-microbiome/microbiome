import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

"""
A class that reproduces a plot of the progress in time of all columns attribute mean.
T-test will be performed between all pairs of groups (component split by a specific timepoint->split by a binary attribute->perform ttest between the groups)
example-https://github.com/sharon200102/Initial_project/blob/ICA_on_the_whole_data/Graphs/All%20timepoints/Graphs%20after%20Log/Progress_in_time_of_column_attribute_mean_ICA.png
The constructor must receive:
    dataframe - An ordinary DataFrame (n_samples,n_components) that its columns mean progress will be plotted.
    time_series- A series object (n_samples,) that represents the time of each row in the dataframe.
Optional Arguments:
    attribute_series- Additionally to time_series, the progress of each column will be also split by an attribute_series.
    (attribute must be binary)
    Plot arguments:
        The user can edit the plot defaults by inserting
        line_styles - array of matplotlib line styles.
        markers - array of matplotlib markers.
        colors- array of matplotlib colors
        figure_size - A tuple for the figure size.
        fontsize - for the legend font size.



"""


class progress_in_time_of_column_attribute_mean(object):
    def __init__(self, dataframe:pd.DataFrame, time_series:pd.Series, attribute_series=None, **kwargs):

        self.dataframe = dataframe
        self.time_series = time_series.apply(str)
        self.attribute_series = attribute_series
        self._pvalues_matrix = None if attribute_series is None else self._create_p_values_matrix()
        self._component_mean = self._component_factorization()
        self.asterisk_matrix = self._pvalues_matrix.applymap(self._transform_p_to_marker)

        self.line_styles = kwargs.get('line_styles', ['solid', 'dashed'])
        self.markers = kwargs.get('markers', range(2, 12))
        self.colors = kwargs.get('colors', ['b', 'g', 'r', 'c', 'm'])
        figure_size = kwargs.get('figure_size', (10, 15))
        self.fig = plt.figure(figsize=figure_size)
        self.new_plot = self.fig.add_subplot(111)
        self.margin = kwargs.get('margin', 0.006)
        self.fontsize = kwargs.get('fontsize', 7)
        self.labels_dict = kwargs.get('labels_dict',None)

    """The function factorizes the dataframe into a list, in the list every component is mapped to a binary tuple.
        both arguments are lists that consist the means in all time points, each one for a different group.
        [([component0_mean_in_tp0_group0,...],[component0_mean_in_tp0_group1,...])*n_components]
    """

    def _component_factorization(self):
        all_componets = []
        component_mean_first_group = []
        component_mean_second_group = []
        for col in self.dataframe.columns:
            relevant_feature = self.dataframe[col]
            for time_point in sorted(self.time_series.unique(),key=int):
                relevant_feature_in_specific_time = relevant_feature[self.time_series == time_point]
                if self.attribute_series is None:
                    component_mean_first_group.append(relevant_feature_in_specific_time.mean())
                else:
                    attribute_first_group = relevant_feature_in_specific_time[
                        self.attribute_series == sorted(self.attribute_series.unique())[0]]
                    attribute_second_group = relevant_feature_in_specific_time[
                        self.attribute_series == sorted(self.attribute_series.unique())[1]]
                    component_mean_first_group.append(attribute_first_group.mean())
                    component_mean_second_group.append(attribute_second_group.mean())

            if not component_mean_second_group:
                all_componets.append((component_mean_first_group))
            else:
                all_componets.append((component_mean_first_group, component_mean_second_group))

            component_mean_first_group = []
            component_mean_second_group = []
        return all_componets

    """ Creates a p_value dataframe with a (n_components,unique time points) shape.
       The value in the (x,y) coordinate represents the p-value of the ttest performed on the groups of component x in time y.  
    """

    def _create_p_values_matrix(self):
        pvalues_matrix = pd.DataFrame(0.0, index=self.dataframe.columns,
                                      columns=list(map(lambda x: str(x), sorted(self.time_series.unique(),key=int))))
        for col in self.dataframe.columns:
            relevant_feature = self.dataframe[col]
            for time_point in sorted(self.time_series.unique(),key=int):
                relevant_feature_in_specific_time = relevant_feature[self.time_series == time_point]
                attribute_first_group = relevant_feature_in_specific_time[
                    self.attribute_series == sorted(self.attribute_series.unique())[0]]
                attribute_second_group = relevant_feature_in_specific_time[
                    self.attribute_series == sorted(self.attribute_series.unique())[1]]
                groups_p_val = stats.ttest_ind(attribute_first_group, attribute_second_group, equal_var=False)[1]
                pvalues_matrix.at[col, str(time_point)] = groups_p_val
        return pvalues_matrix

    """Transforms a pvalue to asterisks"""

    @staticmethod
    def _transform_p_to_marker(p_val):
        if 0.01 < p_val <= 0.05:
            return '*'
        elif 0.001 < p_val <= 0.01:
            return '**'
        elif p_val <= 0.001:
            return '***'
        else:
            return None

    """Adds the mean progress lines to the plot"""

    def _add_lines(self):
        for component, component_name, color, marker in zip(self._component_mean, self.dataframe.columns, self.colors,
                                                            self.markers):
            if self.attribute_series is None:
                for group, line_style in zip(component, self.line_styles):
                    self.new_plot.plot(sorted(self.time_series.unique(),key=int), group, color=color, marker=marker,
                                       linestyle=line_style, label=component_name)
            else:
                for group, attribute_val, line_style in zip(component, sorted(self.attribute_series.unique()),
                                                            self.line_styles):
                    if self.labels_dict is not None:
                        attribute_val = self.labels_dict[attribute_val]
                    self.new_plot.plot(sorted(self.time_series.unique(),key=int), group, color=color, marker=marker,
                                       linestyle=line_style, label="{0} {1}".format(component_name, attribute_val))

    """Adds the ttest p-value asterisks to the plot"""

    def _add_asterisks(self):
        bottom_lim, top_lim = self.new_plot.get_ylim()
        total_margin = 0
        for col in self.asterisk_matrix.columns:
            total_margin = 0
            for asterisks, color in zip(self.asterisk_matrix[col], self.colors):
                total_margin += self.margin
                if asterisks is not None:
                    self.new_plot.annotate(asterisks, (col, bottom_lim - total_margin), color=color)
        self.new_plot.set_ylim(bottom=bottom_lim - total_margin, top=top_lim + total_margin)

    def plot(self):
        self._add_lines()
        if self.attribute_series is not None:
            self._add_asterisks()
        self.new_plot.legend(fontsize=self.fontsize)
        return self.new_plot