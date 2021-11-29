import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import spearmanr
from Plot.plot_bacteria_intraction_network import plot_bacteria_intraction_network
from sklearn import linear_model
from regressors import stats
class bacteria_intraction_in_time(object):
    def __init__(self, dataframe, key_columns_names, time_feature_name,estimator='correlation',p_val_limit = None, binary_colors=['#ff0000', '	#0eff00'],
                 delimiter=None):
        self.dataframe = dataframe
        self.time_feature_name = time_feature_name
        self.key_columns_names = key_columns_names
        self.estimator = estimator
        self.p_val_limit = p_val_limit
        self.groups = dataframe.groupby(self.key_columns_names)
        self.relevant_columns = list(self.dataframe.drop(self.key_columns_names, axis=1).columns)
        self.delta_dataframe, self.feature_value_dataframe = self._group_features_development_in_subsequent_times()
        self.evaluation_dataframe, self.p_value_dict = self._estimator_relationship()
        self.nodes, self.edges = self._remove_unconnected_nodes(self._significant_pvals_to_edges())
        self.numeric_edges = [(self.nodes.index(x), self.nodes.index(y)) for x, y in self.edges]
        self.edges_colors = self._adjust_colors(binary_colors)
        self.nodes = self.nodes if delimiter is not None else [self._cut_node_names(name, delimiter) for name in
                                                               self.nodes]
        self.numeric_nodes = list(range(0, len(self.nodes)))

    def _subsequent_times(self, group):
        time_list = sorted(list(group[self.time_feature_name]))
        return [(current_time, next_time) for current_time, next_time in zip(time_list, time_list[1:]) if
                current_time + 1 == next_time]

    def _group_features_development_in_subsequent_times(self):
        delta_list = []
        value_list = []
        for name, group in self.groups:
            group = group.drop(self.key_columns_names, axis=1)
            for first_time, second_time in self._subsequent_times(group):
                first_row = group[group[self.time_feature_name] == first_time].squeeze()
                second_row = group[group[self.time_feature_name] == second_time].squeeze()
                delta_series = second_row.subtract(first_row)
                delta_list.append(list(delta_series))
                value_list.append(list(first_row))
        delta_df,value_df = pd.DataFrame(delta_list, columns=self.relevant_columns), pd.DataFrame(value_list,
                                                                                     columns=self.relevant_columns)
        delta_df.drop(self.time_feature_name,axis=1,inplace=True)
        value_df.drop(self.time_feature_name,axis=1,inplace=True)
        return delta_df,value_df



    def _estimator_relationship(self):
        evaluation_dataframe = pd.DataFrame(0.0, index=self.feature_value_dataframe.columns,
                                             columns=self.delta_dataframe.columns)
        p_value_dict = {}
        if self.estimator == 'correlation':
            for delta_col in self.delta_dataframe.columns:
                for feature_col in self.feature_value_dataframe.columns:
                    correlation, pvalue = spearmanr(self.feature_value_dataframe[feature_col],
                                                    self.delta_dataframe[delta_col], nan_policy='omit')
                    if pd.isnull(correlation):
                        correlation = 0
                        pvalue = 1
                    evaluation_dataframe.at[feature_col, delta_col] = correlation
                    p_value_dict[(feature_col, delta_col)] = pvalue
        elif self.estimator == 'regression':
            X = self.feature_value_dataframe.values
            for delta_col in self.delta_dataframe.columns:
                y = self.delta_dataframe[delta_col].values

                ols = linear_model.LinearRegression()
                ols.fit(X,y)
                evaluation_dataframe[delta_col] = ols.coef_

                for feature_col,p_val in zip(self.feature_value_dataframe.columns,stats.coef_pval(ols,X,y)):
                    p_value_dict[(feature_col,delta_col)] = p_val

        return evaluation_dataframe, p_value_dict

    def _significant_pvals_to_edges(self):
        if self.p_val_limit is None:
            significant_pvals = fdrcorrection(list(self.p_value_dict.values()))[0]
        else:
            significant_pvals = [p_val<=self.p_val_limit for p_val in self.p_value_dict.values()]
        edges = [edge for i, edge in enumerate(self.p_value_dict.keys()) if significant_pvals[i]]
        return edges

    def _remove_unconnected_nodes(self, significant_edges):

        nodes = []
        edges = significant_edges.copy()
        for possible_node in self.relevant_columns:
            self_edge = (possible_node, possible_node)
            edges_related_to_node = list(
                filter(lambda x: True if x[0] == possible_node or x[1] == possible_node else False, edges))
            if len(edges_related_to_node) == 1 and edges_related_to_node[0] == (self_edge):
                edges.remove(self_edge)
            elif len(edges_related_to_node) >= 1:
                nodes.append(possible_node)
        return nodes, edges

    def _adjust_colors(self, binary_colors):
        return list(
            map(lambda edge: binary_colors[0] if self.evaluation_dataframe.at[edge[0], edge[1]] > 0 else binary_colors[
                1], self.edges))

    @staticmethod
    def _cut_node_names(name, delimiter):
        return name.split(delimiter)[-1]

    def plot(self, **kwargs):
        plot_bacteria_intraction_network(bacteria=self.nodes, node_list=self.numeric_nodes,
                                         edge_list=self.numeric_edges, color_list=self.edges_colors, **kwargs)

    def export_edges_to_csv(self, name_of_file):
        edges_dataframe = pd.DataFrame(self.edges, columns=['Influential feature', 'Affected feature'])
        edges_dataframe.to_csv('{file}.csv'.format(file=name_of_file))
