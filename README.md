# YOLO Lab - Microbiome Research.
We perform multiple microbiome computational projects in collaborations with the Koren lab in Tzfat, and other labs. We develop diagnostic and prognostic tools based on a combination of microbiome and machine learning, as well as advanced methods to predict microbiome dynamics. These projects combine advanced machine learning with the complex dynamics of the microbiome.
### Microbiome Preprocess 
TEXT HERE
### Visualization for microbiome analysis.
We developed several plots that may facilitate in understanding the complicated structure of the microbiome and its connections to different phenomena.
#### Plot/plot_relationship_between_features.py
After the projection of the microbiome to a lower dimension denoted by n, iterate through all n chose two sub groups of columns in the data, and plot the relationship between them.colored by a categorical column inserted.
</p>
For example:
</p>

![Relationship_between_features](https://user-images.githubusercontent.com/28387079/116859587-e19fe380-ac08-11eb-9c7d-8fdf20a9faac.png)
</p>

#### Plot/plot_time_series_analysis.py
After the projection of the microbiome to a lower dimension, the following function aims to visualize the differences between two groups (b.g Normal vs Cancer)
for each feature separately through time.
T-test will be performed between all pairs of groups (component split by a specific timepoint->split by a binary attribute->perform ttest between the groups)
![progress_of_components_in_time](https://user-images.githubusercontent.com/28387079/116861277-86232500-ac0b-11eb-85aa-5ca83fe8ed83.png)
