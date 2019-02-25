echo off
title allergy_multi_grid_tf_with_similiarity_with_censored

FOR /L %%B IN (1,1,5) DO (
	FOR /L %%A IN (1,1,10) DO (
	call C:\Users\Bar\Miniconda3\envs\thesis_env\python run_number_of_process2.py allergy_multi_grid_tf_with_similiarity_with_censored 1 1
)
)
pause