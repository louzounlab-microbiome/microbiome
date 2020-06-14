echo off
set title=gvhd_multi_grid_tf_with_similiarity_with_censored
title %title%

FOR /L %%B IN (1,1,5) DO (
	FOR /L %%A IN (1,1,10) DO (
	call C:\Users\Bar\Miniconda3\envs\thesis_env\python gvhd_run_number_of_process_tf.py %title% 1 1
)
)
pause