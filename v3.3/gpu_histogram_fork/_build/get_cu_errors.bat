@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\C\Documents\Savage22 Server\v3.3\gpu_histogram_fork\_build\LightGBM\build_link"
ninja -j1 CMakeFiles/lightgbm_objs.dir/src/treelearner/cuda_sparse_hist_tree_learner.cu.obj 2>&1
