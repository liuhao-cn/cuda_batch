# nvfortran -Mcuda -Mcudalib=cublas,cusolver cuda_BatchInv_BigMatrix.f90 -o cuda_inv_big
fc          = nvfortran
fc_flag     = -O3 -Mcuda
cuLib       = cublas,cusolver
dir_src     = ./src
dir_out     = ./run
rm_target   = cubatchvar.mod

# note that "make_plm" requires HEALPix 3.5 because it was modified especially for make_plm
# all: make_plm
all: c1 c2 c3 cleanup

big: c1 cleanup

small: c2 cleanup

eigen: c3 cleanup


c1:
    $(fc) $(fc_flag) -Mcudalib=$(cuLib) $(dir_src)/cuda_BatchInv_BigMatrix.f90 -o $(dir_out)/inv_big

c2:
    $(fc) $(fc_flag) -Mcudalib=$(cuLib) $(dir_src)/cuda_BatchInv_SmallMatrix.f90 -o $(dir_out)/inv_small    

c3:
    $(fc) $(fc_flag) -Mcudalib=$(cuLib) $(dir_src)/cuda_BatchEigen.f90 -o $(dir_out)/batch_eigen

cleanup:
    rm $(rm_target)