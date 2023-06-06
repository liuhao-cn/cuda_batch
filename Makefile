# nvfortran -Mcuda -Mcudalib=cublas,cusolver cuda_BatchInv_BigMatrix.f90 -o cuda_inv_big
fc 			= nvfortran
fc_flag 	= -O3 -Mcuda
cuLib		= cublas,cusolver
dir_out		= ./

# note that "make_plm" requires HEALPix 3.5 because it was modified especially for make_plm
# all: make_plm
all: big small eigen

big:
	$(fc) $(fc_flag) -Mcudalib=$(cuLib) cuda_BatchInv_BigMatrix.f90 -o $(dir_out)inv_big

small:
	$(fc) $(fc_flag) -Mcudalib=$(cuLib) cuda_BatchInv_SmallMatrix.f90 -o $(dir_out)inv_small			

eigen:
	$(fc) $(fc_flag) -Mcudalib=$(cuLib) cuda_BatchEigen.f90 -o $(dir_out)batch_eigen
