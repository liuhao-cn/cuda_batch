! This example code will compute the inverse of many small matrices in
! batch. The speed is good for very small matrices but not good for big
! matrices.
!
! To compile: 
! nvfortran -Mcuda -Mcudalib=cublas -O3 cuda_BatchInv_SmallMatrix.f90 -o cuda_inv_small

program main

    use cudafor
    use cublas
    implicit none
    
    integer(4)                                      :: tot_start, tot_end, t_start, t_end
    integer(4)                                      :: stat, n, i, j, batch_size, lwork
    real(8)                                         :: rate
    character(len=160)                              :: cmd
    real(8), allocatable, dimension(:,:,:), pinned  :: mA
    real(8), allocatable, dimension(:,:,:), pinned  :: mB

    type(cublasHandle)                  :: cuHandle
    type(c_devptr), allocatable, device :: devPtrA_d(:), devPtrB_d(:)

    real(8),    allocatable, dimension(:,:,:), device :: mA_dev
    real(8),    allocatable, dimension(:,:,:), device :: mB_dev
    integer(4), allocatable, dimension(  :,:), device :: ipiv_dev
    integer(4), allocatable, dimension(    :), device :: info_dev

    ! read parameters from command line
    call getarg(1, cmd);  read(cmd, *) n             ! size of matrix
    call getarg(2, cmd);  read(cmd, *) batch_size     ! maximum batch length

    write (*,"(A32, I12)") "Matrix size:", n
    write (*,"(A32, I12)") "Batch size:", batch_size

    call random_seed()
    call system_clock(tot_start, rate)

    ! allocation (CPU)
    call system_clock(t_start, rate)
    allocate(mA(n, n, batch_size)); mA = 0
    allocate(mB(n, n, batch_size)); mB = 0
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Allocation of matrices, CPU:", real(t_end - t_start) / real(rate)

    ! Matrix initialization (CPU)
    call system_clock(t_start, rate)
    call random_number(mA)
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Matrix initialization (CPU):", real(t_end - t_start) / real(rate)

    ! Allocation, GPU
    call system_clock(t_start, rate)
    allocate( mA_dev    (n, n, batch_size) ); mA_dev   = 0
    allocate( mB_dev    (n, n, batch_size) ); mB_dev   = 0
    allocate( info_dev  (      batch_size) ); info_dev = 0
    stat = cudaDeviceSynchronize()
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Allocation, GPU:", real(t_end - t_start) / real(rate)

    ! copy matrix from CPU to GPU
    call system_clock(t_start, rate)
    mA_dev = mA
    stat = cudaDeviceSynchronize()
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Data to device (all matrices):", real(t_end - t_start) / real(rate)

    ! make cuHandle, which includes the GPU overhead
    call system_clock(t_start, rate)
    stat = cublasCreate(cuHandle)
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "--Overhead--:", real(t_end - t_start) / real(rate)

    ! allocate the pivot space, GPU
    call system_clock(t_start, rate)
    allocate( ipiv_dev(n, batch_size) ); ipiv_dev = 0
    stat = cudaDeviceSynchronize()
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Allocate workspace:", real(t_end - t_start) / real(rate)

    ! make the c-device pointer
    call system_clock(t_start, rate)
    allocate( devPtrA_d(batch_size) )
    allocate( devPtrB_d(batch_size) )
    do i=1, batch_size
        devPtrA_d(i) = c_devloc(mA_dev(1,1,i))
        devPtrB_d(i) = c_devloc(mB_dev(1,1,i))
    enddo
    stat = cudaDeviceSynchronize()
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Making c-pointer:", real(t_end - t_start) / real(rate)

    ! run the batch
    call system_clock(t_start, rate)
    stat = cublasDgetrfBatched(cuHandle, n, devPtrA_d, n, ipiv_dev(:,1), info_dev, batch_size)
    stat = cublasDgetriBatched(cuHandle, n, devPtrA_d, n, ipiv_dev(:,1), devPtrB_d, n, info_dev, batch_size)
    stat = cudaDeviceSynchronize()
    call system_clock(t_end, rate)
    write (*,"(A32, F12.3)") "--GPU compute time cost--", real(t_end - t_start) / real(rate)

    ! copy the results from GPU to CPU
    call system_clock(t_start, rate)
    mB = mB_dev
    stat = cudaDeviceSynchronize()
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Device to data (1 matrix):", real(t_end - t_start) / real(rate)

    ! compute the total time cost
    call system_clock(tot_end, rate)
    write (*,"(A32, F12.3)") "Total time cost:", real(tot_end - tot_start) / real(rate)

    ! print the sum of results
    write (*,"(A32, F12.3)") "Sum of results:", sum(mB)

end

