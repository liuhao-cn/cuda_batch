! To compile: 
! nvfortran -Mcuda -Mcudalib=cublas,cusolver cuda_BatchEigen.f90 -o cuda_BatchEigen

program test_batch

    use cublas_v2
    use cudafor
    use cusolverDn
    implicit none
    
    type(cusolverDnHandle)          :: cuDnHandle
    integer(kind=cuda_stream_kind)  :: stream
    integer(4)                      :: tot_start, tot_end, t_start, t_end
    integer(4)                      :: stat, n, i, j, batch_size, lwork
    real(8)                         :: rate
    character(len=160)              :: cmd

    real(8),    allocatable, dimension(:,:,:), pinned :: mA
    real(8),    allocatable, dimension(  :,:), pinned :: vA

    real(8),    allocatable, dimension(:,:,:), device :: mA_dev
    real(8),    allocatable, dimension(  :,:), device :: vA_dev
    real(8),    allocatable, dimension(  :,:), device :: work_dev
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
    allocate(vA(   n, batch_size)); vA = 0
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Allocation of matrices, CPU:", real(t_end - t_start) / real(rate)

    ! Matrix initialization, including making the matrix symmetric (CPU)
    call system_clock(t_start, rate)
    call random_number(mA)
    do i=1,batch_size
        do j=1,n-1
            mA(j+1:n,j,i) = mA(j,j+1:n,i)
        enddo
    enddo
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Matrix initialization (CPU):", real(t_end - t_start) / real(rate)

    ! Allocation, GPU
    call system_clock(t_start, rate)
    allocate( mA_dev    (n, n, batch_size) ); mA_dev   = 0
    allocate( vA_dev    (   n, batch_size) ); vA_dev   = 0
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

    ! make cuDnHandle, which includes the GPU overhead
    call system_clock(t_start, rate)
    stat = cusolverDnCreate(cuDnHandle)
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "---Overhead---:", real(t_end - t_start) / real(rate)

    ! allocate the eigen problem workspace, GPU
    call system_clock(t_start, rate)
    stat = cusolverDnDsyevd_buffersize(cuDnHandle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, mA_dev(:,:,1), n, vA_dev(:,1), lwork)
    write (*,"(A32, I12)") "Size of workspace (MB):", lwork/1024**2
    allocate( work_dev(lwork, batch_size) ); work_dev = 0
    stat = cudaDeviceSynchronize()
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Allocate workspace:", real(t_end - t_start) / real(rate)

    ! run the batch
    call system_clock(t_start, rate)
    do i=1, batch_size
        stat = cusolverDnCreate(cuDnHandle)
        stat = cudaStreamCreate(stream)
        stat = cusolverDnSetStream(cuDnHandle, stream)

        stat = cusolverDnDsyevd(cuDnHandle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, mA_dev(:,:,i), &
            n, vA_dev(:,i), work_dev(:,i), lwork, info_dev(i))
    enddo
    stat = cudaDeviceSynchronize()
    call system_clock(t_end, rate)
    write (*,"(A32, F12.3)") "GPU compute time cost:", real(t_end - t_start) / real(rate)

    ! copy the results from GPU to CPU
    call system_clock(t_start, rate)
    mA = mA_dev
    stat = cudaDeviceSynchronize()
    call system_clock(t_end)
    write (*,"(A32, F12.3)") "Device to data (1 matrix):", real(t_end - t_start) / real(rate)

    ! compute the total time cost
    call system_clock(tot_end, rate)
    write (*,"(A32, F12.3)") "Total time cost:", real(tot_end - tot_start) / real(rate)

    ! print the sum of results
    write (*,"(A32, F12.3)") "Sum of results:", sum(mA)

end

