! This example code will compute the inverse of many big matrices in
! batch. The speed is good for big matrices but not good for very small
! matrices.
!
! To compile: 
! nvfortran -Mcuda -Mcudalib=cublas,cusolver cuda_BatchInv_BigMatrix.f90 -o cuda_inv_big

module cuBatchVar
    use cublas_v2
    use cudafor
    use cusolverDn
    implicit none

    type(cusolverDnHandle)        , allocatable  :: cuDnHandle(:)
    integer(kind=cuda_stream_kind), allocatable  :: stream(:)

    real(8),    allocatable, dimension(:,:,:), device :: mA_dev, mB_dev, mC_dev
    real(8),    allocatable, dimension(  :,:), device :: work_dev
    integer(4), allocatable, dimension(  :,:), device :: ipiv_dev
    integer(4), allocatable, dimension(    :), device :: info_dev
end module cuBatchVar

program test_batch

    use cublas_v2
    use cudafor
    use cusolverDn
    use cuBatchVar
    implicit none

    real(8),    allocatable, dimension(:,:,:), pinned   :: mA, mB, mC    
    integer(4)                                          :: t_start, t_end, s_start, s_end
    integer(4)                                          :: stat, n, i, j, batch_size, lwork
    real(8)                                             :: rate
    character(len=160)                                  :: cmd

    ! read parameters from command line
    call getarg(1, cmd);  read(cmd, *) n             ! size of matrix
    call getarg(2, cmd);  read(cmd, *) batch_size    ! maximum batch length

    write (*,"(A32, I12)") "Matrix size:", n
    write (*,"(A32, I12)") "Batch size:", batch_size

    call random_seed()
    call system_clock(t_start, rate)

    ! allocation (CPU)
    call system_clock(s_start, rate)
    allocate(mA(n, n, batch_size)); mA = 0
    allocate(mB(n, n, batch_size)); mB = 0
    call system_clock(s_end)
    write (*,"(A32, F12.3)") "Allocation of matrices, CPU:", real(s_end - s_start) / real(rate)

    ! Matrix initialization, especially, B-matrix should be made to identity matrix (CPU)
    call system_clock(s_start, rate)
    call random_number(mA)
    do i=1,batch_size
        do j=1,n
            mB(j,j,i) = 1
        enddo
    enddo
    call system_clock(s_end)
    write (*,"(A32, F12.3)") "Matrix initialization (CPU):", real(s_end - s_start) / real(rate)

    ! Allocation, GPU
    call system_clock(s_start, rate)
    allocate( mA_dev    (n, n, batch_size) ); mA_dev   = 0
    allocate( mB_dev    (n, n, batch_size) ); mB_dev   = 0
    allocate( ipiv_dev  (   n, batch_size) ); ipiv_dev = 0
    allocate( info_dev(        batch_size) ); info_dev = 0
    stat = cudaDeviceSynchronize()
    call system_clock(s_end)
    write (*,"(A32, F12.3)") "Allocation, GPU:", real(s_end - s_start) / real(rate)

    ! copy matrix from CPU to GPU
    call system_clock(s_start, rate)
    mA_dev = mA
    mB_dev = mB
    stat = cudaDeviceSynchronize()
    call system_clock(s_end)
    write (*,"(A32, F12.3)") "Data to device (all matrices):", real(s_end - s_start) / real(rate)

    ! make cuDnHandle, which includes the GPU overhead
    call system_clock(s_start, rate)
    allocate( cuDnHandle(batch_size) )
    allocate(     stream(batch_size) )
    stat = cusolverDnCreate(cuDnHandle(1))
    call system_clock(s_end)
    write (*,"(A32, F12.3)") "--Overhead--:", real(s_end - s_start) / real(rate)

    ! allocate the eigen problem workspace, GPU
    call system_clock(s_start, rate)
    stat = cusolverDnDgetrf_bufferSize(cuDnHandle(1), n, n, mA_dev(:,:,1), n, lwork)
    write (*,"(A32, I12)") "Size of workspace (MB):", lwork/1024**2
    allocate( work_dev(lwork, batch_size) ); work_dev = 0
    stat = cusolverDnDestroy(cuDnHandle(1))
    stat = cudaDeviceSynchronize()
    call system_clock(s_end)
    write (*,"(A32, F12.3)") "Allocate workspace:", real(s_end - s_start) / real(rate)

    ! run the batch
    call system_clock(s_start, rate)
    do i=1, batch_size
        stat = cusolverDnCreate(cuDnHandle(i))
        stat = cudaStreamCreateWithFlags(stream(i), cudaStreamNonBlocking)
        stat = cusolverDnSetStream(cuDnHandle(i), stream(i))

        stat = cusolverDnDgetrf(cuDnHandle(i), n, n, mA_dev(:,:,i), n, work_dev(:,i), ipiv_dev(:,i), info_dev(i))
        stat = cusolverDnDgetrs(cuDnHandle(i), CUBLAS_OP_N, n, n, mA_dev(:,:,i), n, ipiv_dev(:,i), mB_dev(:,:,i), n, info_dev(i))
    enddo
    stat = cudaDeviceSynchronize()
    call system_clock(s_end, rate)
    write (*,"(A32, F12.3)") "--GPU compute time cost--", real(s_end - s_start) / real(rate)

    ! copy the results from GPU to CPU
    call system_clock(s_start, rate)
    mB = mB_dev
    stat = cudaDeviceSynchronize()
    call system_clock(s_end)
    write (*,"(A32, F12.3)") "Device to data (1 matrix):", real(s_end - s_start) / real(rate)

    ! Clean up
    call system_clock(s_start, rate)
    do i=1, batch_size
        stat = cusolverDnDestroy(cuDnHandle(i))
    enddo
    do i=1, batch_size
        stat = cudaStreamDestroy(stream(i))
    enddo
    stat = cudaDeviceSynchronize()
    call system_clock(s_end, rate)
    write (*,"(A32, F12.3)") "Clean up:", real(s_end - s_start) / real(rate)

    ! compute the total time cost
    call system_clock(t_end, rate)
    write (*,"(A32, F12.3)") "Total time cost:", real(t_end - t_start) / real(rate)

    ! print the sum of results
    write (*,"(A32, F12.3)") "Sum of results:", sum(mB)

end

