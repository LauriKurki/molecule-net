
subroutine peak_dist(nb, nx, ny, nz, dist, N_atom, pos, xyz_start, xyz_step, std)
    real(8), parameter :: pi = 3.141592653589793d0
    integer, parameter :: CUTOFF = 4

    ! Input arguments
    integer, intent(in) :: nb, nx, ny, nz
    real(8) :: dist(:)
    integer, intent(in) :: N_atom(:)
    real(8), intent(in) :: pos(:)
    real(8), intent(in) :: xyz_start(3), xyz_step(3), std

    ! Local variables
    integer :: i, j, k, b, ip, ip0, ind, nxyz
    real(8) :: cov_inv, cutoff2, denom, prefactor
    real(8) :: grid_xyz(3), atom_xyz(3), dp(3), dp2, v

    ! Precompute constants
    nxyz = nx * ny * nz
    cov_inv = 1.0d0 / (std * std)
    cutoff2 = CUTOFF * CUTOFF * std * std
    denom = sqrt(2.0d0 * pi) * std
    prefactor = 1.0d0 / (denom * denom * denom)

    ind = 1
    do i = 1, nx
    do j = 1, ny
        do k = 1, nz
        grid_xyz(1) = xyz_start(1) + (i - 1) * xyz_step(1)
        grid_xyz(2) = xyz_start(2) + (j - 1) * xyz_step(2)
        grid_xyz(3) = xyz_start(3) + (k - 1) * xyz_step(3)

        ip0 = 0
        do b = 1, nb
            v = 0.0d0
            do ip = 1, N_atom(b)
            atom_xyz(1) = pos(3 * (ip0 + ip) - 2)
            atom_xyz(2) = pos(3 * (ip0 + ip) - 1)
            atom_xyz(3) = pos(3 * (ip0 + ip))

            dp(1) = grid_xyz(1) - atom_xyz(1)
            dp(2) = grid_xyz(2) - atom_xyz(2)
            dp(3) = grid_xyz(3) - atom_xyz(3)
            dp2 = dp(1)**2 + dp(2)**2 + dp(3)**2

            if (dp2 < cutoff2) then
                v = v + exp(-0.5d0 * dp2 * cov_inv)
            end if
            end do
            v = v * prefactor
            ip0 = ip0 + N_atom(b)
            dist((b - 1) * nxyz + ind) = v
        end do
        ind = ind + 1
        end do
    end do
    end do
end subroutine peak_dist
