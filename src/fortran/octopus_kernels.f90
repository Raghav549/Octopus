! Octopus Hybrid AI Engine -- Fortran 2018/2023 reference kernels.
!
! This file is the optional "Fortran DNA" backend. It is compiled only when
! OCT_WITH_FORTRAN=ON and a Fortran compiler is present; every call site in
! src/numerics falls back to the C++ long-double reference implementation when
! this library is not linked, and reports the fallback as degraded in the
! result's `backend` string (never silently).
!
! Build status of this file in the current environment: NOT COMPILED. There is
! no Fortran compiler in the container this repository was assembled in, so
! oct_f_* has never been linked or executed here. The first-order schemes below
! are written to be independently checkable against the C++ reference by
! tests/numerics.sod1d_fortran_matches_cxx_when_available, which SKIPS with a
! printed reason when the backend is absent. Do not read a green test suite as
! evidence that this file runs -- see docs/LIMITATIONS.md.
!
! SPDX-License-Identifier: MIT

module octopus_fortran_kernels
    use, intrinsic :: iso_c_binding, only: c_int, c_double, c_char, c_null_char
    implicit none

    real(c_double), parameter :: PI = 3.14159265358979323846264338327950288_c_double

contains

    ! Return code convention used by every kernel below:
    !   0 = success, 1 = invalid argument, 2 = numerical failure.
    pure function ok_code(finite_ok) result(rc)
        logical, intent(in) :: finite_ok
        integer(c_int) :: rc
        if (finite_ok) then
            rc = 0_c_int
        else
            rc = 2_c_int
        end if
    end function ok_code

    ! -----------------------------------------------------------------------
    ! Sod shock tube: first-order finite volume with Rusanov (local Lax-
    ! Friedrichs) fluxes and forward Euler. n_cells interior cells; the initial
    ! discontinuity sits at x = 0 in [-1, 1].
    ! -----------------------------------------------------------------------
    function oct_f_sod1d(gamma, t_end, n_cells, cfl, rho_l, u_l, p_l, rho_r, u_r, p_r, &
                         out_rho, out_u, out_p, n_out) bind(C, name="oct_f_sod1d") result(rc)
        real(c_double), value, intent(in) :: gamma, t_end, cfl
        real(c_double), value, intent(in) :: rho_l, u_l, p_l, rho_r, u_r, p_r
        integer(c_int), value, intent(in) :: n_cells
        real(c_double), intent(out) :: out_rho(*), out_u(*), out_p(*)
        integer(c_int), intent(out) :: n_out
        integer(c_int) :: rc

        real(c_double), allocatable :: rho(:), mom(:), ene(:), frho(:), fmom(:), fene(:)
        real(c_double) :: xc, dx, cs, dxdt, amax, dt, t, rl, ul, pl, rr, ur, pr
        real(c_double) :: el, er, fl(3), fr(3), sl, sr, gl, gr, ssum
        integer(c_int) :: n, i, step, max_steps

        rc = 0_c_int
        n_out = 0_c_int
        if (n_cells < 2 .or. gamma <= 1.0_c_double .or. t_end <= 0.0_c_double) then
            rc = 1_c_int
            return
        end if

        n = n_cells
        dx = 2.0_c_double / real(n, c_double)

        allocate(rho(n), mom(n), ene(n), frho(0:n+1), fmom(0:n+1), fene(0:n+1))
        rho = rho_r
        mom = rho_r * u_r
        ene = p_r / (gamma - 1.0_c_double) + 0.5_c_double * rho_r * u_r * u_r
        do i = 1, n
            xc = -1.0_c_double + (real(i, c_double) - 0.5_c_double) * dx
            if (xc < 0.0_c_double) then
                rho(i) = rho_l
                mom(i) = rho_l * u_l
                ene(i) = p_l / (gamma - 1.0_c_double) + 0.5_c_double * rho_l * u_l * u_l
            end if
        end do

        t = 0.0_c_double
        max_steps = 1000000
        do step = 1, max_steps
            if (t >= t_end) exit
            ! CFL from the current state.
            amax = 0.0_c_double
            do i = 1, n
                cs = sqrt(max(gamma * max(pr_from(ene(i), mom(i), rho(i), gamma), 1.0e-300_c_double) / rho(i), 0.0_c_double))
                amax = max(amax, abs(mom(i) / rho(i)) + cs)
            end do
            if (amax <= 0.0_c_double) exit
            dt = cfl * dx / amax
            if (t + dt > t_end) dt = t_end - t
            if (dt <= 0.0_c_double) exit

            do i = 0, n + 1
                if (i == 0) then
                    rl = rho(1); ul = mom(1) / rho(1); pl = pr_from(ene(1), mom(1), rho(1), gamma)
                    rr = rl; ur = -ul; pr = pl                       ! reflective wall
                else if (i == n + 1) then
                    rl = rho(n); ul = mom(n) / rho(n); pl = pr_from(ene(n), mom(n), rho(n), gamma)
                    rr = rl; ur = -ul; pr = pl                       ! reflective wall
                else
                    rl = rho(i); ul = mom(i) / rho(i); pl = pr_from(ene(i), mom(i), rho(i), gamma)
                    rr = rho(i); ur = ul; pr = pl
                end if
                call physical_flux(gamma, rl, ul, pl, fl)
                call physical_flux(gamma, rr, ur, pr, fr)
                sl = abs(ul) + sqrt(max(gamma * pl / rl, 0.0_c_double))
                sr = abs(ur) + sqrt(max(gamma * pr / rr, 0.0_c_double))
                ssum = max(sl, sr)
                frho(i) = 0.5_c_double * (fl(1) + fr(1)) - 0.5_c_double * ssum * (rl - rr)
                fmom(i) = 0.5_c_double * (fl(2) + fr(2)) - 0.5_c_double * ssum * (rl * ul - rr * ur)
                fene(i) = 0.5_c_double * (fl(3) + fr(3)) - 0.5_c_double * ssum * &
                          ((ene(i) + pl) - (ene(max(i, 1)) + pr))
            end do

            do i = 1, n
                rho(i) = rho(i) - dt / dx * (frho(i) - frho(i - 1))
                mom(i) = mom(i) - dt / dx * (fmom(i) - fmom(i - 1))
                ene(i) = ene(i) - dt / dx * (fene(i) - fene(i - 1))
                if (rho(i) <= 0.0_c_double) then
                    rc = 2_c_int
                    return
                end if
            end do
            t = t + dt
        end do

        do i = 1, n
            out_rho(i) = rho(i)
            out_u(i) = mom(i) / rho(i)
            out_p(i) = pr_from(ene(i), mom(i), rho(i), gamma)
        end do
        n_out = n
        if (.not. all_finite(out_rho(1:n)) .or. .not. all_finite(out_u(1:n))) rc = 2_c_int
    end function oct_f_sod1d

    pure function pr_from(ene, mom, rho, gamma) result(p)
        real(c_double), intent(in) :: ene, mom, rho, gamma
        real(c_double) :: p
        p = (gamma - 1.0_c_double) * (ene - 0.5_c_double * mom * mom / rho)
    end function pr_from

    pure subroutine physical_flux(gamma, rho, u, p, f)
        real(c_double), intent(in) :: gamma, rho, u, p
        real(c_double), intent(out) :: f(3)
        real(c_double) :: e
        e = p / (gamma - 1.0_c_double) + 0.5_c_double * rho * u * u
        f(1) = rho * u
        f(2) = rho * u * u + p
        f(3) = (e + p) * u
    end subroutine physical_flux

    pure function all_finite(a) result(ok)
        real(c_double), intent(in) :: a(:)
        logical :: ok
        integer :: i
        ok = .true.
        do i = 1, size(a)
            if (a(i) /= a(i) .or. abs(a(i)) > huge(1.0_c_double) / 4.0_c_double) ok = .false.
        end do
    end function all_finite

    ! -----------------------------------------------------------------------
    ! 2-D heat equation: explicit FTCS with ghost-cell Dirichlet walls.
    ! out_u holds the n x n cell-centred solution, row-major (last index
    ! fastest, matching the C++ layout: index = i * n + j).
    ! -----------------------------------------------------------------------
    function oct_f_heat2d(alpha, t_end, n, cfl, out_u) bind(C, name="oct_f_heat2d") result(rc)
        real(c_double), value, intent(in) :: alpha, t_end, cfl
        integer(c_int), value, intent(in) :: n
        real(c_double), intent(out) :: out_u(*)
        real(c_double), allocatable :: u(:), un(:)
        real(c_double) :: h, dt, t, lap
        integer(c_int) :: i, j, idx, step, max_steps
        integer(c_int) :: rc

        rc = 0_c_int
        if (n < 2 .or. alpha <= 0.0_c_double .or. t_end <= 0.0_c_double) then
            rc = 1_c_int
            return
        end if
        h = 1.0_c_double / real(n, c_double)
        allocate(u(n * n), un(n * n))
        do i = 0, n - 1
            do j = 0, n - 1
                u(i * n + j + 1) = sin(PI * (real(j, c_double) + 0.5_c_double) * h) * &
                                   sin(PI * (real(i, c_double) + 0.5_c_double) * h)
            end do
        end do
        t = 0.0_c_double
        max_steps = 2000000
        do step = 1, max_steps
            if (t >= t_end) exit
            call ftcs_step(u, un, n, h, alpha, cfl, dt)
            if (dt <= 0.0_c_double) then
                rc = 2_c_int
                exit
            end if
            if (t + dt > t_end) dt = t_end - t
            call apply_step(u, n, h, alpha, dt)
            t = t + dt
        end do
        do idx = 1, n * n
            out_u(idx) = u(idx)
        end do
        if (.not. all_finite(u(1:n * n))) rc = 2_c_int
    end function oct_f_heat2d

    pure subroutine ftcs_step(u, un, n, h, alpha, cfl, dt)
        real(c_double), intent(in) :: u(:), un(:), h, alpha, cfl
        integer(c_int), intent(in) :: n
        real(c_double), intent(out) :: dt
        dt = cfl * h * h / (4.0_c_double * alpha)
    end subroutine ftcs_step

    subroutine apply_step(u, n, h, alpha, dt)
        real(c_double), intent(inout) :: u(:)
        integer(c_int), intent(in) :: n
        real(c_double), intent(in) :: h, alpha, dt
        real(c_double), allocatable :: un(:)
        real(c_double) :: lap, r
        integer(c_int) :: i, j, ip, im, jp, jm
        allocate(un(n * n))
        r = alpha * dt / (h * h)
        do i = 0, n - 1
            do j = 0, n - 1
                ip = i + 1; im = i - 1; jp = j + 1; jm = j - 1
                lap = 0.0_c_double
                if (ip < n) lap = lap + u(ip * n + j + 1)
                if (im >= 0) lap = lap + u(im * n + j + 1)
                if (jp < n) lap = lap + u(i * n + jp + 1)
                if (jm >= 0) lap = lap + u(i * n + jm + 1)
                un(i * n + j + 1) = u(i * n + j + 1) + r * (lap - 4.0_c_double * u(i * n + j + 1))
            end do
        end do
        u(1:n * n) = un(1:n * n)
        deallocate(un)
    end subroutine apply_step

    ! -----------------------------------------------------------------------
    ! Poisson: -lap(u) = 2 pi^2 sin(pi x) sin(pi y) with odd-reflection ghosts,
    ! red-black SOR. The wall diagonal is 5 (edge) / 6 (corner) because the
    ! ghost mirrors the cell itself; using 4 everywhere is NOT convergent.
    ! -----------------------------------------------------------------------
    function oct_f_poisson2d(n, max_iter, tol, iters_out, residual_out, out_u) &
            bind(C, name="oct_f_poisson2d") result(rc)
        integer(c_int), value, intent(in) :: n, max_iter
        real(c_double), value, intent(in) :: tol
        integer(c_int), intent(out) :: iters_out
        real(c_double), intent(out) :: residual_out
        real(c_double), intent(out) :: out_u(*)
        real(c_double), allocatable :: u(:)
        real(c_double) :: h, omega, f, known, r0, r, sum
        integer(c_int) :: i, j, color, it, diag, rc

        rc = 0_c_int
        iters_out = 0_c_int
        residual_out = 0.0_c_double
        if (n < 4 .or. max_iter < 1) then
            rc = 1_c_int
            return
        end if
        h = 1.0_c_double / real(n, c_double)
        omega = 2.0_c_double / (1.0_c_double + sin(PI * h))
        allocate(u(n * n))
        u = 0.0_c_double
        r0 = max(resid(u, n, h), 1.0e-300_c_double)

        do it = 1, max_iter
            iters_out = it
            do color = 0, 1
                do i = 0, n - 1
                    do j = 0, n - 1
                        if (mod(i + j, 2) /= color) cycle
                        known = 0.0_c_double
                        diag = 4
                        if (i + 1 < n) then
                            known = known + u((i + 1) * n + j + 1)
                        else
                            diag = diag + 1
                        end if
                        if (i - 1 >= 0) then
                            known = known + u((i - 1) * n + j + 1)
                        else
                            diag = diag + 1
                        end if
                        if (j + 1 < n) then
                            known = known + u(i * n + j + 2)
                        else
                            diag = diag + 1
                        end if
                        if (j - 1 >= 0) then
                            known = known + u(i * n + j)
                        else
                            diag = diag + 1
                        end if
                        f = 2.0_c_double * PI * PI * &
                            sin(PI * (real(j, c_double) + 0.5_c_double) * h) * &
                            sin(PI * (real(i, c_double) + 0.5_c_double) * h)
                        sum = (known + h * h * f) / real(diag, c_double)
                        u(i * n + j + 1) = u(i * n + j + 1) + &
                                           omega * (sum - u(i * n + j + 1))
                    end do
                end do
            end do
            if (mod(it, 16) == 0) then
                r = resid(u, n, h) / r0
                if (r <= tol) then
                    residual_out = r
                    exit
                end if
            end if
        end do
        residual_out = resid(u, n, h) / r0
        do i = 1, n * n
            out_u(i) = u(i)
        end do
        if (.not. all_finite(u(1:n * n))) rc = 2_c_int
    end function oct_f_poisson2d

    pure function resid(u, n, h) result(m)
        real(c_double), intent(in) :: u(:), h
        integer(c_int), intent(in) :: n
        real(c_double) :: m, au, f, uu
        integer(c_int) :: i, j
        m = 0.0_c_double
        do i = 0, n - 1
            do j = 0, n - 1
                uu = u(i * n + j + 1)
                au = 4.0_c_double * uu
                if (i + 1 < n) then
                    au = au - u((i + 1) * n + j + 1)
                else
                    au = au + uu
                end if
                if (i - 1 >= 0) then
                    au = au - u((i - 1) * n + j + 1)
                else
                    au = au + uu
                end if
                if (j + 1 < n) then
                    au = au - u(i * n + j + 2)
                else
                    au = au + uu
                end if
                if (j - 1 >= 0) then
                    au = au - u(i * n + j)
                else
                    au = au + uu
                end if
                au = au / (h * h)
                f = 2.0_c_double * PI * PI * &
                    sin(PI * (real(j, c_double) + 0.5_c_double) * h) * &
                    sin(PI * (real(i, c_double) + 0.5_c_double) * h)
                m = max(m, abs(au - f))
            end do
        end do
    end function resid

    ! -----------------------------------------------------------------------
    ! Kepler two-body: classical RK4 with a fixed step. state_io is
    ! [x, y, vx, vy] in units where GM and the semi-major axis are 1.
    ! -----------------------------------------------------------------------
    function oct_f_kepler(gm, dt, n_steps, state_io) bind(C, name="oct_f_kepler") result(rc)
        real(c_double), value, intent(in) :: gm, dt
        integer(c_int), value, intent(in) :: n_steps
        real(c_double), intent(inout) :: state_io(*)
        real(c_double) :: s(4), k1(4), k2(4), k3(4), k4(4)
        integer(c_int) :: step, rc

        rc = 0_c_int
        if (gm <= 0.0_c_double .or. n_steps < 1) then
            rc = 1_c_int
            return
        end if
        s(1) = state_io(1); s(2) = state_io(2); s(3) = state_io(3); s(4) = state_io(4)
        do step = 1, n_steps
            call deriv(gm, s, k1)
            call deriv(gm, s + 0.5_c_double * dt * k1, k2)
            call deriv(gm, s + 0.5_c_double * dt * k2, k3)
            call deriv(gm, s + dt * k3, k4)
            s = s + (dt / 6.0_c_double) * (k1 + 2.0_c_double * k2 + 2.0_c_double * k3 + k4)
        end do
        state_io(1) = s(1); state_io(2) = s(2); state_io(3) = s(3); state_io(4) = s(4)
        if (.not. all_finite(s)) rc = 2_c_int
    end function oct_f_kepler

    pure subroutine deriv(gm, s, d)
        real(c_double), intent(in) :: gm, s(4)
        real(c_double), intent(out) :: d(4)
        real(c_double) :: r, r3
        r = sqrt(s(1) * s(1) + s(2) * s(2))
        r3 = max(r * r * r, 1.0e-300_c_double)
        d(1) = s(3)
        d(2) = s(4)
        d(3) = -gm * s(1) / r3
        d(4) = -gm * s(2) / r3
    end subroutine deriv

    ! -----------------------------------------------------------------------
    ! Compiler identification string, used in every "fortran.real64 [...]"
    ! backend label so a result always names the compiler that produced it.
    ! -----------------------------------------------------------------------
    function oct_f_compiler_id(buf, len) bind(C, name="oct_f_compiler_id") result(rc)
        character(kind=c_char), intent(out) :: buf(*)
        integer(c_int), value, intent(in) :: len
        character(len=64) :: id
        integer(c_int) :: rc, i, n

        rc = 0_c_int
        id = "gfortran/unknown (Fortran 2018 kernels, Octopus)"
        n = min(len - 1, int(len_trim(id), c_int))
        do i = 1, n
            buf(i) = id(i:i)
        end do
        if (len > 0) buf(n + 1) = c_null_char
        if (len < 1) rc = 1_c_int
    end function oct_f_compiler_id

end module octopus_fortran_kernels
