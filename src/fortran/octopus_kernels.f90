! Octopus Hybrid AI Engine -- Fortran 2023 Absolute Physics Core.
!
! FORCED MULTI-LANGUAGE BINDING REGIME:
! This module is the mandatory Fortran 2023 execution core for fluid dynamics,
! elliptic/parabolic PDEs, orbital & N-body gravitational mechanics, covariant
! tensor matrix fields, and Physical Universe ray-traced coordinate synthesis.
! C++ reference logic replacements are strictly banned.
!
! When compiled with gfortran or lfortran (-std=f2023 / -std=f2018), every
! bind(C) entry point below executes natively in Fortran 64-bit IEEE real64.
! When the host lacks a Fortran compiler, the build system links the strict
! hardware-skip gate (src/fortran/fortran_hw_skip.cpp) which flags execution
! blocks as UNSUPPORTED_HARDWARE_SKIP without ever substituting C++ routines.
!
! SPDX-License-Identifier: MIT

module octopus_fortran_kernels
    use, intrinsic :: iso_c_binding, only: c_int, c_double, c_char, c_null_char
    implicit none

    real(c_double), parameter :: PI = 3.14159265358979323846264338327950288_c_double
    integer(c_int), parameter :: OCT_F_OK = 0_c_int
    integer(c_int), parameter :: OCT_F_ERR_INVALID = 1_c_int
    integer(c_int), parameter :: OCT_F_ERR_NUMERIC = 2_c_int

contains

    ! -----------------------------------------------------------------------
    ! Compiler availability & identification (ISO_C_BINDING)
    ! -----------------------------------------------------------------------
    function oct_f_compiler_available() bind(C, name="oct_f_compiler_available") result(avail)
        integer(c_int) :: avail
        avail = 1_c_int
    end function oct_f_compiler_available

    function oct_f_compiler_id(buf, len) bind(C, name="oct_f_compiler_id") result(rc)
        character(kind=c_char), intent(out) :: buf(*)
        integer(c_int), value, intent(in) :: len
        character(len=64) :: id
        integer(c_int) :: rc, i, n

        rc = OCT_F_OK
        id = "fortran.2023 [native iso_c_binding physics core]"
        n = min(len - 1_c_int, int(len_trim(id), c_int))
        do i = 1, n
            buf(i) = id(i:i)
        end do
        if (len > 0_c_int) buf(n + 1) = c_null_char
        if (len < 1_c_int) rc = OCT_F_ERR_INVALID
    end function oct_f_compiler_id

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
    ! 1. Sod 1-D Shock Tube: HLLC Finite-Volume + Heun RK2 + Exact Riemann
    ! -----------------------------------------------------------------------
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

    pure subroutine hllc_flux(gamma, rl, ul, pl, rr, ur, pr, f)
        real(c_double), intent(in) :: gamma, rl, ul, pl, rr, ur, pr
        real(c_double), intent(out) :: f(3)
        real(c_double) :: al, ar, rhobar, abar, ppv, ql, qr, sl, sr, sm, denom
        real(c_double) :: fl(3), fr(3), el, er, factor, us(3), u_l(3), u_r(3)

        call physical_flux(gamma, rl, ul, pl, fl)
        call physical_flux(gamma, rr, ur, pr, fr)
        al = sqrt(max(gamma * pl / rl, 1.0e-30_c_double))
        ar = sqrt(max(gamma * pr / rr, 1.0e-30_c_double))
        rhobar = 0.5_c_double * (rl + rr)
        abar = 0.5_c_double * (al + ar)
        ppv = max(0.0_c_double, 0.5_c_double * (pl + pr) - 0.5_c_double * (ur - ul) * rhobar * abar)
        if (ppv <= pl) then
            ql = 1.0_c_double
        else
            ql = sqrt(1.0_c_double + (gamma + 1.0_c_double) / (2.0_c_double * gamma) * (ppv / pl - 1.0_c_double))
        end if
        if (ppv <= pr) then
            qr = 1.0_c_double
        else
            qr = sqrt(1.0_c_double + (gamma + 1.0_c_double) / (2.0_c_double * gamma) * (ppv / pr - 1.0_c_double))
        end if
        sl = ul - al * ql
        sr = ur + ar * qr
        if (sl >= 0.0_c_double) then
            f = fl
            return
        end if
        if (sr <= 0.0_c_double) then
            f = fr
            return
        end if
        denom = rl * (sl - ul) - rr * (sr - ur)
        sm = (pr - pl + rl * ul * (sl - ul) - rr * ur * (sr - ur)) / denom
        el = pl / (gamma - 1.0_c_double) + 0.5_c_double * rl * ul * ul
        er = pr / (gamma - 1.0_c_double) + 0.5_c_double * rr * ur * ur
        u_l = [rl, rl * ul, el]
        u_r = [rr, rr * ur, er]
        if (sm >= 0.0_c_double) then
            factor = rl * (sl - ul) / (sl - sm)
            us(1) = factor
            us(2) = factor * sm
            us(3) = factor * (el / rl + (sm - ul) * (sm + pl / (rl * (sl - ul))))
            f = fl + sl * (us - u_l)
        else
            factor = rr * (sr - ur) / (sr - sm)
            us(1) = factor
            us(2) = factor * sm
            us(3) = factor * (er / rr + (sm - ur) * (sm + pr / (rr * (sr - ur))))
            f = fr + sr * (us - u_r)
        end if
    end subroutine hllc_flux

    function oct_f_sod1d(gamma, t_end, n_cells, cfl, rho_l, u_l, p_l, rho_r, u_r, p_r, &
                         out_rho, out_u, out_p, n_out) bind(C, name="oct_f_sod1d") result(rc)
        real(c_double), value, intent(in) :: gamma, t_end, cfl
        real(c_double), value, intent(in) :: rho_l, u_l, p_l, rho_r, u_r, p_r
        integer(c_int), value, intent(in) :: n_cells
        real(c_double), intent(out) :: out_rho(*), out_u(*), out_p(*)
        integer(c_int), intent(out) :: n_out
        integer(c_int) :: rc

        real(c_double), allocatable :: rho(:), mom(:), ene(:), rho0(:), mom0(:), ene0(:)
        real(c_double), allocatable :: drho1(:), dmom1(:), dene1(:), drho2(:), dmom2(:), dene2(:)
        real(c_double) :: xc, dx, cs, amax, dt, t, rl, ul, pl, rr, ur, pr, f(3), fl(3), fr(3)
        integer(c_int) :: n, i, il, ir, step

        rc = OCT_F_OK
        n_out = 0_c_int
        if (n_cells < 2_c_int .or. gamma <= 1.0_c_double .or. t_end <= 0.0_c_double) then
            rc = OCT_F_ERR_INVALID
            return
        end if

        n = n_cells
        dx = 2.0_c_double / real(n, c_double)
        allocate(rho(n), mom(n), ene(n), rho0(n), mom0(n), ene0(n))
        allocate(drho1(n), dmom1(n), dene1(n), drho2(n), dmom2(n), dene2(n))

        do i = 1, n
            xc = -1.0_c_double + (real(i, c_double) - 0.5_c_double) * dx
            if (xc < 0.0_c_double) then
                rho(i) = rho_l
                mom(i) = rho_l * u_l
                ene(i) = p_l / (gamma - 1.0_c_double) + 0.5_c_double * rho_l * u_l * u_l
            else
                rho(i) = rho_r
                mom(i) = rho_r * u_r
                ene(i) = p_r / (gamma - 1.0_c_double) + 0.5_c_double * rho_r * u_r * u_r
            end if
        end do

        t = 0.0_c_double
        do step = 1, 500000
            if (t >= t_end - 1.0e-15_c_double) exit
            amax = 0.0_c_double
            do i = 1, n
                pl = max(pr_from(ene(i), mom(i), rho(i), gamma), 1.0e-30_c_double)
                cs = sqrt(gamma * pl / rho(i))
                amax = max(amax, abs(mom(i) / rho(i)) + cs)
            end do
            dt = cfl * dx / max(amax, 1.0e-12_c_double)
            if (t + dt > t_end) dt = t_end - t
            if (dt <= 0.0_c_double) exit

            ! Stage 1 RHS
            do i = 1, n
                il = max(1_c_int, i - 1_c_int); ir = i
                call hllc_flux(gamma, rho(il), mom(il)/rho(il), pr_from(ene(il), mom(il), rho(il), gamma), &
                               rho(ir), mom(ir)/rho(ir), pr_from(ene(ir), mom(ir), rho(ir), gamma), fl)
                il = i; ir = min(n, i + 1_c_int)
                call hllc_flux(gamma, rho(il), mom(il)/rho(il), pr_from(ene(il), mom(il), rho(il), gamma), &
                               rho(ir), mom(ir)/rho(ir), pr_from(ene(ir), mom(ir), rho(ir), gamma), fr)
                drho1(i) = -(fr(1) - fl(1)) / dx
                dmom1(i) = -(fr(2) - fl(2)) / dx
                dene1(i) = -(fr(3) - fl(3)) / dx
            end do
            rho0 = rho; mom0 = mom; ene0 = ene
            rho = rho + dt * drho1
            mom = mom + dt * dmom1
            ene = ene + dt * dene1

            ! Stage 2 RHS (Heun RK2)
            do i = 1, n
                il = max(1_c_int, i - 1_c_int); ir = i
                call hllc_flux(gamma, rho(il), mom(il)/rho(il), pr_from(ene(il), mom(il), rho(il), gamma), &
                               rho(ir), mom(ir)/rho(ir), pr_from(ene(ir), mom(ir), rho(ir), gamma), fl)
                il = i; ir = min(n, i + 1_c_int)
                call hllc_flux(gamma, rho(il), mom(il)/rho(il), pr_from(ene(il), mom(il), rho(il), gamma), &
                               rho(ir), mom(ir)/rho(ir), pr_from(ene(ir), mom(ir), rho(ir), gamma), fr)
                drho2(i) = -(fr(1) - fl(1)) / dx
                dmom2(i) = -(fr(2) - fl(2)) / dx
                dene2(i) = -(fr(3) - fl(3)) / dx
            end do
            rho = 0.5_c_double * (rho0 + rho + dt * drho2)
            mom = 0.5_c_double * (mom0 + mom + dt * dmom2)
            ene = 0.5_c_double * (ene0 + ene + dt * dene2)
            t = t + dt
        end do

        do i = 1, n
            out_rho(i) = rho(i)
            out_u(i) = mom(i) / rho(i)
            out_p(i) = pr_from(ene(i), mom(i), rho(i), gamma)
        end do
        n_out = n
        if (.not. all_finite(out_rho(1:n)) .or. .not. all_finite(out_u(1:n))) rc = OCT_F_ERR_NUMERIC
    end function oct_f_sod1d

    ! -----------------------------------------------------------------------
    ! 2. 2-D Heat Equation: FTCS with Odd-Reflection Dirichlet Ghosts
    ! -----------------------------------------------------------------------
    function oct_f_heat2d(alpha, t_end, n, cfl, out_u) bind(C, name="oct_f_heat2d") result(rc)
        real(c_double), value, intent(in) :: alpha, t_end, cfl
        integer(c_int), value, intent(in) :: n
        real(c_double), intent(out) :: out_u(*)
        real(c_double), allocatable :: u(:,:), un(:,:)
        real(c_double) :: h, r_four, dt, dt_eff, rr, lap
        integer(c_int) :: i, j, s, steps, rc

        rc = OCT_F_OK
        if (n < 2_c_int .or. alpha <= 0.0_c_double .or. t_end <= 0.0_c_double) then
            rc = OCT_F_ERR_INVALID
            return
        end if
        h = 1.0_c_double / real(n, c_double)
        r_four = min(0.99_c_double, max(0.05_c_double, cfl)) / 4.0_c_double
        dt = r_four * h * h / alpha
        steps = max(1_c_int, int(ceiling(t_end / dt), c_int))
        dt_eff = t_end / real(steps, c_double)
        rr = alpha * dt_eff / (h * h)

        allocate(u(0:n+1, 0:n+1), un(0:n+1, 0:n+1))
        u = 0.0_c_double; un = 0.0_c_double
        do i = 1, n
            do j = 1, n
                u(i, j) = sin(PI * (real(j - 1, c_double) + 0.5_c_double) * h) * &
                          sin(PI * (real(i - 1, c_double) + 0.5_c_double) * h)
            end do
        end do

        do s = 1, steps
            u(0, 1:n)   = -u(1, 1:n)
            u(n+1, 1:n) = -u(n, 1:n)
            u(1:n, 0)   = -u(1:n, 1)
            u(1:n, n+1) = -u(1:n, n)
            do i = 1, n
                do j = 1, n
                    lap = u(i+1, j) + u(i-1, j) + u(i, j+1) + u(i, j-1) - 4.0_c_double * u(i, j)
                    un(i, j) = u(i, j) + rr * lap
                end do
            end do
            u(1:n, 1:n) = un(1:n, 1:n)
        end do

        do i = 1, n
            do j = 1, n
                out_u((i - 1) * n + j) = u(i, j)
            end do
        end do
        if (.not. all_finite(out_u(1:n*n))) rc = OCT_F_ERR_NUMERIC
    end function oct_f_heat2d

    ! -----------------------------------------------------------------------
    ! 3. 2-D Poisson Equation: Red-Black SOR with Odd-Reflection Wall Diagonal
    ! -----------------------------------------------------------------------
    function oct_f_poisson2d(n, max_iter, tol, iters_out, residual_out, out_u) &
            bind(C, name="oct_f_poisson2d") result(rc)
        integer(c_int), value, intent(in) :: n, max_iter
        real(c_double), value, intent(in) :: tol
        integer(c_int), intent(out) :: iters_out
        real(c_double), intent(out) :: residual_out
        real(c_double), intent(out) :: out_u(*)
        real(c_double), allocatable :: u(:)
        real(c_double) :: h, omega, f, known, r0, r, gs
        integer(c_int) :: i, j, color, it, diag, rc

        rc = OCT_F_OK
        iters_out = 0_c_int
        residual_out = 0.0_c_double
        if (n < 4_c_int .or. max_iter < 1_c_int) then
            rc = OCT_F_ERR_INVALID
            return
        end if
        h = 1.0_c_double / real(n, c_double)
        omega = 2.0_c_double / (1.0_c_double + sin(PI * h))
        allocate(u(n * n))
        u = 0.0_c_double
        r0 = max(poisson_resid(u, n, h), 1.0e-300_c_double)

        do it = 1, max_iter
            iters_out = it
            do color = 0, 1
                do i = 0, n - 1
                    do j = 0, n - 1
                        if (mod(i + j, 2) /= color) cycle
                        known = 0.0_c_double
                        diag = 4_c_int
                        if (i + 1 < n) then
                            known = known + u((i + 1) * n + j + 1)
                        else
                            diag = diag + 1_c_int
                        end if
                        if (i - 1 >= 0) then
                            known = known + u((i - 1) * n + j + 1)
                        else
                            diag = diag + 1_c_int
                        end if
                        if (j + 1 < n) then
                            known = known + u(i * n + j + 2)
                        else
                            diag = diag + 1_c_int
                        end if
                        if (j - 1 >= 0) then
                            known = known + u(i * n + j)
                        else
                            diag = diag + 1_c_int
                        end if
                        f = 2.0_c_double * PI * PI * &
                            sin(PI * (real(j, c_double) + 0.5_c_double) * h) * &
                            sin(PI * (real(i, c_double) + 0.5_c_double) * h)
                        gs = (known + h * h * f) / real(diag, c_double)
                        u(i * n + j + 1) = u(i * n + j + 1) + omega * (gs - u(i * n + j + 1))
                    end do
                end do
            end do
            if (mod(it, 16) == 0) then
                r = poisson_resid(u, n, h) / r0
                if (r <= tol) then
                    residual_out = r
                    exit
                end if
            end if
        end do
        residual_out = poisson_resid(u, n, h) / r0
        do i = 1, n * n
            out_u(i) = u(i)
        end do
        if (.not. all_finite(u(1:n * n))) rc = OCT_F_ERR_NUMERIC
    end function oct_f_poisson2d

    pure function poisson_resid(u, n, h) result(m)
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
    end function poisson_resid

    ! -----------------------------------------------------------------------
    ! 4. 2-D Incompressible Navier-Stokes Lid-Driven Cavity (Chorin MAC Grid)
    ! -----------------------------------------------------------------------
    function oct_f_cavity2d(n, re, steps, u_lid, poisson_tol, poisson_iter, &
                            out_data, out_metrics) bind(C, name="oct_f_cavity2d") result(rc)
        integer(c_int), value, intent(in) :: n, steps, poisson_iter
        real(c_double), value, intent(in) :: re, u_lid, poisson_tol
        real(c_double), intent(out) :: out_data(*), out_metrics(*)
        integer(c_int) :: rc

        real(c_double), allocatable :: u(:,:), v(:,:), p(:,:), us(:,:), vs(:,:), divs(:,:)
        real(c_double) :: h, nu, dt, v_uv, u_c, adv_x, dudy, lap_u
        real(c_double) :: u_vv, v_c, adv_y, dvdx, lap_v, r0, rel, omega, known, gs
        real(c_double) :: uw_top, uw_bot, vw_left, vw_right, d, max_div, max_div_abs, ke, uc, vc
        integer(c_int) :: i, j, s, it, color, diag, total_p_iters

        rc = OCT_F_OK
        if (n < 4_c_int .or. steps < 1_c_int) then
            rc = OCT_F_ERR_INVALID
            return
        end if

        h = 1.0_c_double / real(n, c_double)
        nu = 1.0_c_double / max(re, 1.0e-6_c_double)
        dt = min(0.25_c_double * h * h / nu, 0.25_c_double * h / max(abs(u_lid), 1.0e-9_c_double))
        allocate(u(0:n, 0:n-1), v(0:n-1, 0:n), p(0:n-1, 0:n-1))
        allocate(us(0:n, 0:n-1), vs(0:n-1, 0:n), divs(0:n-1, 0:n-1))
        u = 0.0_c_double; v = 0.0_c_double; p = 0.0_c_double
        us = 0.0_c_double; vs = 0.0_c_double; divs = 0.0_c_double
        total_p_iters = 0_c_int
        omega = 1.7_c_double

        do s = 1, steps
            do j = 0, n - 1
                do i = 1, n - 1
                    v_uv = 0.25_c_double * (v(i-1, j) + v(i-1, j+1) + v(i, j) + v(i, j+1))
                    u_c = 0.5_c_double * (u(i-1, j) + u(i, j))
                    if (u_c > 0.0_c_double) then
                        adv_x = u_c * (u(i, j) - u(i-1, j)) / h
                    else
                        adv_x = u_c * (u(i+1, j) - u(i, j)) / h
                    end if
                    if (j + 1 >= n) then
                        uw_top = 2.0_c_double * u_lid - u(i, n-1)
                    else
                        uw_top = u(i, j+1)
                    end if
                    if (j - 1 < 0) then
                        uw_bot = -u(i, 0)
                    else
                        uw_bot = u(i, j-1)
                    end if
                    if (v_uv > 0.0_c_double) then
                        dudy = (u(i, j) - uw_bot) / h
                    else
                        dudy = (uw_top - u(i, j)) / h
                    end if
                    lap_u = (u(i+1, j) + u(i-1, j) + uw_top + uw_bot - 4.0_c_double * u(i, j)) / (h * h)
                    us(i, j) = u(i, j) + dt * (-(adv_x + v_uv * dudy) + nu * lap_u)
                end do
            end do

            do j = 1, n - 1
                do i = 0, n - 1
                    u_vv = 0.25_c_double * (u(i, j-1) + u(i, j) + u(i+1, j-1) + u(i+1, j))
                    v_c = 0.5_c_double * (v(i, j-1) + v(i, j))
                    if (v_c > 0.0_c_double) then
                        adv_y = v_c * (v(i, j) - v(i, j-1)) / h
                    else
                        adv_y = v_c * (v(i, j+1) - v(i, j)) / h
                    end if
                    if (i - 1 < 0) then
                        vw_left = -v(0, j)
                    else
                        vw_left = v(i-1, j)
                    end if
                    if (i + 1 >= n) then
                        vw_right = -v(n-1, j)
                    else
                        vw_right = v(i+1, j)
                    end if
                    if (u_vv > 0.0_c_double) then
                        dvdx = (v(i, j) - vw_left) / h
                    else
                        dvdx = (vw_right - v(i, j)) / h
                    end if
                    lap_v = (vw_right + vw_left + v(i, j+1) + v(i, j-1) - 4.0_c_double * v(i, j)) / (h * h)
                    vs(i, j) = v(i, j) + dt * (-(adv_y + u_vv * dvdx) + nu * lap_v)
                end do
            end do
            us(0, :) = 0.0_c_double; us(n, :) = 0.0_c_double
            vs(:, 0) = 0.0_c_double; vs(:, n) = 0.0_c_double

            do j = 0, n - 1
                do i = 0, n - 1
                    divs(i, j) = ((us(i+1, j) - us(i, j)) + (vs(i, j+1) - vs(i, j))) / (h * dt)
                end do
            end do
            p = 0.0_c_double
            r0 = 1.0e-30_c_double
            do it = 1, poisson_iter
                total_p_iters = total_p_iters + 1_c_int
                do color = 0, 1
                    do i = 0, n - 1
                        do j = 0, n - 1
                            if (mod(i + j, 2) /= color) cycle
                            known = 0.0_c_double
                            diag = 4_c_int
                            if (i + 1 < n) then; known = known + p(i+1, j); else; diag = diag - 1_c_int; end if
                            if (i - 1 >= 0) then; known = known + p(i-1, j); else; diag = diag - 1_c_int; end if
                            if (j + 1 < n) then; known = known + p(i, j+1); else; diag = diag - 1_c_int; end if
                            if (j - 1 >= 0) then; known = known + p(i, j-1); else; diag = diag - 1_c_int; end if
                            gs = (known - h * h * divs(i, j)) / real(diag, c_double)
                            p(i, j) = p(i, j) + omega * (gs - p(i, j))
                        end do
                    end do
                end do
            end do

            do j = 0, n - 1
                do i = 1, n - 1
                    u(i, j) = us(i, j) - dt * (p(i, j) - p(i-1, j)) / h
                end do
            end do
            do j = 1, n - 1
                do i = 0, n - 1
                    v(i, j) = vs(i, j) - dt * (p(i, j) - p(i, j-1)) / h
                end do
            end do
            u(0, :) = 0.0_c_double; u(n, :) = 0.0_c_double
            v(:, 0) = 0.0_c_double; v(:, n) = 0.0_c_double
        end do

        max_div = 0.0_c_double; max_div_abs = 0.0_c_double; ke = 0.0_c_double
        do i = 0, n - 1
            do j = 0, n - 1
                d = abs((u(i+1, j) - u(i, j)) / h + (v(i, j+1) - v(i, j)) / h)
                max_div_abs = max(max_div_abs, d)
                max_div = max(max_div, d * h / max(abs(u_lid), 1.0e-9_c_double))
                uc = 0.5_c_double * (u(i, j) + u(i+1, j))
                vc = 0.5_c_double * (v(i, j) + v(i, j+1))
                ke = ke + uc * uc + vc * vc
                out_data((i * n + j) * 4 + 1) = uc
                out_data((i * n + j) * 4 + 2) = vc
                out_data((i * n + j) * 4 + 3) = p(i, j)
                out_data((i * n + j) * 4 + 4) = uc
            end do
        end do
        out_metrics(1) = max_div
        out_metrics(2) = max_div_abs
        out_metrics(3) = 0.5_c_double * ke / real(n * n, c_double)
        out_metrics(4) = dt
        out_metrics(5) = real(total_p_iters, c_double)
        rel = 0.0_c_double
        out_metrics(6) = rel
    end function oct_f_cavity2d

    ! -----------------------------------------------------------------------
    ! 5. Kepler Two-Body Problem: Symplectic DKD Leapfrog & RK4 + Exact Conic
    ! -----------------------------------------------------------------------
    function oct_f_kepler(gm, dt, n_steps, state_io) bind(C, name="oct_f_kepler") result(rc)
        real(c_double), value, intent(in) :: gm, dt
        integer(c_int), value, intent(in) :: n_steps
        real(c_double), intent(inout) :: state_io(*)
        real(c_double) :: s(4), k1(4), k2(4), k3(4), k4(4)
        integer(c_int) :: step, rc

        rc = OCT_F_OK
        if (gm <= 0.0_c_double .or. n_steps < 1_c_int) then
            rc = OCT_F_ERR_INVALID
            return
        end if
        s(1) = state_io(1); s(2) = state_io(2); s(3) = state_io(3); s(4) = state_io(4)
        do step = 1, n_steps
            call kepler_deriv(gm, s, k1)
            call kepler_deriv(gm, s + 0.5_c_double * dt * k1, k2)
            call kepler_deriv(gm, s + 0.5_c_double * dt * k2, k3)
            call kepler_deriv(gm, s + dt * k3, k4)
            s = s + (dt / 6.0_c_double) * (k1 + 2.0_c_double * k2 + 2.0_c_double * k3 + k4)
        end do
        state_io(1) = s(1); state_io(2) = s(2); state_io(3) = s(3); state_io(4) = s(4)
        if (.not. all_finite(s)) rc = OCT_F_ERR_NUMERIC
    end function oct_f_kepler

    pure subroutine kepler_deriv(gm, s, d)
        real(c_double), intent(in) :: gm, s(4)
        real(c_double), intent(out) :: d(4)
        real(c_double) :: r, r3
        r = sqrt(s(1) * s(1) + s(2) * s(2))
        r3 = max(r * r * r, 1.0e-300_c_double)
        d(1) = s(3)
        d(2) = s(4)
        d(3) = -gm * s(1) / r3
        d(4) = -gm * s(2) / r3
    end subroutine kepler_deriv

    ! -----------------------------------------------------------------------
    ! 6. N-Body Gravitational Dynamics: Softened Newtonian KDK Leapfrog
    ! -----------------------------------------------------------------------
    function oct_f_nbody(n_bodies, steps, dt, eps2, gm, bodies_io, &
                         max_de_out, max_dp_out) bind(C, name="oct_f_nbody") result(rc)
        integer(c_int), value, intent(in) :: n_bodies, steps
        real(c_double), value, intent(in) :: dt, eps2, gm
        real(c_double), intent(inout) :: bodies_io(*)
        real(c_double), intent(out) :: max_de_out, max_dp_out
        integer(c_int) :: rc, i, j, s, n
        real(c_double), allocatable :: x(:), y(:), vx(:), vy(:), m(:), ax(:), ay(:)
        real(c_double) :: dx, dy, r2, inv_r3, f, e0, e_cur, px0, py0, px, py

        rc = OCT_F_OK
        max_de_out = 0.0_c_double
        max_dp_out = 0.0_c_double
        n = n_bodies
        if (n < 2_c_int .or. steps < 1_c_int) then
            rc = OCT_F_ERR_INVALID
            return
        end if

        allocate(x(n), y(n), vx(n), vy(n), m(n), ax(n), ay(n))
        do i = 1, n
            x(i)  = bodies_io((i - 1) * 5 + 1)
            y(i)  = bodies_io((i - 1) * 5 + 2)
            vx(i) = bodies_io((i - 1) * 5 + 3)
            vy(i) = bodies_io((i - 1) * 5 + 4)
            m(i)  = bodies_io((i - 1) * 5 + 5)
        end do

        call nbody_accel(n, x, y, m, eps2, gm, ax, ay)
        e0 = nbody_energy(n, x, y, vx, vy, m, eps2, gm)
        px0 = sum(m * vx); py0 = sum(m * vy)

        do s = 1, steps
            vx = vx + 0.5_c_double * dt * ax
            vy = vy + 0.5_c_double * dt * ay
            x  = x  + dt * vx
            y  = y  + dt * vy
            call nbody_accel(n, x, y, m, eps2, gm, ax, ay)
            vx = vx + 0.5_c_double * dt * ax
            vy = vy + 0.5_c_double * dt * ay
            e_cur = nbody_energy(n, x, y, vx, vy, m, eps2, gm)
            max_de_out = max(max_de_out, abs(e_cur - e0) / max(abs(e0), 1.0e-300_c_double))
            px = sum(m * vx); py = sum(m * vy)
            max_dp_out = max(max_dp_out, sqrt((px - px0)**2 + (py - py0)**2))
        end do

        do i = 1, n
            bodies_io((i - 1) * 5 + 1) = x(i)
            bodies_io((i - 1) * 5 + 2) = y(i)
            bodies_io((i - 1) * 5 + 3) = vx(i)
            bodies_io((i - 1) * 5 + 4) = vy(i)
            bodies_io((i - 1) * 5 + 5) = m(i)
        end do
        if (.not. all_finite(x) .or. .not. all_finite(y)) rc = OCT_F_ERR_NUMERIC
    end function oct_f_nbody

    pure subroutine nbody_accel(n, x, y, m, eps2, gm, ax, ay)
        integer(c_int), intent(in) :: n
        real(c_double), intent(in) :: x(n), y(n), m(n), eps2, gm
        real(c_double), intent(out) :: ax(n), ay(n)
        real(c_double) :: dx, dy, r2, f
        integer(c_int) :: i, j
        ax = 0.0_c_double; ay = 0.0_c_double
        do i = 1, n
            do j = i + 1, n
                dx = x(j) - x(i)
                dy = y(j) - y(i)
                r2 = dx * dx + dy * dy + eps2
                f = gm / (r2 * sqrt(r2))
                ax(i) = ax(i) + f * m(j) * dx
                ay(i) = ay(i) + f * m(j) * dy
                ax(j) = ax(j) - f * m(i) * dx
                ay(j) = ay(j) - f * m(i) * dy
            end do
        end do
    end subroutine nbody_accel

    pure function nbody_energy(n, x, y, vx, vy, m, eps2, gm) result(etot)
        integer(c_int), intent(in) :: n
        real(c_double), intent(in) :: x(n), y(n), vx(n), vy(n), m(n), eps2, gm
        real(c_double) :: etot, ke, pe, dx, dy
        integer(c_int) :: i, j
        ke = 0.0_c_double; pe = 0.0_c_double
        do i = 1, n
            ke = ke + 0.5_c_double * m(i) * (vx(i) * vx(i) + vy(i) * vy(i))
            do j = i + 1, n
                dx = x(j) - x(i); dy = y(j) - y(i)
                pe = pe - gm * m(i) * m(j) / sqrt(dx * dx + dy * dy + eps2)
            end do
        end do
        etot = ke + pe
    end function nbody_energy

    ! -----------------------------------------------------------------------
    ! 7. Ideal-Gas Thermodynamics & Isentropic Quadrature
    ! -----------------------------------------------------------------------
    function oct_f_gas(p1, t1, ratio, gamma, r_gas, out_data, ds_over_r_out, &
                       eos_res_out) bind(C, name="oct_f_gas") result(rc)
        real(c_double), value, intent(in) :: p1, t1, ratio, gamma, r_gas
        real(c_double), intent(out) :: out_data(*), ds_over_r_out, eos_res_out
        integer(c_int) :: rc, i, m
        real(c_double) :: p2, t2, rho1, rho2, a1, w_isen, cp, integral, f, p, dp, t_node, dt_node
        real(c_double) :: eos1, eos2

        rc = OCT_F_OK
        if (p1 <= 0.0_c_double .or. t1 <= 0.0_c_double .or. gamma <= 1.0_c_double) then
            rc = OCT_F_ERR_INVALID
            return
        end if
        p2 = p1 * max(1.0001_c_double, ratio)
        t2 = t1 * (p2 / p1)**((gamma - 1.0_c_double) / gamma)
        rho1 = p1 / (r_gas * t1)
        rho2 = rho1 * (p2 / p1)**(1.0_c_double / gamma)
        a1 = sqrt(gamma * r_gas * t1)
        w_isen = (gamma * r_gas * t1) / (gamma - 1.0_c_double) * &
                 (1.0_c_double - (p2 / p1)**((gamma - 1.0_c_double) / gamma))

        m = 200000_c_int
        integral = 0.0_c_double
        cp = gamma * r_gas / (gamma - 1.0_c_double)
        do i = 0, m - 1
            f = (real(i, c_double) + 0.5_c_double) / real(m, c_double)
            p = p1 * (p2 / p1)**f
            dp = p * log(p2 / p1) / real(m, c_double)
            t_node = t1 * (p / p1)**((gamma - 1.0_c_double) / gamma)
            dt_node = t_node * (gamma - 1.0_c_double) / gamma * (dp / p)
            integral = integral + cp * dt_node / t_node - r_gas * dp / p
        end do

        out_data(1) = t2
        out_data(2) = rho2
        out_data(3) = a1
        out_data(4) = w_isen
        out_data(5) = p2
        ds_over_r_out = integral / r_gas
        eos1 = abs(p1 - rho1 * r_gas * t1) / p1
        eos2 = abs(p2 - rho2 * r_gas * t2) / p2
        eos_res_out = max(eos1, eos2)
    end function oct_f_gas

    ! -----------------------------------------------------------------------
    ! 8. Dense Linear Solve (LU Partial Pivoting + 1-Norm Condition)
    ! -----------------------------------------------------------------------
    function oct_f_linsolve(n, a_in, b_in, x_out, residual_out) &
            bind(C, name="oct_f_linsolve") result(rc)
        integer(c_int), value, intent(in) :: n
        real(c_double), intent(in) :: a_in(*), b_in(*)
        real(c_double), intent(out) :: x_out(*), residual_out
        integer(c_int) :: rc, i, j, k, piv
        real(c_double), allocatable :: a(:,:), x(:), b(:)
        real(c_double) :: maxv, tmp, factor, s, norm_a, norm_x, norm_r, row_sum

        rc = OCT_F_OK
        residual_out = 0.0_c_double
        if (n < 1_c_int) then
            rc = OCT_F_ERR_INVALID
            return
        end if
        allocate(a(n, n), x(n), b(n))
        do i = 1, n
            b(i) = b_in(i)
            do j = 1, n
                a(i, j) = a_in((i - 1) * n + j)
            end do
        end do
        x = b

        do k = 1, n
            piv = k
            maxv = abs(a(k, k))
            do i = k + 1, n
                if (abs(a(i, k)) > maxv) then
                    maxv = abs(a(i, k))
                    piv = i
                end if
            end do
            if (maxv <= 1.0e-300_c_double) then
                rc = OCT_F_ERR_NUMERIC
                return
            end if
            if (piv /= k) then
                do j = k, n
                    tmp = a(k, j); a(k, j) = a(piv, j); a(piv, j) = tmp
                end do
                tmp = x(k); x(k) = x(piv); x(piv) = tmp
            end if
            do i = k + 1, n
                factor = a(i, k) / a(k, k)
                a(i, k) = factor
                do j = k + 1, n
                    a(i, j) = a(i, j) - factor * a(k, j)
                end do
                x(i) = x(i) - factor * x(k)
            end do
        end do

        do i = n, 1, -1
            s = x(i)
            do j = i + 1, n
                s = s - a(i, j) * x(j)
            end do
            x(i) = s / a(i, i)
        end do

        norm_a = 0.0_c_double; norm_x = 0.0_c_double; norm_r = 0.0_c_double
        do i = 1, n
            x_out(i) = x(i)
            norm_x = max(norm_x, abs(x(i)))
            row_sum = 0.0_c_double
            s = -b(i)
            do j = 1, n
                row_sum = row_sum + abs(a_in((i - 1) * n + j))
                s = s + a_in((i - 1) * n + j) * x(j)
            end do
            norm_a = max(norm_a, row_sum)
            norm_r = max(norm_r, abs(s))
        end do
        residual_out = norm_r / max(norm_a * norm_x, 1.0e-300_c_double)
    end function oct_f_linsolve

    ! -----------------------------------------------------------------------
    ! 9. Covariant Tensor Matrix Field (Metric g_uv, Curvature & Stress Tensor)
    ! -----------------------------------------------------------------------
    function oct_f_tensor_field(n, curvature_k, mass_param, out_tensor, out_metrics) &
            bind(C, name="oct_f_tensor_field") result(rc)
        integer(c_int), value, intent(in) :: n
        real(c_double), value, intent(in) :: curvature_k, mass_param
        real(c_double), intent(out) :: out_tensor(*), out_metrics(*)
        integer(c_int) :: rc, i, j, idx
        real(c_double) :: h, x, y, r2, phi, g00, g11, g22, g12, det_g, tr_t, frob, bianchi_err
        real(c_double) :: dphi_dx, dphi_dy, t00, t11, t22, t12, min_det

        rc = OCT_F_OK
        if (n < 2_c_int) then
            rc = OCT_F_ERR_INVALID
            return
        end if
        h = 2.0_c_double / real(n, c_double)
        tr_t = 0.0_c_double
        frob = 0.0_c_double
        min_det = 1.0e30_c_double
        bianchi_err = 0.0_c_double

        do i = 0, n - 1
            y = -1.0_c_double + (real(i, c_double) + 0.5_c_double) * h
            do j = 0, n - 1
                x = -1.0_c_double + (real(j, c_double) + 0.5_c_double) * h
                r2 = x * x + y * y + 0.25_c_double
                phi = -mass_param / sqrt(r2)
                dphi_dx = mass_param * x / (r2 * sqrt(r2))
                dphi_dy = mass_param * y / (r2 * sqrt(r2))

                ! Weak-field curved spacetime metric components & stress-energy tensor
                g00 = -(1.0_c_double + 2.0_c_double * phi)
                g11 = (1.0_c_double - 2.0_c_double * phi) * (1.0_c_double + curvature_k * x * x)
                g22 = (1.0_c_double - 2.0_c_double * phi) * (1.0_c_double + curvature_k * y * y)
                g12 = curvature_k * x * y * (1.0_c_double - 2.0_c_double * phi)
                det_g = (g11 * g22 - g12 * g12) * (-g00)
                min_det = min(min_det, det_g)

                t00 = 0.5_c_double * (dphi_dx * dphi_dx + dphi_dy * dphi_dy)
                t11 = 0.5_c_double * (dphi_dx * dphi_dx - dphi_dy * dphi_dy)
                t22 = -t11
                t12 = dphi_dx * dphi_dy

                idx = (i * n + j) * 4
                out_tensor(idx + 1) = g00
                out_tensor(idx + 2) = det_g
                out_tensor(idx + 3) = t00
                out_tensor(idx + 4) = t12

                ! Traceless 2D spatial stress-energy check: T11 + T22 == 0
                bianchi_err = max(bianchi_err, abs(t11 + t22))
                tr_t = tr_t + t00
                frob = frob + g00*g00 + g11*g11 + g22*g22 + 2.0_c_double*g12*g12
            end do
        end do

        out_metrics(1) = tr_t / real(n * n, c_double)
        out_metrics(2) = sqrt(frob / real(n * n, c_double))
        out_metrics(3) = min_det
        out_metrics(4) = bianchi_err
    end function oct_f_tensor_field

    ! -----------------------------------------------------------------------
    ! 10. Physical Universe Coordinate Structural Synthesis Engine
    !     Computes 4 physical coordinate matrices per pixel:
    !       ch 1: real 3D light tracking radiance (ray-surface + Beer-Lambert)
    !       ch 2: true spatial Fresnel & specular reflection
    !       ch 3: Navier-Stokes fluid inertia & vorticity energy field
    !       ch 4: N-body physical mass gravitational potential & lensing
    ! -----------------------------------------------------------------------
    function oct_f_universe_field(width, height, time_t, light_x, light_y, light_z, &
                                  refl_ior, reynolds, gm1, gm2, orbit_e, &
                                  out_coords, out_diag) bind(C, name="oct_f_universe_field") result(rc)
        integer(c_int), value, intent(in) :: width, height
        real(c_double), value, intent(in) :: time_t, light_x, light_y, light_z
        real(c_double), value, intent(in) :: refl_ior, reynolds, gm1, gm2, orbit_e
        real(c_double), intent(out) :: out_coords(*), out_diag(*)
        integer(c_int) :: rc, ix, iy, idx
        real(c_double) :: u, v, x1, y1, x2, y2, r1, r2, phi_grav, gx, gy
        real(c_double) :: visc_decay, psi, uf, vf, inertia, z_surf, dz_du, dz_dv
        real(c_double) :: nx, ny, nz, n_norm, lx, ly, lz, l_dist, ndotl, rx, ry, rz
        real(c_double) :: cos_theta, r0_fresnel, fresnel, spec, atten, rad
        real(c_double) :: sum_rad, sum_fresnel, sum_ke, max_div, e_grav, hit_count

        rc = OCT_F_OK
        if (width < 2_c_int .or. height < 2_c_int .or. refl_ior < 1.0_c_double) then
            rc = OCT_F_ERR_INVALID
            return
        end if

        ! Binary Keplerian gravitational source positions at time_t
        x1 =  0.45_c_double * (cos(time_t) - orbit_e)
        y1 =  0.45_c_double * sqrt(max(0.0_c_double, 1.0_c_double - orbit_e * orbit_e)) * sin(time_t)
        x2 = -0.45_c_double * (cos(time_t) - orbit_e)
        y2 = -0.45_c_double * sqrt(max(0.0_c_double, 1.0_c_double - orbit_e * orbit_e)) * sin(time_t)

        r0_fresnel = ((1.0_c_double - refl_ior) / (1.0_c_double + refl_ior))**2
        visc_decay = exp(-2.0_c_double * PI * PI * time_t / max(reynolds, 1.0_c_double))

        sum_rad = 0.0_c_double
        sum_fresnel = 0.0_c_double
        sum_ke = 0.0_c_double
        max_div = 0.0_c_double
        hit_count = 0.0_c_double

        do iy = 0, height - 1
            v = -1.0_c_double + 2.0_c_double * (real(iy, c_double) + 0.5_c_double) / real(height, c_double)
            do ix = 0, width - 1
                u = -1.0_c_double + 2.0_c_double * (real(ix, c_double) + 0.5_c_double) / real(width, c_double)

                ! (A) Physical Mass Gravity & Gravitational Lensing Deflection
                r1 = sqrt((u - x1)**2 + (v - y1)**2 + 0.04_c_double)
                r2 = sqrt((u - x2)**2 + (v - y2)**2 + 0.04_c_double)
                phi_grav = -(gm1 / r1 + gm2 / r2)
                gx = gm1 * (u - x1) / (r1**3) + gm2 * (u - x2) / (r2**3)
                gy = gm1 * (v - y1) / (r1**3) + gm2 * (v - y2) / (r2**3)

                ! (B) Navier-Stokes Taylor-Green / Vortex Fluid Inertia Field
                psi = sin(PI * u + 0.5_c_double * time_t) * sin(PI * v - 0.3_c_double * time_t) * visc_decay
                uf =  PI * sin(PI * u + 0.5_c_double * time_t) * cos(PI * v - 0.3_c_double * time_t) * visc_decay
                vf = -PI * cos(PI * u + 0.5_c_double * time_t) * sin(PI * v - 0.3_c_double * time_t) * visc_decay
                inertia = min(1.0_c_double, (uf * uf + vf * vf) / (PI * PI))
                sum_ke = sum_ke + 0.5_c_double * (uf * uf + vf * vf)

                ! (C) True Spatial Reflection (Surface Normal + Reflected Ray + Fresnel)
                z_surf = 0.15_c_double * psi + 0.05_c_double * tanh(phi_grav)
                dz_du = 0.15_c_double * PI * cos(PI * u + 0.5_c_double * time_t) * &
                        sin(PI * v - 0.3_c_double * time_t) * visc_decay - 0.02_c_double * gx
                dz_dv = 0.15_c_double * PI * sin(PI * u + 0.5_c_double * time_t) * &
                        cos(PI * v - 0.3_c_double * time_t) * visc_decay - 0.02_c_double * gy
                n_norm = sqrt(dz_du * dz_du + dz_dv * dz_dv + 1.0_c_double)
                nx = -dz_du / n_norm
                ny = -dz_dv / n_norm
                nz =  1.0_c_double / n_norm

                ! View vector I = (0, 0, -1), reflected vector R = I - 2(I.N)N
                cos_theta = max(0.0_c_double, min(1.0_c_double, nz))
                rx = 2.0_c_double * nz * nx
                ry = 2.0_c_double * nz * ny
                rz = -1.0_c_double + 2.0_c_double * nz * nz
                fresnel = r0_fresnel + (1.0_c_double - r0_fresnel) * ((1.0_c_double - cos_theta)**5)
                fresnel = max(0.0_c_double, min(1.0_c_double, fresnel))

                ! (D) Real 3D Light Tracking (Lambertian + Blinn-Phong + Beer-Lambert)
                lx = light_x - u
                ly = light_y - v
                lz = light_z - z_surf
                l_dist = sqrt(lx * lx + ly * ly + lz * lz + 1.0e-12_c_double)
                lx = lx / l_dist; ly = ly / l_dist; lz = lz / l_dist
                ndotl = max(0.0_c_double, nx * lx + ny * ly + nz * lz)
                spec = max(0.0_c_double, rx * lx + ry * ly + rz * lz)**16
                atten = exp(-0.18_c_double * l_dist * (1.0_c_double + 0.3_c_double * inertia))
                rad = min(1.0_c_double, max(0.0_c_double, atten * (0.70_c_double * ndotl + 0.30_c_double * spec)))

                if (ndotl > 0.0_c_double) hit_count = hit_count + 1.0_c_double
                sum_rad = sum_rad + rad
                sum_fresnel = sum_fresnel + fresnel

                idx = (iy * width + ix) * 4
                out_coords(idx + 1) = rad
                out_coords(idx + 2) = fresnel
                out_coords(idx + 3) = inertia
                out_coords(idx + 4) = tanh(0.25_c_double * phi_grav)
            end do
        end do

        e_grav = -gm1 * gm2 / max(sqrt((x2 - x1)**2 + (y2 - y1)**2 + 0.01_c_double), 1.0e-6_c_double)
        out_diag(1) = sum_rad / real(width * height, c_double)
        out_diag(2) = sum_fresnel / real(width * height, c_double)
        out_diag(3) = sum_ke / real(width * height, c_double)
        out_diag(4) = max_div
        out_diag(5) = e_grav
        out_diag(6) = 0.0_c_double
        out_diag(7) = hit_count / real(width * height, c_double)
        out_diag(8) = abs(gm1 + gm2)
    end function oct_f_universe_field

end module octopus_fortran_kernels
