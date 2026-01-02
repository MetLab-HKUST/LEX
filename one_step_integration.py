""" Functions for one step of the LEX model integration """

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import advection as adv
import namelist_n_constants as nl
import radiation as rad
import pressure_equations as pres_eqn
import pressure_gradient_coriolis as pres_grad
import boundary_conditions as bc
import turbulence as turb


@partial(jax.jit, static_argnames=['model_opt', 'dt'])
def rk_sub_step0(phys_state_now, phys_state, base_state, grids, model_opt, dt):
    """ one sub-step for the SSPRK3 integration """
 
    rho0_theta0, rho0, theta0, pi0, qv0, surface_t = base_state
    x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, cc1, cc2, tauh, tauf = grids
    solver_opt, damp_opt, rad_opt, cor_opt, sfc_opt, pic_opt, turb_opt = model_opt

    if solver_opt == 1:
        theta_now0, u_now0, v_now0, w_now0, pip_prev, qv_now0 = phys_state_now
        theta_now, u_now, v_now, w_now, _, qv_now = phys_state
    elif solver_opt == 2:
        theta_now0, u_now0, v_now0, w_now0, pip_now0, qv_now0 = phys_state_now
        theta_now, u_now, v_now, w_now, pip_now, qv_now = phys_state

    # diabatic heating  
    if rad_opt:
        heating_now = get_heating(theta_now, theta0)
    else:
        heating_now = np.zeros((nl.nx, nl.ny, nl.nz))  # ignore heating for the warm bubble case

    if sfc_opt:
        z_bottom = z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
        u_bottom = 0.5 * (u_now[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz] +
                          u_now[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz])
        v_bottom = 0.5 * (v_now[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz] +
                          v_now[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz])
        theta_bottom = theta_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
        q_bottom = qv_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
        rho_bottom = rho0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
        tau_x, tau_y, sen, evap, t_ref, q_ref, u10n = bc.atm_ocn_flux(z_bottom, u_bottom, v_bottom, theta_bottom,
                                                                      q_bottom, rho_bottom, surface_t)
    else:
        tau_x = np.zeros((nl.nx, nl.ny))
        tau_y = np.zeros((nl.nx, nl.ny))
        sen = np.zeros((nl.nx, nl.ny))
        evap = np.zeros((nl.nx, nl.ny))
        t_ref = np.zeros((nl.nx, nl.ny))
        q_ref = np.zeros((nl.nx, nl.ny))
        u10n = np.zeros((nl.nx, nl.ny))

    # Turbulence model
    if turb_opt == 1 or turb_opt == 2:  # Smagorinsky
        s11, s22, s33, s12, s13, s23, deform = turb.compute_deformation(rho0, u_now, v_now, w_now, x3d, y3d, z3d, x3d4u,
                                                                        y3d4v, z3d4w)
        n2 = turb.compute_nm(theta_now, pi0, qv_now, z3d)
        # n2 = turb.compute_nm(theta_now, qv_now, qc_now, z3d)
        if turb_opt==1:
            km, kh = turb.compute_k(deform, n2, x3d4u, y3d4v, z3d4w)
            sgs_u, sgs_v, sgs_w, sgs_theta, sgs_qv = turb.compute_smag(
                rho0, km, kh, s11, s22, s33, s12, s13, s23, theta_now, qv_now, # qc_now, qr_now, nc_now, nr_now,
                x3d, y3d, z3d, x3d4u, y3d4v, z3d4w)
        elif turb_opt==2:
            kmh, kmv, khh, khv = turb.compute_k2(rho0, s11, s22, s33, s12, s13, s23, n2, x3d4u, y3d4v, z3d4w, z3d)
            sgs_u, sgs_v, sgs_w, sgs_theta, sgs_qv = turb.compute_smag2(
                rho0, kmh, kmv, khh, khv, s11, s22, s33, s12, s13, s23, theta_now, qv_now, # qc_now, qr_now, nc_now, nr_now,
                x3d, y3d, z3d, x3d4u, y3d4v, z3d4w)

        sgs_tend = (sgs_u, sgs_v, sgs_w, sgs_theta, sgs_qv)
    else:
        sgs_tend = None

    buoyancy, b8w = pres_eqn.calculate_buoyancy(theta0, theta_now, qv0, qv_now)
    rtt = pres_eqn.calculate_rtt(rho0_theta0, theta_now, qv_now, buoyancy)
    flow_divergence = adv.get_divergence(rho0, u_now, v_now, w_now, x3d4u, y3d4v, z3d4w)
    adv4u, adv4v, adv4w = prep_momentum_eqn(rho0, u_now, v_now, w_now, flow_divergence, tau_x, tau_y,
                                            x3d, y3d, z3d, x3d4u, y3d4v, z3d4w)
    if cor_opt == 1:
        fu, fv = pres_grad.calculate_coriolis_force(u_now, v_now)
    elif cor_opt == 2:
        fu, fv = pres_grad.calculate_coriolis_force_with_lspgrad(u_now, v_now)
    else:
        fu = np.zeros((nl.nx, nl.ny + 1, nl.nz))
        fv = np.zeros((nl.nx + 1, nl.ny, nl.nz))

    if solver_opt == 1:
        # update pi' equation
        pip_now, info = solve_pres_eqn(pip_prev, rho0_theta0, pi0, rtt, adv4u, adv4v, adv4w,
                                    fu, fv, buoyancy, x3d, x3d4u, y3d, y3d4v, z3d, z3d4w)
        pip_now = padding3_array(pip_now)
        if pic_opt:  # correct pi'
            pip_const = correct_pip_constant2(pi0, theta0, qv0, # qc0, qr0,
                                            pip_prev, theta_now, qv_now, # qc_prev, qr_prev,
                                            theta_now, qv_now, # qc_now, qr_now,
                                            pip_now, x3d4u, y3d4v, z3d4w)
            pip_now = pip_now + pip_const
        else:
            pip_const = -999.9

        # update momentum equations
        theta_rho = theta_now * (1.0 + nl.reps * qv_now) / (1.0 + qv_now)  # density potential temperature
        u_next, v_next, w_next, du_dt, dv_dt, dw_dt = update_momentum_eqn_euler(u_now0, v_now0, w_now0, pi0, pip_now,
                                                                                theta_rho, adv4u, adv4v, adv4w, fu, fv, b8w,
                                                                                x3d, y3d, z3d, dt)

        # Turbulence model updates
        if turb_opt == 1 or turb_opt == 2:    # Smagorinsky
            u_next = u_next + padding3u_array(sgs_u * dt)
            v_next = v_next + padding3v_array(sgs_v * dt)
            w_next = w_next + padding3_array(sgs_w * dt)
            du_dt = du_dt + sgs_u
            dv_dt = dv_dt + sgs_v
            dw_dt = dw_dt + sgs_w

    elif solver_opt == 2:
        dt_sound = nl.dt / nl.n_sound
        rk_n_sound = round(dt / dt_sound)
        # number of acoustic steps of each RK substep
        
        if turb_opt == 1 or turb_opt == 2:
            rhs_u = adv4u + fv + sgs_u 
            rhs_v = adv4v - fu + sgs_v
            rhs_w = adv4w + b8w + sgs_w
        else:
            rhs_u = adv4u + fv 
            rhs_v = adv4v - fu 
            rhs_w = adv4w + b8w         
        rhs_pi = 0.0    # placeholder. We may add diabatic terms later.
        
        theta_rho = theta_now * (1.0 + nl.reps * qv_now) / (1.0 + qv_now)  # density potential temperature
        u_next, v_next, w_next, pip_next = sound(u_now0, v_now0, w_now0, pi0, pip_now0, 
                                                 rhs_u, rhs_v, rhs_w, rhs_pi, theta_rho, rho0, 
                                                 rk_n_sound, dt_sound, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, cc1, cc2)        
        du_dt = (u_next - u_now0) / dt
        dv_dt = (v_next - v_now0) / dt 
        dw_dt = (w_next - w_now0) / dt
        d_pip_dt = (pip_next - pip_now0) / dt

    # update theta equation 
    theta_next, d_theta_dt = update_theta_euler(rho0, theta_now0, theta_now, u_now, v_now, w_now, 
                                                flow_divergence, sen / nl.Cp, heating_now, x3d4u, y3d4v, z3d4w, dt)
    # water vapor; cloud variable equations in the future
    # rho_now = get_rho(pi0, pip_now, theta_rho, qv_now)    # real microphysics may need it
    qv_next, d_qv_dt = update_qv_euler(rho0, qv_now0, qv_now, u_now, v_now, w_now,
                                       flow_divergence, evap, x3d4u, y3d4v, z3d4w, dt)

    # Turbulence model updates
    if turb_opt == 1 or turb_opt == 2:    # Smagorinsky
        theta_next = theta_next + padding3_array(sgs_theta * dt)
        d_theta_dt = d_theta_dt + sgs_theta
        qv_next = qv_next + padding3_array(sgs_qv * dt)
        d_qv_dt = d_qv_dt + sgs_qv

    # Rayleigh damping
    if damp_opt:
        u_tend, v_tend, w_tend, theta_tend = bc.rayleigh_damping(tauh, tauf, u_next, v_next, w_next, theta_next)
        u_next = u_next + padding3u_array(u_tend * dt)
        v_next = v_next + padding3v_array(v_tend * dt)
        w_next = w_next + padding3_array(w_tend * dt)
        du_dt = du_dt + u_tend
        dv_dt = dv_dt + v_tend
        dw_dt = dw_dt + w_tend
        theta_next = theta_next + padding3_array(theta_tend * dt)
        d_theta_dt = d_theta_dt + theta_tend 
        # Ignore for the warm bubble case

    if solver_opt == 1:
        phys_state = (theta_next, u_next, v_next, w_next, pip_now, qv_next)
        tends = (d_theta_dt, du_dt, dv_dt, dw_dt, d_qv_dt)
    elif solver_opt == 2:
        phys_state = (theta_next, u_next, v_next, w_next, pip_next, qv_next)
        tends = (d_theta_dt, d_pip_dt, du_dt, dv_dt, dw_dt, d_qv_dt)
        info = -999
        pip_const = -999

    sfc_etc = (info, pip_const, tau_x, tau_y, sen, evap, t_ref, q_ref, u10n)

    return phys_state, tends, sfc_etc, heating_now, sgs_tend


@partial(jax.jit, static_argnames=['model_opt', 'dt'])
def rk_sub_step_other(phys_state_now, phys_state, base_state, grids, heating, sfc_others, sgs_tend, model_opt, dt):
    """ one sub-step for the SSPRK3 integration

    Physical forcing can be calculated for the first sub-step and then be kept as constant. Heating and surface fluxes
    are the forcing we have for now.
    """
    rho0_theta0, rho0, theta0, pi0, qv0, surface_t = base_state
    x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, cc1, cc2, tauh, tauf = grids
    solver_opt, damp_opt, rad_opt, cor_opt, sfc_opt, pic_opt, turb_opt = model_opt

    if solver_opt == 1:
        theta_now0, u_now0, v_now0, w_now0, pip_prev, qv_now0 = phys_state_now
        theta_now, u_now, v_now, w_now, _, qv_now = phys_state
    elif solver_opt == 2:
        theta_now0, u_now0, v_now0, w_now0, pip_now0, qv_now0 = phys_state_now
        theta_now, u_now, v_now, w_now, pip_now, qv_now = phys_state

    heating_now = heating
    _, _, tau_x, tau_y, sen, evap, _, _, _ = sfc_others

    buoyancy, b8w = pres_eqn.calculate_buoyancy(theta0, theta_now, qv0, qv_now)
    if solver_opt == 1:
        rtt = pres_eqn.calculate_rtt(rho0_theta0, theta_now, qv_now, buoyancy)
    flow_divergence = adv.get_divergence(rho0, u_now, v_now, w_now, x3d4u, y3d4v, z3d4w)
    adv4u, adv4v, adv4w = prep_momentum_eqn(rho0, u_now, v_now, w_now, flow_divergence, tau_x, tau_y,
                                            x3d, y3d, z3d, x3d4u, y3d4v, z3d4w)
    if cor_opt == 1:
        fu, fv = pres_grad.calculate_coriolis_force(u_now, v_now)
    elif cor_opt == 2:
        fu, fv = pres_grad.calculate_coriolis_force_with_lspgrad(u_now, v_now)
    else:
        fu = np.zeros((nl.nx, nl.ny + 1, nl.nz))
        fv = np.zeros((nl.nx + 1, nl.ny, nl.nz))

    if solver_opt == 1:
        # update pi' equation
        pip_now, info = solve_pres_eqn(pip_prev, rho0_theta0, pi0, rtt, adv4u, adv4v, adv4w,
                                    fu, fv, buoyancy, x3d, x3d4u, y3d, y3d4v, z3d, z3d4w)
        pip_now = padding3_array(pip_now)
        if pic_opt:  # correct pi'
            pip_const = correct_pip_constant2(pi0, theta0, qv0, # qc0, qr0,
                                            pip_prev, theta_now, qv_now, # qc_prev, qr_prev,
                                            theta_now, qv_now, # qc_now, qr_now,
                                            pip_now, x3d4u, y3d4v, z3d4w)
            pip_now = pip_now + pip_const

        # update momentum equations
        theta_rho = theta_now * (1.0 + nl.reps * qv_now) / (1.0 + qv_now)  # density potential temperature
        u_next, v_next, w_next, du_dt, dv_dt, dw_dt = update_momentum_eqn_euler(u_now0, v_now0, w_now0, pi0, pip_now,
                                                                                theta_rho, adv4u, adv4v, adv4w, fu, fv, b8w,
                                                                                x3d, y3d, z3d, dt)

        # Turbulence model
        if turb_opt == 1 or turb_opt == 2:    # Smagorinsky
            sgs_u, sgs_v, sgs_w, sgs_theta, sgs_qv = sgs_tend        
            u_next = u_next + padding3u_array(sgs_u * dt)
            v_next = v_next + padding3v_array(sgs_v * dt)
            w_next = w_next + padding3_array(sgs_w * dt)
            du_dt = du_dt + sgs_u
            dv_dt = dv_dt + sgs_v
            dw_dt = dw_dt + sgs_w

    if solver_opt == 2:
        dt_sound = nl.dt / nl.n_sound
        rk_n_sound = round(dt / dt_sound)
        # number of acoustic steps of each RK substep

        if turb_opt == 1 or turb_opt == 2:
            # turbulence model used
            sgs_u, sgs_v, sgs_w, sgs_theta, sgs_qv = sgs_tend           
            rhs_u = adv4u + fv + sgs_u 
            rhs_v = adv4v - fu + sgs_v
            rhs_w = adv4w + b8w + sgs_w
        else:
            rhs_u = adv4u + fv 
            rhs_v = adv4v - fu 
            rhs_w = adv4w + b8w 
        rhs_pi = 0.0   # placeholder. We may add diabatic terms later.
        
        theta_rho = theta_now * (1.0 + nl.reps * qv_now) / (1.0 + qv_now)  # density potential temperature
        u_next, v_next, w_next, pip_next = sound(u_now0, v_now0, w_now0, pi0, pip_now0, 
                                                 rhs_u, rhs_v, rhs_w, rhs_pi, theta_rho, rho0, 
                                                 rk_n_sound, dt_sound, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, cc1, cc2)
        du_dt = (u_next - u_now0) / dt
        dv_dt = (v_next - v_now0) / dt 
        dw_dt = (w_next - w_now0) / dt
        d_pip_dt = (pip_next - pip_now0) / dt
        
    # update theta equation
    theta_next, d_theta_dt = update_theta_euler(rho0, theta_now0, theta_now, u_now, v_now, w_now,
                                                flow_divergence, sen / nl.Cp, heating_now, x3d4u, y3d4v, z3d4w, dt)
    # water vapor; cloud variable equations in the future
    # rho_now = get_rho(pi0, pip_now, theta_rho, qv_now)    # real microphysics may need it
    qv_next, d_qv_dt = update_qv_euler(rho0, qv_now0, qv_now, u_now, v_now, w_now,
                                       flow_divergence, evap, x3d4u, y3d4v, z3d4w, dt)
    # Turbulence model
    if turb_opt == 1 or turb_opt == 2:    # Smagorinsky
        theta_next = theta_next + padding3_array(sgs_theta * dt)
        d_theta_dt = d_theta_dt + sgs_theta 
        qv_next = qv_next + padding3_array(sgs_qv * dt)
        d_qv_dt = d_qv_dt + sgs_qv        
        
    # Rayleigh damping
    if damp_opt:
        u_tend, v_tend, w_tend, theta_tend = bc.rayleigh_damping(tauh, tauf, u_next, v_next, w_next, theta_next)
        u_next = u_next + padding3u_array(u_tend * dt)
        v_next = v_next + padding3v_array(v_tend * dt)
        w_next = w_next + padding3_array(w_tend * dt)
        du_dt = du_dt + u_tend
        dv_dt = dv_dt + v_tend
        dw_dt = dw_dt + w_tend
        theta_next = theta_next + padding3_array(theta_tend * dt)
        # Ignore for the warm buble case

    if solver_opt == 1:
        phys_state = (theta_next, u_next, v_next, w_next, pip_now, qv_next)
        tends = (d_theta_dt, du_dt, dv_dt, dw_dt, d_qv_dt)
    elif solver_opt == 2:
        phys_state = (theta_next, u_next, v_next, w_next, pip_next, qv_next)
        tends = (d_theta_dt, d_pip_dt, du_dt, dv_dt, dw_dt, d_qv_dt)

    return phys_state, tends


def update_theta_euler(rho0_now, theta_now0, theta_now, u_now, v_now, w_now, flow_divergence, sfc_flux, heating, x3d4u, y3d4v,
                       z3d4w, dt):
    """ Update theta to get the next-step values """
    d_theta_dt = compute_theta_tendency(rho0_now, theta_now, u_now, v_now, w_now, flow_divergence, sfc_flux, heating,
                                        x3d4u, y3d4v, z3d4w)
    theta_next = theta_now0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + d_theta_dt * dt
    theta_next3 = padding3_array(theta_next)
    return theta_next3, d_theta_dt


def update_qv_euler(rho0_now, qv_now0, qv_now, u_now, v_now, w_now, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w, dt):
    """ Update theta to get the next-step values """
    d_qv_dt = compute_qv_tendency(rho0_now, qv_now, u_now, v_now, w_now, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w)
    qv_next = qv_now0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + d_qv_dt * dt
    qv_next3 = padding3_array(qv_next)
    return qv_next3, d_qv_dt


def solve_pres_eqn(pip_prev, rho0_theta0, pi0, rtt, adv4u, adv4v, adv4w, fu, fv, buoyancy,
                   x3d, x3d4u, y3d, y3d4v, z3d, z3d4w):
    """ Solve the Poisson-like equation for pressure perturbations, pi\' (pip) """
    rhs, rhs_adv, rhs_cor, rhs_buoy, rhs_pres = pres_eqn.rhs_of_pressure_equation(rho0_theta0, pi0, rtt, adv4u,
                                                                                  adv4v, adv4w, fu, fv, buoyancy,
                                                                                  x3d, x3d4u, y3d, y3d4v, z3d4w)

    tol = 1.0e-4  # the tolerance level needs to be tested and tuned.
    atol = 1.0e-8
    # using previous step pi\' as the first guess x0
    # pip, info = jax.scipy.sparse.linalg.gmres(
    #     lambda beta:pres_eqn.laplace_of_pressure(x3d, x3d4u, y3d, y3d4v, z3d, z3d4w, rtt, beta),
    #     rhs, x0=pip_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
    #     tol=tol, atol=atol, maxiter=100, solve_method='incremental')
    pip, info = jax.scipy.sparse.linalg.bicgstab(
        lambda beta: pres_eqn.laplace_of_pressure(x3d, x3d4u, y3d, y3d4v, z3d, z3d4w, rtt, beta),
        rhs, x0=pip_prev[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz],
        tol=tol, atol=atol, maxiter=100)

    return pip, info


def correct_pip_constant(rho0_now, pip_now, x3d4u, y3d4v, z3d4w):
    """ Add a constant to ensure pi' sum is zero """
    numerator = -space_integration(rho0_now * pip_now, x3d4u, y3d4v, z3d4w)
    denominator = space_integration(rho0_now, x3d4u, y3d4v, z3d4w)
    return numerator / denominator


def correct_pip_constant2(pi0_now, theta0_now, qv0_now, # qc0_now, qr0_now,
                          pip_prev, theta_prev, qv_prev, # qc_prev, qr_prev,
                          theta_now, qv_now, # qc_now, qr_now,
                          pip_now, x3d4u, y3d4v, z3d4w):
    """ Add a constant to ensure total mass change is zero (to the accuracy of linearized approximation) """

    # theta_r0 = theta0_now * (1.0 + nl.repsm1 * qv0_now - qc0_now - qr0_now)  # base state density potential temperature
    # theta_r1 = theta_prev * (1.0 + nl.repsm1 * qv_prev - qc_prev - qr_prev) - theta_r0
    # theta_r2 = theta_now * (1.0 + nl.repsm1 * qv_now - qc_now - qr_now) - theta_r0
    theta_r0 = theta0_now * (1.0 + nl.repsm1 * qv0_now)  # base state density potential temperature
    theta_r1 = theta_prev * (1.0 + nl.repsm1 * qv_prev) - theta_r0
    theta_r2 = theta_now * (1.0 + nl.repsm1 * qv_now) - theta_r0
    beta = nl.p00 * nl.Cv / nl.Rd ** 2 * pi0_now ** ((nl.Cv - nl.Rd) / nl.Rd) / theta_r0
    alpha = nl.p00 * pi0_now ** (nl.Cv / nl.Rd) / nl.Rd / theta_r0 / theta_r0
    numerator = space_integration(beta * (pip_prev - pip_now), x3d4u, y3d4v, z3d4w) + space_integration(
        alpha * (theta_r2 - theta_r1), x3d4u, y3d4v, z3d4w)
    denominator = space_integration(beta, x3d4u, y3d4v, z3d4w)
    # denominator_a = jnp.where((denominator < 1.0e-6) & (denominator >= 0.0), 1.0e-6, denominator)
    # denominator_b = jnp.where((denominator_a > -1.0e-6) & (denominator_a < 0.0), -1.0e-6, denominator_a)
    return numerator/denominator


def sound(u_now, v_now, w_now, pi0_now, pip_now, rhs_u, rhs_v, rhs_w, rhs_pi, theta_rho, rho0, rk_n_sound, dt_sound, 
          x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, cc1, cc2):
    """ Acoustic steps to advance u, v, w, and pi' """

    piadv = pi0_now + pip_now
    pi0_now0 = pi0_now
    pip_now0 = pip_now

    for i in range(rk_n_sound):
        pres_grad4u, pres_grad4v, pres_grad4w = pres_grad.pressure_gradient_force(pi0_now, pip_now, theta_rho, x3d, y3d, z3d)
        du_dt = pres_grad4u + rhs_u
        u_next = u_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + du_dt * dt_sound
        dv_dt = pres_grad4v + rhs_v
        v_next = v_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + dv_dt * dt_sound
        dw_dt = pres_grad4w + rhs_w
        w_next = w_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + dw_dt * dt_sound
        u_next = padding3u_array(u_next)
        v_next = padding3v_array(v_next)
        w_next = padding3_array(w_next)
        w_next = bc.set_w_bc(w_next)

        divergence = adv.get_divergence2(u_next, v_next, w_next, x3d4u, y3d4v, z3d4w)
        adv4pi = adv.advection_pi(rho0, pi0_now0, pip_now0, u_next, v_next, w_next, x3d, y3d, z3d, cc1, cc2)
        d_pip_dt = -nl.Rd / nl.Cv * (
            piadv[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * divergence[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz]
            ) + adv4pi + rhs_pi 
        pip_next = pip_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + 1.1*d_pip_dt * dt_sound
        # the 1.1 factor is added to dampen divergence, gamma_d = 0.1 is a typical choice in CM1 and WRF
        pip_next = padding3_array(pip_next)

        u_now = u_next
        v_now = v_next
        w_now = w_next
        pip_now = pip_next
    # end of acoustic steps

    return u_next, v_next, w_next, pip_next


def update_momentum_eqn_euler(u_now0, v_now0, w_now0, pi0_now, pip_now, theta_now, adv4u, adv4v, adv4w, fu, fv, b8w, x3d,
                              y3d, z3d, dt):
    """ Update momentum to get the next-step values """
    pres_grad4u, pres_grad4v, pres_grad4w = pres_grad.pressure_gradient_force(pi0_now, pip_now, theta_now, x3d, y3d,
                                                                              z3d)
    du_dt = adv4u + pres_grad4u + fv
    u_next = u_now0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + du_dt * dt
    dv_dt = adv4v + pres_grad4v - fu
    v_next = v_now0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + dv_dt * dt
    dw_dt = adv4w + pres_grad4w + b8w
    w_next = w_now0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] + dw_dt * dt
    u_next3 = padding3u_array(u_next)
    v_next3 = padding3v_array(v_next)
    w_next3 = padding3_array(w_next)
    w_next3 = bc.set_w_bc(w_next3)
    return u_next3, v_next3, w_next3, du_dt, dv_dt, dw_dt


def prep_momentum_eqn(rho0, u, v, w, flow_divergence, u_sfc_flux, v_sfc_flux, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w):
    """ Calculate advection tendencies for momentum equations """
    weps = 1.0e-16  # set based on CM1 subroutines
    adv4u = adv.advection_u(rho0, u, v, w, weps, flow_divergence, u_sfc_flux, x3d, y3d4v, z3d4w)
    adv4v = adv.advection_v(rho0, u, v, w, weps, flow_divergence, v_sfc_flux, x3d4u, y3d, z3d4w)
    adv4w = adv.advection_w(rho0, u, v, w, weps, flow_divergence, x3d4u, y3d4v, z3d)
    return adv4u, adv4v, adv4w


def compute_theta_tendency(rho0, theta, u, v, w, flow_divergence, sfc_flux, heating, x3d4u, y3d4v, z3d4w):
    """ Compute the tendency for theta """
    weps = 1.0e-17  # chosen based on CM1 weps for theta equation
    adv_theta = adv.advection_scalar(rho0, theta, u, v, w, weps, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w)
    tendency = adv_theta + heating
    return tendency


def compute_qv_tendency(rho0, qv, u, v, w, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w):
    """ Compute the tendency for qv """
    weps = 1.0e-18  # CM1 weps for qv equation is 1e-20, but that leads to NaN in LEX, so we use this larger value
    adv_qv = adv.advection_scalar(rho0, qv, u, v, w, weps, flow_divergence, sfc_flux, x3d4u, y3d4v, z3d4w)
    tendency = adv_qv  # + microphysics in the future
    return tendency


def get_heating(theta, theta0_ic):
    """ Compute the heating rate for theta

    In a full complexity LES, heating should include the effects of radiation and microphysics. Here we use Newtonian
    relaxation for the time being.
    """
    heating = rad.newton_radiation(theta, theta0_ic)  # unit: K/s
    # heating here is Hm/Cp/pi0, the tendency of theta due to heating, where Hm has the unit J/s/kg
    return heating


def space_integration(scalar, x3d4u, y3d4v, z3d4w):
    """ Compute the spatial integration of a scalar """
    dx = x3d4u[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] - x3d4u[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy,
                                                                           nl.ngz:-nl.ngz]
    dy = y3d4v[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz:-nl.ngz] - y3d4v[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1),
                                                                           nl.ngz:-nl.ngz]
    dz = z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz + 1:-nl.ngz] - z3d4w[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy,
                                                                           nl.ngz:-(nl.ngz + 1)]
    return jnp.sum(scalar[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz:-nl.ngz] * dx * dy * dz)


def get_rho(pi0, pip, theta, qv):
    """ Compute density """
    pi = pi0 + pip
    t = pi * theta  # temperature
    p = pi ** (nl.Cp / nl.Rd) * nl.p00
    rho = p / nl.Rd / ((1.0 + nl.repsm1 * qv) * t)
    return rho


def padding3_array(arr):
    """ Padding an array with three ghost points on each side """
    arr_x = jnp.concatenate((arr[-nl.ngx:, :, :], arr, arr[0:nl.ngx, :, :]), axis=0)
    arr_xy = jnp.concatenate((arr_x[:, -nl.ngx:, :], arr_x, arr_x[:, 0:nl.ngx, :]), axis=1)
    x_size, y_size, _ = jnp.shape(arr_xy)
    bottom = jnp.reshape(arr_xy[:, :, 0], (x_size, y_size, 1))
    top = jnp.reshape(arr_xy[:, :, -1], (x_size, y_size, 1))
    arr_xyz = jnp.concatenate((bottom, arr_xy, top), axis=2)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.
    return arr_xyz


def padding3u_array(arr):
    """ Padding an array at U point with three ghost points on each side """
    # arr0 = arr.at[-(nl.ngx+1), :, :].set(arr[nl.ngx, :, :])  # make sure b.c. is same U
    arr_x = jnp.concatenate((arr[-(nl.ngx+1):-1, :, :], arr, arr[1:nl.ngx+1, :, :]), axis=0)
    arr_xy = jnp.concatenate((arr_x[:, -nl.ngy:, :], arr_x, arr_x[:, 0:nl.ngy, :]), axis=1)
    x_size, y_size, _ = jnp.shape(arr_xy)
    bottom = jnp.reshape(arr_xy[:, :, 0], (x_size, y_size, 1))
    top = jnp.reshape(arr_xy[:, :, -1], (x_size, y_size, 1))
    arr_xyz = jnp.concatenate((bottom, arr_xy, top), axis=2)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.
    return arr_xyz


def padding3v_array(arr):
    """ Padding an array at V point with three ghost points on each side """
    # arr0 = arr.at[:, -(nl.ngy+1), :].set(arr[:, nl.ngy, :])  # make sure b.c is same V
    arr_x = jnp.concatenate((arr[-nl.ngx:, :, :], arr, arr[0:nl.ngx, :, :]), axis=0)
    arr_xy = jnp.concatenate((arr_x[:, -(nl.ngy+1):-1, :], arr_x, arr_x[:, 1:nl.ngy+1, :]), axis=1)
    x_size, y_size, _ = jnp.shape(arr_xy)
    bottom = jnp.reshape(arr_xy[:, :, 0], (x_size, y_size, 1))
    top = jnp.reshape(arr_xy[:, :, -1], (x_size, y_size, 1))
    arr_xyz = jnp.concatenate((bottom, arr_xy, top), axis=2)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.
    return arr_xyz


def padding3_0_array(arr):
    """ Padding an array with three ghost points on each side """
    x0s = jnp.zeros((nl.ngx, arr.shape[1], arr.shape[2]))
    arr_x = jnp.concatenate((x0s, arr, x0s), axis=0)
    y0s = jnp.zeros((arr_x.shape[0], nl.ngy, arr_x.shape[2]))
    arr_xy = jnp.concatenate((y0s, arr_x, y0s), axis=1)
    x_size, y_size, _ = jnp.shape(arr_xy)
    bottom = jnp.reshape(arr_xy[:, :, 0], (x_size, y_size, 1))
    top = jnp.reshape(arr_xy[:, :, -1], (x_size, y_size, 1))
    arr_xyz = jnp.concatenate((bottom, arr_xy, top), axis=2)
    # The ghost points at the bottom and top are more like placeholders, without practical use for now.
    return arr_xyz


def padding2_array(arr):
    """ Padding a 2D array with three ghost points on each side """
    arr_x = jnp.concatenate((arr[-nl.ngx:, :,], arr, arr[0:nl.ngx, :,]), axis=0)
    arr_xy = jnp.concatenate((arr_x[:, -nl.ngy:], arr_x, arr_x[:, 0:nl.ngy]), axis=1)
    return arr_xy
