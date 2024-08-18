""" One sprint means many steps that do not need to be interrupted and after which we save the data to an output file """

from functools import partial
import jax
import numpy as np
import one_step_integration as one
import advection as adv
import boundary_conditions as bc
import pressure_equations as pres_eqn
import pressure_gradient_coriolis as pres_grad
import namelist_n_constants as nl


@partial(jax.jit, static_argnames=['model_opt'])
def first_step_integration_ssprk3(phys_state, base_state, grids, model_opt):
    """ The first step using RK4 method """
    int_opt = model_opt[0]
    xi0 = phys_state
    xi1, _, sfc_others, heating = one.rk_sub_step0(xi0, xi0, base_state, grids, model_opt, nl.dt)
    if int_opt == 2:
        _, _, _, _, pip_now, _ = xi1
        xi2, _ = one.rk_sub_step_other(xi1, xi1, base_state, grids, heating, sfc_others, model_opt, nl.dt)
        xi2 = tuple(map(lambda x, y: 0.75 * x + 0.25 * y, xi0, xi2))
        xi3, _ = one.rk_sub_step_other(xi2, xi2, base_state, grids, heating, sfc_others, model_opt, nl.dt)
        (theta_next, u_next, v_next, w_next, _, qv_next) = tuple(map(lambda x, y: 1.0 / 3.0 * x + 2.0 / 3.0 * y, xi0, xi3))
    # for SSPRK3, we only want to get surface flux for the initial time

    if int_opt == 2:    # leapfrog
        theta_now, u_now, v_now, w_now, _, qv_now = phys_state
        phys_state = (theta_now, theta_next, pip_now, qv_now, qv_next, u_now, u_next, v_now, v_next, w_now, w_next)
    # for SSPRK3, just output the I.C.

    return phys_state, sfc_others


@partial(jax.jit, static_argnames=['model_opt'])
def ssprk3_sprint(phys_state, base_state, grids, model_opt):
    """ Integration using SSPRK3 method for sprint_n steps """
    for i in range(nl.sprint_n):
        xi0 = phys_state
        xi1, _, sfc_others, heating = one.rk_sub_step0(xi0, xi0, base_state, grids, model_opt, nl.dt)
        _, _, _, _, pip_now, _ = xi1
        xi2, _ = one.rk_sub_step_other(xi1, xi1, base_state, grids, heating, sfc_others, model_opt, nl.dt)
        xi2 = tuple(map(lambda x, y: 0.75*x + 0.25*y, xi0, xi2))
        xi3, _ = one.rk_sub_step_other(xi2, xi2, base_state, grids, heating, sfc_others, model_opt, nl.dt)
        (theta_next, u_next, v_next, w_next, _, qv_next) = tuple(map(lambda x, y: 1.0/3.0*x + 2.0/3.0*y, xi0, xi3))

        phys_state = (theta_next, u_next, v_next, w_next, pip_now, qv_next)

    return phys_state, sfc_others


@partial(jax.jit, static_argnames=['model_opt'])
def leapfrog_sprint(phys_state, base_state, grids, model_opt):
    """ Integrate with leapfrog method for sprint_n steps

    Here one sprint is a number of consecutive steps, between which we don't need save data to an output file. After
    one sprint, we return to the main program and save data to a file
    """
    rho0_theta0, rho0, theta0, pi0, qv0, surface_t = base_state
    x3d, y3d, z3d, x3d4u, y3d4v, z3d4w, tauh, tauf = grids
    int_opt, damp_opt, rad_opt, cor_opt, sfc_opt, pic_opt = model_opt

    for i in range(nl.sprint_n):
        theta_prev, theta_now, pip_prev, qv_prev, qv_now, u_prev, u_now, v_prev, v_now, w_prev, w_now = phys_state
        # update theta equation
        if rad_opt:
            heating_now = one.get_heating(theta_now, theta0)
        else:
            heating_now = np.zeros((nl.nx, nl.ny, nl.nz))    # ignore heating for the warm bubble case

        flow_divergence = adv.get_divergence(rho0, u_now, v_now, w_now, x3d4u, y3d4v, z3d4w)
        if sfc_opt:
            z_bottom = z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
            u_bottom = 0.5 * (u_now[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz] +
                              u_now[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz])
            v_bottom = 0.5 * (v_now[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz] +
                              v_now[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz])
            theta_bottom = theta_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
            q_bottom = qv_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
            rho_bottom = rho0[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
            # this is supposed to be density here, but we cannot know it before know pi', which requires us know surface
            # fluxes and advection tendencies. So we have to use rho0 to approximate here.
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

        theta_next = one.update_theta_leapfrog(theta_prev, rho0, theta_now, u_now, v_now, w_now, flow_divergence,
                                               sen/nl.Cp, heating_now, x3d4u, y3d4v, z3d4w)

        # update pi' equation
        theta_p_now = theta_now - theta0
        buoyancy, b8w = pres_eqn.calculate_buoyancy(theta0, theta_p_now, qv0, qv_now)
        rtt = pres_eqn.calculate_rtt(rho0_theta0, theta_now, qv_now, buoyancy)
        adv4u, adv4v, adv4w = one.prep_momentum_eqn(rho0, u_now, v_now, w_now, flow_divergence, tau_x, tau_y,
                                                    x3d, y3d, z3d, x3d4u, y3d4v, z3d4w)
        if cor_opt:
            fu, fv = pres_grad.calculate_coriolis_force(u_now, v_now)
        else:
            fu = np.zeros((nl.nx, nl.ny + 1, nl.nz))
            fv = np.zeros((nl.nx + 1, nl.ny, nl.nz))

        pip_now, info = one.solve_pres_eqn(pip_prev, rho0_theta0, pi0, rtt, u_now, v_now, w_now, adv4u, adv4v, adv4w,
                                           fu, fv, buoyancy, x3d, x3d4u, y3d, y3d4v, z3d, z3d4w)
        pip_now = one.padding3_array(pip_now)

        if pic_opt:
            pip_const = one.correct_pip_constant2(pi0, theta0, qv0, pip_prev, theta_prev, qv_prev,
                                                  theta_now, qv_now, pip_now, x3d4u, y3d4v, z3d4w)
            # pip_const is the correction constant
            pip_now = pip_now + pip_const
        else:
            pip_const = -999.9
    
        # update momentum equations
        theta_rho = theta_now * (1.0 + nl.repsm1 * qv_now)  # density potential temperature
        u_next, v_next, w_next = one.update_momentum_eqn_leapfrog(u_prev, v_prev, w_prev, pi0, pip_now, theta_rho,
                                                                  adv4u, adv4v, adv4w, fu, fv, b8w, x3d, y3d, z3d)

        # water vapor; cloud variable equations in the future
        # rho_now = one.get_rho(pi0_now, pip_now, theta_rho, qv_now)    # real microphysics may need it
        qv_next = one.update_qv_leapfrog(qv_prev, rho0, qv_now, u_now, v_now, w_now, flow_divergence, evap,
                                         x3d4u, y3d4v, z3d4w)

        # Rayleigh damping
        if damp_opt:
            u_tend, v_tend, w_tend, theta_tend = bc.rayleigh_damping(tauh, tauf, u_now, v_now, w_now, theta_now)
            u_next = u_next + one.padding3_array(u_tend * nl.dt * 2.0)
            v_next = v_next + one.padding3_array(v_tend * nl.dt * 2.0)
            w_next = w_next + one.padding3_array(w_tend * nl.dt * 2.0)
            theta_next = theta_next + one.padding3_array(theta_tend * nl.dt * 2.0)
        # Ignore for the warm buble case

        # apply Asselin filter
        u_now, v_now, w_now, theta_now, qv_now = one.asselin_filter(
                        u_prev, v_prev, w_prev, theta_prev, qv_prev,
                        u_now, v_now, w_now, theta_now, qv_now,
                        u_next, v_next, w_next, theta_next, qv_next)

        # done with one sprint
        phys_state = (theta_now, theta_next, pip_now, qv_now, qv_next, u_now, u_next, v_now, v_next, w_now, w_next)

    sfc_others = (info, pip_const, tau_x, tau_y, sen, evap, t_ref, q_ref, u10n)
    return phys_state, sfc_others
