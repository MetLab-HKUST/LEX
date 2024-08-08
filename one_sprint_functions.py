""" One sprint means many steps that do not need to be interrupted and after which we save the data to an output file """

import jax
import one_step_integration as one
import advection as adv
import boundary_conditions as bc
import pressure_equations as pres_eqn
import pressure_gradient_coriolis as pres_grad
import namelist_n_constants as nl


@jax.jit
def first_step_integration(rho0_theta0_now, rho0_now, theta0_now, theta0_ic, theta_now, qv_now, u_now, v_now, w_now,
                           pi0_now, pip_now, surface_t, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w):
    """ The first step using Euler method """
    # update rho0*theta0 equation
    heating_now = one.get_heating(theta_now, theta0_ic)
    rho0_theta0_next, rho0_next, theta0_next, pi0_next, rho0_theta0_heating_now, rho0_theta0_tend_now = one.update_rho0_theta0_euler(
        rho0_theta0_now, rho0_now, u_now, v_now, w_now, heating_now, x3d4u, y3d4v, z3d4w)
    # update theta equation
    flow_divergence = adv.get_divergence(rho0_now, u_now, v_now, w_now, x3d4u, y3d4v, z3d4w)
    z_bottom = z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
    u_bottom = 0.5 * (u_now[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz] +
                      u_now[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz])
    v_bottom = 0.5 * (v_now[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz] +
                      v_now[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz])
    theta_bottom = theta_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
    q_bottom = qv_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
    rho_bottom = rho0_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
    tau_x, tau_y, sen, evap, t_ref, q_ref, u10n = bc.atm_ocn_flux(z_bottom, u_bottom, v_bottom, theta_bottom, q_bottom,
                                                                  rho_bottom, surface_t)
    theta_next = one.update_theta_euler(rho0_now, theta_now, u_now, v_now, w_now, flow_divergence, sen / nl.Cp,
                                        heating_now, x3d4u, y3d4v, z3d4w)
    # update pi' equation
    theta_p_now = theta_now - theta0_now
    buoyancy, b8w = pres_eqn.calculate_buoyancy(theta0_now, theta_p_now, qv_now)
    rtt = pres_eqn.calculate_rtt(rho0_theta0_now, theta_now, buoyancy)
    adv4u, adv4v, adv4w = one.prep_momentum_eqn(rho0_now, u_now, v_now, w_now, flow_divergence, tau_x, tau_y,
                                                x3d, y3d, z3d, x3d4u, y3d4v, z3d4w)
    fu, fv = pres_grad.calculate_coriolis_force(u_now, v_now)
    pip_now = one.solve_pres_eqn(pip_now, rho0_theta0_now, pi0_now, rtt, u_now, v_now, w_now, adv4u, adv4v, adv4w,
                                 fu, fv, buoyancy, rho0_theta0_heating_now, rho0_theta0_heating_now,
                                 rho0_theta0_tend_now, rho0_theta0_tend_now, x3d, x3d4u, y3d, y3d4v, z3d4w)
    pip_const = one.correct_pip_constant(rho0_now, theta0_now, pi0_now, rho0_next, theta0_next, pi0_next,
                                         pi0_now, pip_now, x3d4u, y3d4v, z3d4w)
    # pip_const is the correction constant for conserving energy, based on Durran 2008.
    pip_now = pip_now + pip_const

    # update momentum equations
    u_next, v_next, w_next = one.update_momentum_eqn_euler(u_now, v_now, w_now, pi0_now, pip_now, theta_now,
                                                           adv4u, adv4v, adv4w, fu, fv, b8w, x3d, y3d, z3d)
    # cloud variable equations
    qv_next = one.update_qv_euler(rho0_now, qv_now, u_now, v_now, w_now, flow_divergence, evap, x3d4u, y3d4v, z3d4w)

    return (rho0_theta0_next, rho0_next, theta0_next, pi0_next, rho0_theta0_heating_now, rho0_theta0_tend_now,
            tau_x, tau_y, sen, evap, t_ref, q_ref, u10n, theta_next, qv_next, pip_now, pip_const,
            u_next, v_next, w_next)


@jax.jit
def leapfrog_sprint(rho0_theta0_prev, rho0_theta0_now, rho0_prev, rho0_now, theta0_prev, theta0_now, theta0_ic,
                    theta_prev, theta_now, pi0_prev, pi0_now, pip_prev,
                    qv_prev, qv_now, u_prev, u_now, v_prev, v_now, w_prev, w_now,
                    rho0_theta0_heating_prev, rho0_theta0_tend_prev,
                    surface_t, x3d, y3d, z3d, x3d4u, y3d4v, z3d4w):
    """ Integrate with leapfrog method for sprint_n steps

     Here one sprint is a number of consecutive steps, between which we don't need save data to an output file. After
     one sprint, we return to the main program and save data to a file
     """

    for i in range(nl.sprint_n):
        # update rho0*theta0 equation
        heating_now = one.get_heating(theta_now, theta0_ic)
        rho0_theta0_next, rho0_next, theta0_next, pi0_next, rho0_theta0_heating_now, rho0_theta0_tend_now = one.update_rho0_theta0_leapfrog(
            rho0_theta0_prev, rho0_theta0_now, rho0_now, u_now, v_now, w_now, heating_now, x3d4u, y3d4v, z3d4w)
        # update theta equation
        flow_divergence = adv.get_divergence(rho0_now, u_now, v_now, w_now, x3d4u, y3d4v, z3d4w)
        z_bottom = z3d[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
        u_bottom = 0.5 * (u_now[nl.ngx:-(nl.ngx + 1), nl.ngy:-nl.ngy, nl.ngz] +
                          u_now[nl.ngx + 1:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz])
        v_bottom = 0.5 * (v_now[nl.ngx:-nl.ngx, nl.ngy:-(nl.ngy + 1), nl.ngz] +
                          v_now[nl.ngx:-nl.ngx, nl.ngy + 1:-nl.ngy, nl.ngz])
        theta_bottom = theta_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
        q_bottom = qv_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
        rho_bottom = rho0_now[nl.ngx:-nl.ngx, nl.ngy:-nl.ngy, nl.ngz]
        # this is supposed to be density here, but we cannot know it before know pi', which requires us know surface
        # fluxes and advection tendencies. So we have to use rho0 to approximate here.
        tau_x, tau_y, sen, evap, t_ref, q_ref, u10n = bc.atm_ocn_flux(z_bottom, u_bottom, v_bottom, theta_bottom,
                                                                      q_bottom, rho_bottom, surface_t)
        theta_next = one.update_theta_leapfrog(theta_prev, rho0_now, theta_now, u_now, v_now, w_now, flow_divergence,
                                               sen/nl.Cp, heating_now, x3d4u, y3d4v, z3d4w)
        # update pi' equation
        theta_p_now = theta_now - theta0_now
        buoyancy, b8w = pres_eqn.calculate_buoyancy(theta0_now, theta_p_now, qv_now)
        rtt = pres_eqn.calculate_rtt(rho0_theta0_now, theta_now, buoyancy)
        adv4u, adv4v, adv4w = one.prep_momentum_eqn(rho0_now, u_now, v_now, w_now, flow_divergence, tau_x, tau_y,
                                                    x3d, y3d, z3d, x3d4u, y3d4v, z3d4w)
        fu, fv = pres_grad.calculate_coriolis_force(u_now, v_now)
        pip_now = one.solve_pres_eqn(pip_prev, rho0_theta0_now, pi0_now, rtt, u_now, v_now, w_now, adv4u, adv4v, adv4w,
                                     fu, fv, buoyancy, rho0_theta0_heating_prev, rho0_theta0_heating_now,
                                     rho0_theta0_tend_prev, rho0_theta0_tend_now, x3d, x3d4u, y3d, y3d4v, z3d4w)
        pip_const = one.correct_pip_constant(rho0_prev, theta0_prev, pi0_prev, rho0_next, theta0_next, pi0_next,
                                             pi0_now, pip_now, x3d4u, y3d4v, z3d4w)
        # pip_const is the correction constant for conserving energy, based on Durran 2008.
        pip_now = pip_now + pip_const

        # update momentum equations
        u_next, v_next, w_next = one.update_momentum_eqn_leapfrog(u_prev, v_prev, w_prev, pi0_now, pip_now, theta_now,
                                                                  adv4u, adv4v, adv4w, fu, fv, b8w, x3d, y3d, z3d)

        # cloud variable equations
        rho_now = one.get_rho(pi0_now, pip_now, theta_now, qv_now)
        qv_next = one.update_qv_leapfrog(qv_prev, rho0_now, qv_now, u_now, v_now, w_now, flow_divergence, evap,
                                         x3d4u, y3d4v, z3d4w)
        # apply Asselin filter
        u_now, v_now, w_now, rho0_theta0_now, rho0_now, theta0_now, theta_now, qv_now = one.asselin_filter(
                        u_prev, v_prev, w_prev, rho0_theta0_prev, theta_prev, qv_prev,
                        u_now, v_now, w_now, rho0_theta0_now, theta_now, qv_now,
                        u_next, v_next, w_next, rho0_theta0_next, theta_next, qv_next, z3d4w)
        # replace 'prev' and 'now' by 'now‘ and 'next'
        theta_prev = theta_now
        theta_now = theta_next
        theta0_prev = theta0_now
        theta0_now = theta0_next
        rho0_theta0_prev = rho0_theta0_now
        rho0_theta0_now = rho0_theta0_next
        rho0_prev = rho0_now
        rho0_now = rho0_next
        rho0_theta0_heating_prev = rho0_theta0_heating_now
        rho0_theta0_tend_prev = rho0_theta0_tend_now
        pi0_prev = pi0_now
        pi0_now = pi0_next
        pip_prev = pip_now
        u_prev = u_now
        u_now = u_next
        v_prev = v_now
        v_now = v_next
        w_prev = w_now
        w_now = w_next
        qv_prev = qv_now
        qv_now = qv_next
        rho_prev = rho_now
    # done with one sprint

    return (rho0_theta0_prev, rho0_theta0_now, rho0_prev, rho0_now, theta0_prev, theta0_now,
            theta_prev, theta_now, pi0_prev, pi0_now, pip_prev,
            qv_prev, qv_now, u_prev, u_now, v_prev, v_now, w_prev, w_now,
            rho0_theta0_heating_prev, rho0_theta0_tend_prev,
            rho_prev, pip_const, tau_x, tau_y, sen, evap, t_ref, q_ref, u10n)
