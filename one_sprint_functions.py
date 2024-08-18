""" One sprint means many steps that do not need to be interrupted and after which we save the data to an output file """

from functools import partial
import jax
import one_step_integration as one
import namelist_n_constants as nl


@partial(jax.jit, static_argnames=['model_opt'])
def first_step_integration_ssprk3(phys_state, base_state, grids, model_opt):
    """ The first step using SSPRK4 method, mainly needed by the leapfrog scheme """
    int_opt = model_opt[0]
    xi0 = phys_state
    xi1, _, sfc_others, heating = one.rk_sub_step0(xi0, xi0, base_state, grids, model_opt, nl.dt/2.0)
    if int_opt == 2:
        _, _, _, _, pip_now, _ = xi1
        xi2, _ = one.rk_sub_step_other(xi1, xi1, base_state, grids, heating, sfc_others, model_opt, nl.dt/2.0)
        xi3, _ = one.rk_sub_step_other(xi2, xi2, base_state, grids, heating, sfc_others, model_opt, nl.dt/2.0)
        xi3 = tuple(map(lambda x, y: 2.0/3.0*x + 1.0/3.0*y, xi0, xi3))
        xi4, _ = one.rk_sub_step_other(xi3, xi3, base_state, grids, heating, sfc_others, model_opt, nl.dt/2.0)
        theta_next, u_next, v_next, w_next, _, qv_next = xi4
    # for SSPRK3, we only want to get surface flux for the initial time

    if int_opt == 2:    # leapfrog
        theta_now, u_now, v_now, w_now, _, qv_now = phys_state
        phys_state = (theta_now, theta_next, pip_now, qv_now, qv_next, u_now, u_next, v_now, v_next, w_now, w_next)
    # for SSPRK3, just output the I.C.

    return phys_state, sfc_others


@partial(jax.jit, static_argnames=['model_opt'])
def ssprk3_sprint(phys_state, base_state, grids, model_opt):
    """ Integration using SSPRK3 method for sprint_n steps

    Four-stage SSPRK4 based on Durran (2010) Page 56.
    """
    for i in range(nl.sprint_n):
        xi0 = phys_state
        xi1, _, sfc_others, heating = one.rk_sub_step0(xi0, xi0, base_state, grids, model_opt, nl.dt/2.0)
        _, _, _, _, pip_now, _ = xi1
        xi2, _ = one.rk_sub_step_other(xi1, xi1, base_state, grids, heating, sfc_others, model_opt, nl.dt/2.0)
        xi3, _ = one.rk_sub_step_other(xi2, xi2, base_state, grids, heating, sfc_others, model_opt, nl.dt/2.0)
        xi3 = tuple(map(lambda x, y: 2.0/3.0*x + 1.0/3.0*y, xi0, xi3))
        xi4, _ = one.rk_sub_step_other(xi3, xi3, base_state, grids, heating, sfc_others, model_opt, nl.dt/2.0)
        theta_next, u_next, v_next, w_next, _, qv_next = xi4
        phys_state = (theta_next, u_next, v_next, w_next, pip_now, qv_next)

    return phys_state, sfc_others


@partial(jax.jit, static_argnames=['model_opt'])
def leapfrog_sprint(phys_state, base_state, grids, model_opt):
    """ Integrate with leapfrog method for sprint_n steps

    Here one sprint is a number of consecutive steps, between which we don't need save data to an output file. After
    one sprint, we return to the main program and save data to a file
    """

    for i in range(nl.sprint_n):
        theta_prev, theta_now, pip_prev, qv_prev, qv_now, u_prev, u_now, v_prev, v_now, w_prev, w_now = phys_state

        phys_state_prev = (theta_prev, u_prev, v_prev, w_prev, pip_prev, qv_prev)
        phys_state_now = (theta_now, u_now, v_now, w_now, pip_prev, qv_now)

        phys_state_next, _, sfc_others, _ = one.rk_sub_step0(phys_state_prev, phys_state_now, base_state, grids, model_opt, 2.0*nl.dt)

        theta_next, u_next, v_next, w_next, pip_now, qv_next = phys_state_next

        # apply Asselin filter
        u_now, v_now, w_now, theta_now, qv_now = one.asselin_filter(
                        u_prev, v_prev, w_prev, theta_prev, qv_prev,
                        u_now, v_now, w_now, theta_now, qv_now,
                        u_next, v_next, w_next, theta_next, qv_next)

        phys_state = (theta_now, theta_next, pip_now, qv_now, qv_next, u_now, u_next, v_now, v_next, w_now, w_next)

    return phys_state, sfc_others
