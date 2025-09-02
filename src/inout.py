import jax
import jax.numpy as jnp
from levels import cdervx01, cdervy01, cdervz01


def sp_properties_helper(pst, dx, dy, dz, xx, yy, zz):
    psx, psw = cdervx01(dx, pst)
    psy, ps2 = cdervy01(dy, pst)
    psw = psw.at[...].add(ps2)
    psz, ps3 = cdervz01(dz, pst)
    psw = psw.at[...].add(ps3)

    cc = jnp.array([
        jnp.sum(
            jnp.real(pst) * (yy * jnp.imag(psz) - zz * jnp.imag(psy)) +
            jnp.imag(pst) * (zz * jnp.real(psy) - yy * jnp.real(psz))
        ),
        jnp.sum(
            jnp.real(pst) * (zz * jnp.imag(psx) - xx * jnp.imag(psz)) +
            jnp.imag(pst) * (xx * jnp.real(psz) - zz * jnp.real(psx))
        ),
        jnp.sum(
            jnp.real(pst) * (xx * jnp.imag(psy) - yy * jnp.imag(psx)) +
            jnp.imag(pst) * (yy * jnp.real(psx) - xx * jnp.real(psy))
        )
    ])

    kin = -jnp.sum(
        jnp.real(pst) * jnp.real(psw) + jnp.imag(pst) * jnp.imag(psw)
    )

    xpar = jnp.sum(
        jnp.real(pst) * jnp.real(pst[:, ::-1, ::-1, ::-1]) +
        jnp.imag(pst) * jnp.imag(pst[:, ::-1, ::-1, ::-1])
    )

    ss = jnp.array([
        jnp.sum(
            jnp.real(jnp.conjugate(pst[0,...]) * pst[1,...]) +
            jnp.real(jnp.conjugate(pst[1,...]) * pst[0,...])
        ),
        jnp.sum(
            jnp.real(jnp.conjugate(pst[0,...]) * pst[1,...] * (0.0 - 1.0j)) +
            jnp.real(jnp.conjugate(pst[1,...]) * pst[0,...] * (0.0 + 1.0j))
        ),
        jnp.sum(
            jnp.real(jnp.conjugate(pst[0,...]) * pst[0,...]) -
            jnp.real(jnp.conjugate(pst[1,...]) * pst[1,...])
        )
    ])

    return ss, cc, kin, xpar

sp_properties_helper_vmap = jax.vmap(
    sp_properties_helper,
    in_axes=(0, None, None, None, None, None, None)
)

@jax.jit
def sp_properties(forces, grids, levels, moment):
    xx = (grids.x - moment.cmtot[0])[jnp.newaxis,:,jnp.newaxis,jnp.newaxis]
    yy = (grids.y - moment.cmtot[1])[jnp.newaxis,jnp.newaxis,:,jnp.newaxis]
    zz = (grids.z - moment.cmtot[2])[jnp.newaxis,jnp.newaxis,jnp.newaxis,:]

    ss, cc, kin, xpar = sp_properties_helper_vmap(
        levels.psi, grids.dx, grids.dy, grids.dz, xx, yy, zz
    )

    levels.sp_spin = levels.sp_spin.at[...].set(
        0.5 * grids.wxyz * ss
    )
    levels.sp_orbital = levels.sp_orbital.at[...].set(
        grids.wxyz * cc
    )
    levels.sp_kinetic = levels.sp_kinetic.at[...].set(
        grids.wxyz * jnp.where(levels.isospin == 0, forces.h2m[0], forces.h2m[1]) * kin
    )
    levels.sp_parity = levels.sp_parity.at[...].set(
        grids.wxyz * xpar
    )

    return levels
