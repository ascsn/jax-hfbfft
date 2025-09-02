import jax
import jax.numpy as jnp
from reader import read_yaml
from params import init_params
from forces import init_forces
from grids import init_grids
from densities import init_densities
from meanfield import init_meanfield
from levels import init_levels
from static import init_static, statichf
from coulomb import init_coulomb
from moment import init_moment
from energies import init_energies
from inout import sp_properties

from test import *

jax.config.update('jax_enable_x64', True)
#jax.config.update('jax_platform_name', 'cpu')

config = read_yaml('_config.yml')

params = init_params(**config.get('params', {}))
forces = init_forces(params, **config.get('force', {}))
grids = init_grids(params, **config.get('grids', {}))
densities = init_densities(grids)
meanfield = init_meanfield(grids)
levels = init_levels(grids, **config.get('levels', {}))
forces, static = init_static(forces, levels, **config.get('static', {}))
coulomb = init_coulomb(grids)
energies = init_energies()
moment = init_moment(jnp.array([0, 0, 0]))

levels.psi = levels.psi.at[...].set(load5d('psi'))

# print(f"ehf: {energies.ehf}")
# print(f"ehfprev: {energies.ehfprev}")
# print(f"efluct1prev: {energies.efluct1prev}")
# print(f"efluct2prev: {energies.efluct2prev}")
# print(f"serr: {static.serr}")

# print(f"psi sum: {jnp.sum(levels.psi)}")
# print(f"levels.wocc: {jnp.sum(levels.wocc)}")
# print(f"psi sum: {jnp.sum(levels.psi)}")

# print(f"static.x0dmp: {static.x0dmp}")


# coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static = statichf(
#     coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static
# )

import timeit
start_time = timeit.default_timer()

coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static = statichf(
    coulomb, densities, energies, forces, grids, levels, meanfield, moment, params, static
)
jax.block_until_ready(coulomb)
jax.block_until_ready(densities)
jax.block_until_ready(energies)
jax.block_until_ready(forces)
jax.block_until_ready(grids)
jax.block_until_ready(levels)
jax.block_until_ready(meanfield)
jax.block_until_ready(moment)
jax.block_until_ready(params)
jax.block_until_ready(static)

end_time = timeit.default_timer()

total_time = end_time - start_time

print(f"total_time: {total_time}")

# print(f"sp_energy: {levels.sp_energy}")
# print(f"static.x0dmp: {static.x0dmp}")
# print(params.iteration)
print("done")













'''
params.iteration = 4

levels.wocc = levels.wocc.at[...].set(1.0)
levels.wstates = levels.wstates.at[...].set(1.0)
levels.wguv = levels.wguv.at[...].set(0.0)
levels.pairwg = levels.pairwg.at[...].set(1.0)
forces.tbcs = True

levels.psi = levels.psi.at[...].set(load5d('diagstep_psi'))
levels.hampsi = levels.hampsi.at[...].set(load5d('diagstep_hampsi'))
levels.hmfpsi = levels.hampsi.at[...].set(load5d('diagstep_hmfpsi'))

# levels.psi = levels.psi.at[...].set(1)
# levels.hampsi = levels.hampsi.at[...].set(0.4)
# levels.hmfpsi = levels.hampsi.at[...].set(1.2)

sp_energy = load1d_real('diagstep_sp_energy', 132)

energies, levels, static = diagstep(energies, forces, grids, levels, static, True, True)


print(jnp.max(jnp.abs(sp_energy - levels.sp_energy)))
# 4.902744876744691e-13



import timeit
start_time = timeit.default_timer()
for i in range(100):
    energies, levels, static = diagstep(energies, forces, grids, levels, static, i % 2 == 0, i % 2 == 0)
    jax.block_until_ready(energies)
    jax.block_until_ready(static)
    jax.block_until_ready(levels)
end_time = timeit.default_timer()

total_time = end_time - start_time
average_time = total_time / 100

print(average_time)
# 0.030063006047159432
'''





















'''
levels.wocc = levels.wocc.at[...].set(1.0)
levels.wstates = levels.wstates.at[...].set(1.0)
levels.wguv = levels.wguv.at[...].set(0.0)
levels.pairwg = levels.pairwg.at[...].set(1.0)


levels.psi = levels.psi.at[...].set(0.5)
densities.rho = densities.rho.at[...].set(1)
densities.chi = densities.chi.at[...].set(0.5)
densities.tau = densities.tau.at[...].set(1.3)
densities.current = densities.current.at[...].set(1.3)
densities.sdens = densities.sdens.at[...].set(1.1)
densities.sodens = densities.sodens.at[...].set(1.2)

# levels.psi = levels.psi.at[...].set(load5d('psi'))
# densities.rho = densities.rho.at[...].set(load4d_real('rho'))
# densities.chi = densities.chi.at[...].set(load4d_real('chi'))
# densities.tau = densities.tau.at[...].set(load4d_real('tau'))
# densities.current = densities.current.at[...].set(load5d_real('current'))
# densities.sdens = densities.sdens.at[...].set(load5d_real('sdens'))
# densities.sodens = densities.sodens.at[...].set(load5d_real('sodens'))

# rho = load4d_real('res_rho')
# chi = load4d_real('res_chi')
# tau = load4d_real('res_tau')
# current = load5d_real('res_current')
# sdens = load5d_real('res_sdens')
# sodens = load5d_real('res_sodens')

rho = jnp.ones((2,48,48,48))
chi = jnp.ones((2,48,48,48))
tau = jnp.ones((2,48,48,48))
current = jnp.ones((2,3,48,48,48))
sdens = jnp.ones((2,3,48,48,48))
sodens = jnp.ones((2,3,48,48,48))

res = add_density(densities, grids, levels)

print(jnp.max(jnp.abs(rho - res.rho)))
print(jnp.max(jnp.abs(chi - res.chi)))
print(jnp.max(jnp.abs(tau - res.tau)))
print(jnp.max(jnp.abs(current - res.current)))
print(jnp.max(jnp.abs(sdens - res.sdens)))
print(jnp.max(jnp.abs(sodens - res.sodens)))

#8.326672684688674e-17
#0.0
#1.6653345369377348e-16
#1.734723475976807e-17
#8.131516293641283e-18
#1.1275702593849246e-17

import timeit
start_time = timeit.default_timer()
for _ in range(500):
    res = add_density(densities, grids, levels)
    jax.block_until_ready(res)
end_time = timeit.default_timer()

total_time = end_time - start_time
average_time = total_time / 500

print(average_time)
'''







'''
levels.psi = levels.psi.at[...].set(load5d('psi'))

# sp_spin = load2d_real('sp_spin', 3, 132)
# sp_orbital = load2d_real('sp_orbital', 3, 132)
# sp_kinetic = load1d_real('sp_kinetic', 132)
# sp_parity = load1d_real('sp_parity', 132)

sp_spin = jnp.ones((132, 3))
sp_orbital = jnp.ones((132, 3))
sp_kinetic = jnp.ones((132))
sp_parity = jnp.ones((132))

res = sp_properties(forces, grids, levels, moment)

print("sp_spin")
print(jnp.max(jnp.abs(res.sp_spin - sp_spin)))

print("sp_orbital")
print(jnp.max(jnp.abs(res.sp_orbital - sp_orbital)))

print("sp_kinetic")
print(jnp.max(jnp.abs(res.sp_kinetic - sp_kinetic)))

print("sp_parity")
print(jnp.max(jnp.abs(res.sp_parity - sp_parity)))




#relative_error = jnp.abs(res.sp_spin.flatten() - sp_spin.flatten()) / jnp.abs(sp_spin.flatten())

#print(jnp.max(relative_error))

#print(sp_spin[5,:])
#print(res.sp_spin[5,:])


import timeit
start_time = timeit.default_timer()
for _ in range(100):
    res = sp_properties(forces, grids, levels, moment)
    jax.block_until_ready(res)
end_time = timeit.default_timer()

total_time = end_time - start_time
average_time = total_time / 100

print(average_time)
'''










'''
# densities.rho = densities.rho.at[...].set(load4d_real('rho'))
# densities.sodens = densities.sodens.at[...].set(load5d_real('sodens'))
# densities.tau = densities.tau.at[...].set(load4d_real('tau'))
# densities.sdens = densities.sdens.at[...].set(load5d_real('sdens'))
# densities.current = densities.current.at[...].set(load5d_real('current'))
# densities.chi = densities.chi.at[...].set(load4d_real('chi'))


densities.rho = densities.rho.at[...].set(1.0)
densities.sodens = densities.sodens.at[...].set(2)
densities.tau = densities.tau.at[...].set(3)
densities.sdens = densities.sdens.at[...].set(4)
densities.current = densities.current.at[...].set(4)
densities.chi = densities.chi.at[...].set(2)


# upot = load4d_real('upot')
# divaq = load4d_real('divaq')
# wlspot = load5d_real('wlspot')
# dbmass = load5d_real('dbmass')
# spot = load5d_real('spot')
# wcoul = load3d_real('wcoul')
# bmass = load4d_real('bmass')
# aq = load5d_real('aq')
# v_pair = load4d_real('v_pair')

upot = jnp.ones((2,48,48,48), dtype=jnp.float64)
divaq = jnp.ones((2,48,48,48), dtype=jnp.float64)
wlspot = jnp.ones((2,3,48,48,48), dtype=jnp.float64)
dbmass = jnp.ones((2,3,48,48,48), dtype=jnp.float64)
spot = jnp.ones((2,48,3,48,48), dtype=jnp.float64)
wcoul = jnp.ones((48,48,48), dtype=jnp.float64)
bmass = jnp.ones((2,48,48,48), dtype=jnp.float64)
aq = jnp.ones((2,3,48,48,48), dtype=jnp.float64)
v_pair = jnp.ones((2,48,48,48), dtype=jnp.float64)


res1, res2 = skyrme(coulomb, densities, forces, grids, meanfield, params, static)

print(jnp.max(jnp.abs(upot - res1.upot)))
print(jnp.max(jnp.abs(bmass - res1.bmass)))
print(jnp.max(jnp.abs(wlspot - res1.wlspot)))
print(jnp.max(jnp.abs(aq - res1.aq)))
print(jnp.max(jnp.abs(divaq - res1.divaq)))
print(jnp.max(jnp.abs(dbmass - res1.dbmass)))
print(jnp.max(jnp.abs(v_pair - res1.v_pair)))
# print(jnp.max(jnp.abs(wcoul - res2)))

import timeit
start_time = timeit.default_timer()
for _ in range(100):
    res1, res2 = skyrme(coulomb, densities, forces, grids, meanfield, params, static)
    jax.block_until_ready(res1)
    jax.block_until_ready(res2)
end_time = timeit.default_timer()

total_time = end_time - start_time
average_time = total_time / 100

print(average_time)
# 0.0021801677842934928
'''


'''
grstep
levels.wocc = levels.wocc.at[...].set(1.0)
levels.wstates = levels.wstates.at[...].set(1.0)
levels.wguv = levels.wguv.at[...].set(0.0)
levels.pairwg = levels.pairwg.at[...].set(1.0)

forces.tbcs = True

levels.psi = levels.psi.at[...].set(load5d('psi'))
levels.lagrange = levels.psi.at[...].set(load5d('lagrange'))

levels.sp_energy = levels.sp_energy.at[...].set(load1d_real('sp_energy', 132))
levels.sp_efluct1 = levels.sp_efluct1.at[...].set(load1d_real('sp_efluct1', 132))

denerg = load1d_real('denerg', 1)

meanfield.upot = meanfield.upot.at[...].set(load4d_real('upot'))
meanfield.bmass = meanfield.bmass.at[...].set(load4d_real('bmass'))
meanfield.divaq = meanfield.divaq.at[...].set(load4d_real('divaq'))
meanfield.v_pair = meanfield.v_pair.at[...].set(load4d_real('v_pair'))
meanfield.aq = meanfield.aq.at[...].set(load5d_real('aq'))
meanfield.spot = meanfield.spot.at[...].set(load5d_real('spot'))
meanfield.wlspot = meanfield.wlspot.at[...].set(load5d_real('wlspot'))
meanfield.dbmass = meanfield.dbmass.at[...].set(load5d_real('dbmass'))

#psin = load4d('psin')
#res_psin = load4d('res_psin')
#res_psi_mf = load4d('res_psi_mf')
#delpsi = load4d('delpsi')

#res1, res2, res3 = hpsi01(grids, meanfield, 0, 1.0, 0.0, psin)


#print(jnp.max(jnp.abs(res_psin - res1)))
#print(jnp.max(jnp.abs(res_psi_mf - res2)))
#print(jnp.max(jnp.abs(delpsi - res3)))
#print(jnp.max(res_psin))
#print(jnp.max(res1))
res_psi = load5d('res_psi')
res_delpsi = load5d('res_delpsi')
res_hmfpsi = load5d('res_hmfpsi')
res_hampsi = load5d('res_hampsi')

size = 132
nst = jnp.arange(0, 132)
spe_mf = jnp.arange(0, 132)
iq = jnp.zeros(size, dtype=int)
iq = iq.at[size // 2:].set(1)
static.x0dmp = 1.0908270059034080
# print(forces.h2ma)
forces.h2ma = 20.735530853271484
params.iteration = 4



res_levels, res_static = grstep(forces, grids, levels, meanfield, params, static)

print(jnp.max(jnp.abs(res_psi - res_levels.psi)))
print(jnp.max(jnp.abs(res_delpsi - res_levels.delpsi)))
print(jnp.max(jnp.abs(res_hmfpsi - res_levels.hmfpsi)))
print(jnp.max(jnp.abs(res_hampsi - res_levels.hampsi)))


print(res_static.sumflu)
print(res_static.delesum)
print(res_levels.sp_energy)

# psin, hampsi, spe_mf_new, denerg, hmfpsi, delpsi = grstep_vmap(params, forces, grids, meanfield, levels, static, nst, levels.isospin, levels.sp_energy, levels.psi, levels.lagrange)



# print(jnp.sum(res_psi))
# print(jnp.sum(psin))
# print(jnp.max(jnp.abs(psin - res_psi)))

# print(jnp.sum(res_hmfpsi))
# print(jnp.sum(hmfpsi))
# print(jnp.max(jnp.abs(hmfpsi - res_hmfpsi)))

# print(jnp.sum(res_psi))
# print(jnp.sum(psin))
# print(jnp.max(jnp.abs(res_delpsi - delpsi)))
# print(jnp.max(jnp.abs(res_psi - psin)))

# print(jnp.sum(denerg))

import timeit
start_time = timeit.default_timer()
for i in range(150):
    params.iteration = i + 10
    res_levels, res_static = grstep(forces, grids, levels, meanfield, params, static)
    jax.block_until_ready(res_levels)
    jax.block_until_ready(res_static)
end_time = timeit.default_timer()

total_time = end_time - start_time
average_time = total_time / 150

print(average_time)
'''
