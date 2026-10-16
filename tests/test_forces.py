"""
Regression tests for the force parameter file (data/_forces.yml).

Every entry is checked against values from an independent source (the
published parameters, or HFBTHO's coupling constants for UNEDF), so a
transcription error in the YAML fails here instead of silently changing the
physics.  The spin-orbit check guards the t4/b4p convention: a standard force
with strength W0 has C^nablaJ_0 = -3 W0/4 and C^nablaJ_1 = -W0/4.
"""

import pytest

from jax_hfbfft.core.force import Force


# Published Skyrme parameters: t0..t3 (MeV fm^...), x0..x3, W0, alpha.  The
# spin-orbit term is the standard one, b4 = b4' = W0/2 (t4 = W0 in the file).
SKYRME = {
    # Chabanat et al., Nucl. Phys. A 635, 231 (1998)
    "SLy4": (-2488.91, 486.82, -546.39, 13777.0, 0.834, -0.344, -1.0, 1.354, 123.0, 1 / 6),
    "SLy5": (-2484.88, 483.13, -549.40, 13763.0, 0.778, -0.328, -1.0, 1.267, 126.0, 1 / 6),
    "SLy6": (-2479.50, 462.18, -448.61, 13673.0, 0.825, -0.465, -1.0, 1.355, 122.0, 1 / 6),
    # Krivine, Treiner, Bohigas, Nucl. Phys. A 336, 155 (1980)
    "SkM": (-2645.0, 385.0, -120.0, 15595.0, 0.09, 0.0, 0.0, 0.0, 130.0, 1 / 6),
    # Bartel et al., Nucl. Phys. A 386, 79 (1982)
    "SkM*": (-2645.0, 410.0, -135.0, 15595.0, 0.09, 0.0, 0.0, 0.0, 130.0, 1 / 6),
    # Dobaczewski, Flocard, Treiner, Nucl. Phys. A 422, 103 (1984)
    "SkP": (-2931.70, 320.62, -337.41, 18708.97, 0.29215, 0.65318, -0.53732, 0.18103, 100.0, 1 / 6),
    # Beiner et al., Nucl. Phys. A 238, 29 (1975)
    "SIII": (-1128.75, 395.0, -95.0, 14000.0, 0.45, 0.0, 0.0, 1.0, 120.0, 1.0),
}

# Coupling constants printed by HFBTHO for its UNE0 / UNE1 functionals
# (the code the UNEDF fits were made with).
UNEDF = {
    "UNEDF0": dict(Crho0=-706.382928878428856, Crho1=240.049520427681131,
                   Crho0D=868.871771539645351, Crho1D=-69.0518957481631617,
                   Ctau0=-12.9172408208016112, Ctau1=-45.1894169426759476,
                   Cdrho0=-55.2606, Cdrho1=-55.6226,
                   CdJ0=-79.5308, CdJ1=45.6302, power=0.321955989588264435),
    "UNEDF1": dict(Crho0=-779.373008720865300, Crho1=287.722131583286796,
                   Crho0D=891.477890442349690, Crho1D=-200.587774317884879,
                   Ctau0=-0.989915057807676746, Ctau1=-33.6320970701835549,
                   Cdrho0=-45.1351310222373030, Cdrho1=-145.382167908056999,
                   CdJ0=-74.0263331764599002, CdJ1=-35.6582611147916992,
                   power=0.270018011502707600),
}

# Centre-of-mass treatment each functional was fitted with (see the zpe
# values in the header of _forces.yml).
ZPE = {"SLy4": 0, "SLy5": 0, "SLy6": 3, "SkM*": 0, "SkP": 0, "SIII": 0,
       "UNEDF0": 0, "UNEDF1": 2}


@pytest.mark.parametrize("name", sorted(SKYRME))
def test_skyrme_parameters(name):
    t0, t1, t2, t3, x0, x1, x2, x3, w0, alpha = SKYRME[name]
    f = Force.from_name(name)
    got = (f.t0, f.t1, f.t2, f.t3, f.x0, f.x1, f.x2, f.x3, f.power)
    want = (t0, t1, t2, t3, x0, x1, x2, x3, alpha)
    for g, w in zip(got, want):
        assert float(g) == pytest.approx(w, rel=1e-4, abs=1e-4)
    # standard spin-orbit: b4 = b4' = W0/2
    assert float(f.CdJ0) == pytest.approx(-0.75 * w0, rel=1e-6)
    assert float(f.CdJ1) == pytest.approx(-0.25 * w0, rel=1e-6)


@pytest.mark.parametrize("name", sorted(UNEDF))
def test_unedf_coupling_constants(name):
    f = Force.from_name(name)
    for key, want in UNEDF[name].items():
        assert float(getattr(f, key)) == pytest.approx(want, rel=1e-6, abs=1e-5), key


@pytest.mark.parametrize("name", sorted(ZPE))
def test_cm_correction_flag(name):
    assert int(Force.from_name(name).zpe) == ZPE[name]
