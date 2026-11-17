"""End-to-end CLI runs: the analysis block after a paired static, and the strength function of a kicked TDHF run."""

from pathlib import Path

import yaml

from jax_hfbfft.cli import run_calculation

CONFIG = Path(__file__).resolve().parents[1] / "config.yml"


def _run(tmp_path, protons, neutrons, npsi, force, iterations, **sections):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg['nucleus'] = dict(protons=protons, neutrons=neutrons, name=f"Z{protons}N{neutrons}")
    cfg['force'].update(force)
    cfg['grid'] = dict(nx=12, ny=12, nz=12, dx=1.2, dy=1.2, dz=1.2)
    cfg['basis'] = dict(npsi_neutron=npsi[0], npsi_proton=npsi[1])
    cfg['initialization'].update(ho_length_x=1.8, ho_length_y=1.8, ho_length_z=1.8)
    cfg['iteration'].update(max_iterations=iterations, convergence_threshold=1e-20)
    cfg['output'].update(output_dir=str(tmp_path), print_interval=10**6, sinfo_interval=10**6,
                         save_densities=False)
    cfg.update(sections)
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(cfg))
    run_calculation(str(path))
    return next(p for p in tmp_path.iterdir() if p.is_dir())


def test_analysis_block(tmp_path):
    run_dir = _run(tmp_path, 8, 12, (20, 12), dict(ipair=6, v0_neutron=350, v0_proton=350, state_cutoff=[0.0, 0.0]),
                   200, analysis=dict(particle_number_projection=True, n_gauge=9, form_factors_q=[0.5, 1.0]))
    text = (run_dir / "analysis.res").read_text()
    assert "PAV total energy" in text and "P(N):" in text and "R_ch" in text
    assert len([l for l in text.splitlines() if l.strip().startswith(("0.5", "1.0"))]) == 2


def test_tdhf_writes_the_strength_function(tmp_path):
    run_dir = _run(tmp_path, 8, 8, (8, 8), dict(ipair=0), 300,
                   dynamics=dict(enabled=True, method='tdhf', dt=0.2, steps=200, diag_interval=5,
                                 kick=dict(type='monopole', eta=1e-3)))
    rows = (run_dir / "strength.dat").read_text().splitlines()
    assert rows[0].startswith("#") and len(rows) > 100
