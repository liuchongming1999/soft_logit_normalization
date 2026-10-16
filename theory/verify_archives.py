"""Verify the frozen population-flow/orbit archives without training a network."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute-selected', action='store_true',
                        help='also regenerate three existing population cases in memory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    population = root / 'population_flow'
    manifest = json.loads((population / 'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        assert digest(population / name) == expected, name
    records = {p.stem: json.loads(p.read_text())
               for p in (population / 'results').glob('*.json')}
    assert len(records) == len(manifest['cases']) == 240
    for case in manifest['cases']:
        row = records[case['id']]
        assert all(row[key] == value for key, value in case.items())
        assert row['source_sha256'] == manifest['source_sha256']
        archive = population / 'results' / (case['id'] + '.npz')
        assert digest(archive) == row['archive_sha256'], archive.name
        with np.load(archive, allow_pickle=False) as data:
            np.testing.assert_array_equal(data['time'], manifest['times'])
            assert data['measured'].shape[0] == 101
            assert np.isfinite(data['measured']).all()
            assert np.isfinite(data['analytic']).all()

    regenerated = []
    if args.recompute_selected:
        sys.path.insert(0, str(population))
        spec = importlib.util.spec_from_file_location(
            'population_archive_runner', population / 'run_experiments.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for k, gamma in [(2, 0.), (3, .7), (10, 1.)]:
            case = next(c for c in manifest['cases']
                        if c['group'] == 'baseline' and c['k'] == k
                        and c['gamma'] == gamma and c['tau'] == 1.
                        and c['rho'] == .01)
            _, generated = runner.one_case(case)
            with np.load(population / 'results' / (case['id'] + '.npz'),
                         allow_pickle=False) as archived:
                for key, value in generated.items():
                    np.testing.assert_allclose(value, archived[key],
                                               rtol=2e-10, atol=2e-12,
                                               err_msg=case['id'] + ':' + key)
            regenerated.append(case['id'])

    binary = root / 'binary_orbits'
    summary = json.loads((binary / 'summary.json').read_text())
    assert summary['passed'] and summary['case_count'] == 432
    assert digest(binary / 'simulate.py') == summary['source_sha256']
    assert digest(binary / 'trajectories.npz') == summary['trajectory_sha256']
    with np.load(binary / 'trajectories.npz', allow_pickle=False) as data:
        positions = data['cartesian']
        radius = np.linalg.norm(positions, axis=-1)
        gamma = data['gammas']
        invariant = radius**gamma * np.abs(positions[..., 1])**(1-gamma)
        drift = float(np.max(np.abs(invariant / invariant[0] - 1)))
        assert drift < 2e-8
        assert positions.shape == (151, 432, 2)
        assert np.max(np.abs(positions - data['polar'])) < 2e-8
    moment = json.loads((binary / 'second_moment_check.json').read_text())
    assert moment['symmetric_population_groups'] == 48
    assert moment['max_relative_second_moment_transport_error'] < 1e-8
    print(json.dumps(dict(passed=True, population_cases=240, binary_orbit_cases=432,
                          symmetric_moment_groups=48,
                          regenerated_population_cases=regenerated,
                          binary_invariant_max_relative_drift=drift,
                          new_neural_network_training=False), indent=2))


if __name__ == '__main__':
    main()
