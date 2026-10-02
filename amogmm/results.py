# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

import glob
import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def load(name):
    """All rows of results/<name>.json and results/<name>-*.json. A row repeated in a later
    file replaces the earlier one, so a partial rerun can be written to its own file."""
    rows = {}
    paths = glob.glob(os.path.join(RESULTS, f'{name}.json'))
    paths += sorted(glob.glob(os.path.join(RESULTS, f'{name}-*.json')))
    for path in paths:
        with open(path) as f:
            for r in json.load(f):
                rows[tuple(r[k] for k in ('dataset', 'model', 'labels', 'seed',
                                          'held_out') if k in r)] = r
    return list(rows.values())


def save(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(rows, f)
