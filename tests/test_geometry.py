"""Grid geometry: square ids, (row, col) positions and the map layout.

The Milan grid is a 100x100 lattice whose ids run row-major from 1, and the report
locates cells on it ("5161 sits at row 51, column 60, over the Duomo"). Nothing in the
training path depends on that mapping, so without these tests the helpers that carry it
would be unexercised, and a silent off-by-one would only surface as a mislabelled figure
in the write-up. The three cells used below are the study's top-three areas, with the
positions recorded in ``configs/data.yaml``.
"""

from __future__ import annotations

import numpy as np
import pytest

from milan_traffic.config import GRID_SIDE, N_SQUARES, rowcol_to_square, square_to_rowcol
from milan_traffic.dataio import to_grid

# square_id -> (row, col), as written in configs/data.yaml for the top-three areas.
KNOWN_POSITIONS = {5161: (51, 60), 5059: (50, 58), 5259: (52, 58)}


@pytest.mark.parametrize("square_id,position", sorted(KNOWN_POSITIONS.items()))
def test_documented_positions_of_the_reported_cells(square_id, position):
    assert square_to_rowcol(square_id) == position
    assert rowcol_to_square(*position) == square_id


def test_ids_run_row_major_from_one():
    assert square_to_rowcol(1) == (0, 0)
    assert square_to_rowcol(GRID_SIDE) == (0, GRID_SIDE - 1)
    assert square_to_rowcol(GRID_SIDE + 1) == (1, 0)
    assert square_to_rowcol(N_SQUARES) == (GRID_SIDE - 1, GRID_SIDE - 1)


def test_rowcol_to_square_inverts_square_to_rowcol_everywhere():
    ids = np.arange(1, N_SQUARES + 1)
    rows, cols = np.divmod(ids - 1, GRID_SIDE)
    # The same convention the ingestion uses to write square_totals.csv.
    assert [square_to_rowcol(int(i)) for i in ids[::137]] == list(
        zip(rows[::137].tolist(), cols[::137].tolist(), strict=True)
    )
    assert all(
        rowcol_to_square(int(r), int(c)) == int(i) for i, r, c in zip(ids, rows, cols, strict=True)
    )


def test_to_grid_places_each_square_at_its_row_and_column():
    values = np.arange(N_SQUARES, dtype=np.float64)  # values[k] belongs to square k + 1
    grid = to_grid(values)

    assert grid.shape == (GRID_SIDE, GRID_SIDE)
    for square_id, (row, col) in KNOWN_POSITIONS.items():
        assert grid[row, col] == values[square_id - 1]


def test_to_grid_accepts_any_square_number_of_values():
    grid = to_grid(np.arange(9))
    np.testing.assert_array_equal(grid, np.arange(9).reshape(3, 3))


def test_to_grid_rejects_a_vector_that_is_not_a_square():
    with pytest.raises(ValueError, match="do not form a square grid"):
        to_grid(np.arange(N_SQUARES - 1))
