import numpy as np

from drone_alignment.alignment.local_mesh_aligner import compute_sparse_local_displacements
from drone_alignment.alignment.displacement_field import RegularizedMeshField
from drone_alignment.config.schema import LocalMeshConfig, RoadGridConfig
from drone_alignment.quality.local_mesh_visualization import (
    generate_displacement_field_diagnostic, generate_sparse_vector_diagnostic,
)


def test_sparse_vector_diagnostic_writes_png(tmp_path):
    image = np.full((64, 64), 0.2, dtype=np.float32)
    image[:, 25:30] = 0.9
    mask = np.ones_like(image, dtype=bool)
    result = compute_sparse_local_displacements(
        image, image, mask, mask, (0.0, 0.0),
        RoadGridConfig(blur_kernel_size=3),
        LocalMeshConfig(grid_rows=2, grid_cols=2, max_search_radius_px=4,
                        min_road_points=5, min_tree_points=5, min_match_confidence=0.0,
                        ambiguity_margin=0.01),
    )
    output = generate_sparse_vector_diagnostic(image, mask, result, tmp_path / "diagnostic.png", max_dimension=128)
    assert output.exists()
    assert output.stat().st_size > 0
    field = RegularizedMeshField(
        np.array([0.0, 32.0, 63.0]), np.array([0.0, 32.0, 63.0]),
        np.full((3, 3, 2), [1.0, 0.0]),
    )
    field_output = generate_displacement_field_diagnostic(image, mask, field, tmp_path / "field.png", max_dimension=128)
    assert field_output.exists()
