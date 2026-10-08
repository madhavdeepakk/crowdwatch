import numpy as np
import pytest

from crowdwatch.geometry import (Homography, points_in_polygon, polygon_area,
                                 projection_along_segment, signed_distance_to_line)


def test_polygon_area_square_and_triangle():
    assert polygon_area([(0, 0), (4, 0), (4, 3), (0, 3)]) == pytest.approx(12)
    assert polygon_area([(0, 0), (4, 0), (0, 3)]) == pytest.approx(6)
    assert polygon_area([(0, 0), (1, 1)]) == 0


def test_points_in_polygon_handles_concave_shapes():
    # an L shape: the notch at top right is outside
    poly = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    pts = [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5), (3, 3)]
    assert points_in_polygon(pts, poly).tolist() == [True, True, True, False, False]


def test_points_in_polygon_empty_input():
    assert points_in_polygon(np.zeros((0, 2)), [(0, 0), (1, 0), (1, 1)]).shape == (0,)


def test_signed_distance_left_is_positive():
    # walking up the screen (towards smaller y), screen-left is on your left
    d = signed_distance_to_line([(-1, 5), (1, 5)], a=(0, 10), b=(0, 0))
    assert d[0] > 0 > d[1]
    assert abs(d[0]) == pytest.approx(1)


def test_projection_along_segment():
    s = projection_along_segment([(0, 0), (5, 3), (10, 0), (15, 0)], a=(0, 0), b=(10, 0))
    assert s.tolist() == pytest.approx([0, 0.5, 1, 1.5])


def test_homography_recovers_scale_and_area():
    image = [(0, 0), (200, 0), (200, 100), (0, 100)]
    floor = [(0, 0), (20, 0), (20, 10), (0, 10)]
    h = Homography(image, floor)
    assert h.to_floor([(100, 50)])[0] == pytest.approx([10, 5])
    assert polygon_area(h.to_floor(image)) == pytest.approx(200)


def test_homography_handles_perspective():
    # a trapezoid in the image that is a square on the floor
    image = [(40, 20), (160, 20), (200, 100), (0, 100)]
    floor = [(0, 0), (10, 0), (10, 10), (0, 10)]
    h = Homography(image, floor)
    assert np.allclose(h.to_floor(image), floor, atol=1e-6)


def test_homography_rejects_too_few_points():
    with pytest.raises(ValueError):
        Homography([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)])
