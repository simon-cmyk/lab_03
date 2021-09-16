import warnings
from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import visgeom as vg
from pylie import SE2, SO2
from shapely import geometry as sg, ops as so


def plot_se2_covariance_on_manifold(ax, dist, n=50, chi2_val=11.345, right_perturbation=True, fill_alpha=0., fill_color='lightsteelblue'):
    u, s, _ = np.linalg.svd(dist.covariance)
    scale = np.sqrt(chi2_val * s)

    x, y, z = vg.utils.generate_ellipsoid(n, pose=(u, np.zeros([3, 1])), scale=scale)

    tangent_points = np.vstack((x.flatten(), y.flatten(), z.flatten()))

    num_samples = tangent_points.shape[1]
    transl_points = np.zeros([num_samples, 2])

    if right_perturbation:
        for i in range(num_samples):
            transl_points[i, :] = (dist.mean + tangent_points[:, [i]]).translation.T
    else:
        for i in range(num_samples):
            transl_points[i, :] = (SE2.Exp(tangent_points[:, [i]]) @ dist.mean).translation.T

    p_grid = np.reshape(transl_points, [(n + 1), (n + 1), 2])
    polygons = extract_polygon_slices(p_grid)
    union = so.unary_union(polygons)
    if not union.geom_type == 'Polygon':
        warnings.warn("Could not find a closed boundary", RuntimeWarning)
        return

    ax.fill(*union.exterior.xy, alpha=fill_alpha, facecolor=fill_color)
    ax.plot(*union.exterior.xy, color=fill_color)


def extract_polygon_slices(grid_2d):
    p_a = grid_2d[:-1, :-1]
    p_b = grid_2d[:-1, 1:]
    p_c = grid_2d[1:, 1:]
    p_d = grid_2d[1:, :-1]

    quads = np.concatenate((p_a, p_b, p_c, p_d), axis=2)

    m, n, _ = grid_2d.shape
    quads = quads.reshape(((m-1) * (n-1), 4, 2))

    return [sg.Polygon(t).buffer(0.0001, cap_style=2, join_style=2) for t in quads]


def plot_2d_frame(ax, pose, **kwargs):
    """Plot the pose (R, t) in the global frame.
    Keyword Arguments
        * *alpha* -- Alpha value (transparency), default 1
        * *axis_colors* -- List of colors for each axis, default ('r', 'g')
        * *scale* -- Scale factor, default 1.0
        * *text* -- Text description plotted at pose origin, default ''
    :param ax: Current axes
    :param pose: The pose (R, t) of the local frame relative to the global frame,
        where R is a 2x2 rotation matrix and t is a 2D column vector.
    :param kwargs: See above
    :return: List of artists.
    """
    R, t = pose
    alpha = kwargs.get('alpha', 1)
    axis_colors = kwargs.get('axis_colors', ('r', 'g'))
    scale = kwargs.get('scale', 1)
    text = kwargs.get('text', '')

    artists = []

    # If R is a valid rotation matrix, the columns are the local orthonormal basis vectors in the global frame.
    for i in range(0, 2):
        axis_line = np.column_stack((t, t + R[:, i, np.newaxis] * scale))
        artists.extend(
            ax.plot(axis_line[0, :], axis_line[1, :], axis_colors[i] + '-', alpha=alpha))

    if text:
        artists.extend([ax.text(t[0, 0], t[1, 0], text, fontsize='large')])

    return artists


@dataclass
class MultivariateNormalParameters:
    mean: Any
    covariance: np.ndarray


def main():
    # TODO: Play around with different distributions.
    # TODO: Try propagating through inverese, composistion, relative element.
    T_ab = SE2((SO2(), np.array([[4], [0]])))
    Sigma_a_ab = np.array([[0.1, 0.00, 0.00], [0.00, 2, 0.4], [0.00, 0.4, 0.1]])
    T_ab_dist = MultivariateNormalParameters(T_ab, Sigma_a_ab)

    # Plot
    matplotlib.use('qt5agg')
    plt.figure()
    ax = plt.axes()

    plot_se2_covariance_on_manifold(ax, T_ab_dist, chi2_val=7.815, n=50, fill_color='green', fill_alpha=0.1)
    plot_se2_covariance_on_manifold(ax, T_ab_dist, chi2_val=4.108, n=50, fill_color='green', fill_alpha=0.1)
    plot_se2_covariance_on_manifold(ax, T_ab_dist, chi2_val=1.213, n=50, fill_color='green', fill_alpha=0.1)
    plot_2d_frame(ax, T_ab_dist.mean.to_tuple(), scale=0.5, text='$\\mathcal{F}_b$')
    plot_2d_frame(ax, SE2().to_tuple(), scale=0.5, text='$\\mathcal{F}_a$')

    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
