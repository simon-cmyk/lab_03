import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
from pylie import SO3, SE3

"""Exercise 1 - Linear interpolation of poses on the manifold"""


def interpolate_lie_element(alpha, X_1: SE3, X_2: SE3):
    """Perform linear interpolation on the manifold

    :param alpha: A scalar interpolation factor in [0, 1]
    :param X_1: First element
    :param X_2: Second element
    :return: The interpolated element
    """

    return X_1 + alpha * (X_2 - X_1)   # Mock implementation, do something else!

def separate_interpolation():
    T_1 = SE3((SO3.from_roll_pitch_yaw(np.pi / 4, 0, np.pi / 2), np.array([[1, 1, 1]]).T))

    # Define the second pose.
    T_2 = SE3((SO3.from_roll_pitch_yaw(-np.pi / 6, np.pi / 4, np.pi / 2), np.array([[1, 4, 2]]).T))

    # Plot the interpolation.
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot the poses.
    vg.plot_pose(ax, T_1.to_tuple())
    vg.plot_pose(ax, T_2.to_tuple())

    # Plot the interpolated poses.
    for alpha in np.linspace(0, 1, 20):
        T = interpolate_lie_element(alpha, T_1, T_2)
        vg.plot_pose(ax, T.to_tuple(), alpha=0.1, axis_colors=['b', 'b', 'b'])
    
    R_0 = SO3.from_roll_pitch_yaw(np.pi / 4, 0, -np.pi / 2)
    R_1 = SO3.from_roll_pitch_yaw(-np.pi / 6, np.pi / 4, np.pi / 2)

    t0 = np.array([[1, 1, 1]]).T
    t1 = np.array([[1, 4, 2]]).T

    for alpha in np.linspace(0, 1, 20):
        R = interpolate_lie_element(alpha, R_0, R_1)
        t = interpolate_lie_element(alpha, t0, t1)
        vg.plot_pose(ax, (R.matrix, t), alpha=0.1, axis_colors=['r', 'r', 'r'])
    plt.title('Interpolation of poses, red: separate interpolation, blue: combined interpolation')
    # Show figure.
    vg.plot.axis_equal(ax)
    plt.show()

def extrapolating():
    T_1 = SE3((SO3.from_roll_pitch_yaw(np.pi / 4, 0, np.pi / 2), np.array([[1, 1, 1]]).T))

    # Define the second pose.
    T_2 = SE3((SO3.from_roll_pitch_yaw(-np.pi / 6, np.pi / 4, np.pi / 2), np.array([[1, 4, 2]]).T))

    # Plot the interpolation.
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot the poses.
    vg.plot_pose(ax, T_1.to_tuple())
    vg.plot_pose(ax, T_2.to_tuple())

    # Plot the interpolated poses.
    for alpha in np.linspace(-1, 2, 20):
        T = interpolate_lie_element(alpha, T_1, T_2)
        vg.plot_pose(ax, T.to_tuple(), alpha=0.1, axis_colors=['b', 'b', 'b'])
    
    R_0 = SO3.from_roll_pitch_yaw(np.pi / 4, 0, -np.pi / 2)
    R_1 = SO3.from_roll_pitch_yaw(-np.pi / 6, np.pi / 4, np.pi / 2)

    t0 = np.array([[1, 1, 1]]).T
    t1 = np.array([[1, 4, 2]]).T

    for alpha in np.linspace(-1, 2, 20):
        R = interpolate_lie_element(alpha, R_0, R_1)
        t = interpolate_lie_element(alpha, t0, t1)
        vg.plot_pose(ax, (R.matrix, t), alpha=0.1, axis_colors=['r', 'r', 'r'])
    plt.title('Extrapolation of poses, red: separate, blue: combined')
    # Show figure.
    vg.plot.axis_equal(ax)
    plt.show()


def main():
    # Define the first pose.
    T_1 = SE3((SO3.from_roll_pitch_yaw(np.pi / 4, 0, np.pi / 2), np.array([[1, 1, 1]]).T))

    # Define the second pose.
    T_2 = SE3((SO3.from_roll_pitch_yaw(-np.pi / 6, np.pi / 4, np.pi / 2), np.array([[1, 4, 2]]).T))

    # Plot the interpolation.
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot the poses.
    vg.plot_pose(ax, T_1.to_tuple())
    vg.plot_pose(ax, T_2.to_tuple())

    # Plot the interpolated poses.
    for alpha in np.linspace(0, 1, 20):
        T = interpolate_lie_element(alpha, T_1, T_2)
        vg.plot_pose(ax, T.to_tuple(), alpha=0.1)

    # Show figure.
    vg.plot.axis_equal(ax)
    plt.show()

    separate_interpolation()

    extrapolating()

if __name__ == "__main__":
    main()
