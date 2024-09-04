from ex2_mean_pose import draw_random_poses
import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
from pylie import SO3, SE3

"""Exercise 3 - Propagating uncertainty in backprojection"""


def backproject(u, z, f, c):
    """The inverse camera model backprojects a pixel u with depth z out to a 3D point in the camera coordinate frame.

    :param u: Pixel points, a 2xn matrix of pixel position column vectors
    :param z: Depths, an n-vector of depths.
    :param f: Focal lengths, a 2x1 column vector [fu, fv].T
    :param c: Principal point, a 2x1 column vector [cu, cv].T
    :return: A 3xn matrix of column vectors representing the backprojected points
    """
    # TODO 1: Implement backprojection.
    fu, fv = f.flatten()
    cu, cv = c.flatten()

    x_c = np.array([[ z[i]*(ui[0] - cu)/fu, z[i]* (ui[1] - cv)/fv, z[i]] for i, ui in enumerate(u.T)])

    return x_c.T


def main():
    # Define camera parameters.
    f = np.array([[100, 100]]).T  # "Focal lengths"  [fu, fv].T
    c = np.array([[50, 50]]).T  # Principal point [cu, cv].T

    # Define camera pose distribution.
    R_w_c = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    t_w_c = np.array([[1, 0, 0]]).T
    mean_T_w_c = SE3((SO3(R_w_c), t_w_c))
    Sigma_T_w_c = np.diag(np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]) ** 2)

    # Define pixel position distribution.
    mean_u = np.array([[50, 50]]).T
    Sigma_u = np.diag(np.array([1, 1]) ** 2)

    # Define depth distribution.
    mean_z = 10
    var_z = 0.1 ** 2

    #########
    fu, fv = f.flatten()
    cu, cv = c.flatten()
    K = np.array([[fu, 0,  cu],
                  [0,  fv, cv],
                  [0,  0,  1]])
    ##########

    # TODO 2: Approximate the distribution of the 3D point in world coordinates:
    u_, v_ = mean_u.flatten()
    x_c_mean = mean_z * np.array([(u_ - cu)/fu, (v_ - cv)/fv, 1])
    
    print(f'x_c_mean {x_c_mean}')
    # TODO 2: Propagate the expected point
    x_w_mean = mean_T_w_c * x_c_mean.reshape((3, 1))
    print(f'x_w_mean: {x_w_mean}')
    # TODO 2: Propagate the uncertainty.
    
    Jac_f_T_wc = SE3.jac_action_Xx_wrt_X(SE3((SO3(R_w_c), np.zeros((3, 1)))), x_c_mean)
    Jac_f_u = SO3(R_w_c) * np.array([[mean_z / fu, 0], [0, mean_z / fv], [0, 0]])
    Jac_f_z = SO3(R_w_c) * (np.linalg.inv(K) @ np.vstack([mean_u, 1])).reshape((3, 1))

    print(f'J_f_T_wc: {Jac_f_T_wc}')
    print(f'J_f_u: {Jac_f_u}')
    print(f'J_f_z: {Jac_f_z}')


    cov_x_w = Jac_f_T_wc @ Sigma_T_w_c @ Jac_f_T_wc.T
    cov_x_w += Jac_f_u @ Sigma_u @ Jac_f_u.T
    cov_x_w += var_z * Jac_f_z @ Jac_f_z.T

    print(cov_x_w)

    # Simulate points from the true distribution.
    num_draws = 1000
    random_poses = draw_random_poses(mean_T_w_c, Sigma_T_w_c, num_draws)
    random_u = np.random.multivariate_normal(mean_u.flatten(), Sigma_u, num_draws).T
    random_z = np.random.normal(mean_z, np.sqrt(var_z), num_draws)
    rand_x_c = backproject(random_u, random_z, f, c)
    rand_x_w = np.zeros((3, num_draws))
    for i in range(num_draws):
        rand_x_w[:, [i]] = random_poses[i] * rand_x_c[:, [i]]

    # Plot result.
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot camera poses.
    vg.plot_pose(ax, mean_T_w_c.to_tuple())

    # Plot simulated points.
    ax.plot(rand_x_w[0, :], rand_x_w[1, :], rand_x_w[2, :], 'k.', alpha=0.1)

    # Plot the estimated mean pose.

    # print(x_w_mean)
    # print(cov_x_w)

    vg.plot_covariance_ellipsoid(ax, x_w_mean, cov_x_w)

    # Show figure.
    vg.plot.axis_equal(ax)
    plt.show()


if __name__ == "__main__":
    main()
