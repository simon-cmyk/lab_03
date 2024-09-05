from ex2_mean_pose import draw_random_poses
import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
from pylie import SO3, SE3
from scipy.linalg import block_diag

"""Exercise 3 - Propagating uncertainty in backprojection"""


def backproject(u, z, f, c):
    """The inverse camera model backprojects a pixel u with depth z out to a 3D point in the camera coordinate frame.

    :param u: Pixel points, a 2xn matrix of pixel position column vectors
    :param z: Depths, an n-vector of depths.
    :param f: Focal lengths, a 2x1 column vector [fu, fv].T
    :param c: Principal point, a 2x1 column vector [cu, cv].T
    :return: A 3xn matrix of column vectors representing the backprojected points
    """
    # Done 1: Implement backprojection.
    fu, fv = f.flatten()
    cu, cv = c.flatten()

    if isinstance(z, int):
        return np.array([ z*(u[0] - cu)/fu, z* (u[1] - cv)/fv, z])
        
    x_c = np.array([[ z[i]*(ui[0] - cu)/fu, z[i]* (ui[1] - cv)/fv, z[i]] for i, ui in enumerate(u.T)])

    return x_c.T

def find_sigma_points(mean, cov, alpha, beta, kappa, mean_T_w_c):
    """Compute sigma points for the given mean and covariance.
        The state is vee(T), u, z, (But we are not using the vee(T) in this case (directly using the mean_T_w_c, because of implementation issue))
    """
    n = len(mean)
    lam = alpha**2 * (n + kappa) - n
    sqrt_cov = np.sqrt(n + lam) * np.linalg.cholesky(cov)
    
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = mean
    
    poses = np.full(2* n + 1, mean_T_w_c, dtype=object)
    poses[0] = mean_T_w_c
    for i in range(n):
        poses[1 + i] = mean_T_w_c + sqrt_cov[:6, i].reshape((6, 1))
        poses[1 + i + n] = mean_T_w_c + -1 *sqrt_cov[:6, i].reshape((6, 1))
        sigma_points[i + 1][6:] = mean[6:] + sqrt_cov[6:, i]
        sigma_points[i + 1 + n][6:] = mean[6:] - sqrt_cov[6:, i]
    
    # Compute weights
    W_m, W_c = np.zeros(2 * n + 1), np.zeros(2 * n + 1)
    W_m[0] = lam / (n + lam)
    W_c[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    W_m[1:] = W_c[1:] = 1 / (2 * (n + lam))
    
    return sigma_points, W_m, W_c, poses


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
    Sigma_T_w_c = np.diag(np.array([0.1, 0.1, 0.1, 0.01, 0.2, 0.01]) ** 2)

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

    # Done 2: Approximate the distribution of the 3D point in world coordinates:

    x_c_mean = backproject(mean_u.flatten(), mean_z, f, c)

    # Done Propagate the expected point
    x_w_mean = mean_T_w_c * x_c_mean.reshape((3, 1))
    # Done: Propagate the uncertainty.
    
    Jac_f_T_wc = SE3.jac_action_Xx_wrt_X(SE3((SO3(R_w_c), np.zeros((3, 1)))), x_c_mean)
    Jac_f_u = SO3(R_w_c) * np.array([[mean_z / fu, 0], [0, mean_z / fv], [0, 0]])
    Jac_f_z = SO3(R_w_c) * (np.linalg.inv(K) @ np.vstack([mean_u, 1])).reshape((3, 1))

    print(f'J_f_T_wc: {Jac_f_T_wc}')
    print(f'J_f_u: {Jac_f_u}')
    print(f'J_f_z: {Jac_f_z}')


    cov_x_w = Jac_f_T_wc @ Sigma_T_w_c @ Jac_f_T_wc.T
    cov_x_w += Jac_f_u @ Sigma_u @ Jac_f_u.T
    cov_x_w += var_z * Jac_f_z @ Jac_f_z.T

    # print(cov_x_w)

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

    # Plot camera pose.
    vg.plot_pose(ax, mean_T_w_c.to_tuple())

    # Plot simulated points.
    ax.plot(rand_x_w[0, :], rand_x_w[1, :], rand_x_w[2, :], 'g.', alpha=0.05)

    # Plot the estimated mean pose.
    vg.plot_covariance_ellipsoid(ax, x_w_mean, cov_x_w, color='r')

    # Estimate the covariance of the simulated points.
    mean_ = rand_x_w.mean(keepdims=True, axis=1)
    cov_ = np.cov(rand_x_w)
    vg.plot_covariance_ellipsoid(ax, mean_, cov_, color='g')

    # Unscented Transform (First 6 elements are pose (and are just dummy values), next 2 are u, last is z)
    X_mean = np.concatenate([mean_T_w_c.Log().flatten(), mean_u.flatten(), [mean_z]])
    X_cov = block_diag(Sigma_T_w_c, Sigma_u, var_z)


    # Compute the sigma points and weights. (Use the mean_T_w_c as the pose directly), Get the poses as well
    sigma_points, W_m, W_c, poses = find_sigma_points(X_mean, X_cov, alpha=0.75, beta=2, kappa=20, mean_T_w_c=mean_T_w_c)

    # Propagate the sigma points through the function.
    sigma_points_f = np.zeros((3, len(sigma_points)))
 
    for i, params in enumerate(sigma_points):
        pose_sigma = poses[i]
        u_sigma = params[6:8].reshape((2, 1))
        z_sigma = params[8].flatten()
        x_c_sigma = backproject(u_sigma, z_sigma, f, c)
        sigma_points_f[:, i] = (pose_sigma * x_c_sigma).flatten()
      
        vg.plot_pose(ax, pose_sigma.to_tuple(), color='k',axis_colors=['k', 'k', 'k'], alpha=0.05)

    # plot the points
    ax.plot(sigma_points_f[0, :], sigma_points_f[1, :], sigma_points_f[2, :], 'b.', alpha=0.5)

    # Compute the mean and covariance of the propagated sigma points.
    x_w_mean_unscented = np.sum(W_m * sigma_points_f, axis=1, keepdims=True).reshape((3, 1))
   
    cov_x_w_unscented = np.zeros((3, 3))
    for i in range(sigma_points_f.shape[1]):
        diff = sigma_points_f[:, i].reshape((3, 1)) - x_w_mean_unscented
        cov_x_w_unscented += W_c[i] * (diff @ diff.T)

    print(f'mean {x_w_mean_unscented}')
    print(f'cov {cov_x_w_unscented}')

    vg.plot_covariance_ellipsoid(ax, x_w_mean_unscented, cov_x_w_unscented, color='b')
    # Show figure.
    vg.plot.axis_equal(ax)
    plt.legend(['Estimated cov ellipsoid', 'Simulated cov ellipsoid', 'Unscented transform cov ellipsoid'])
    plt.show()


if __name__ == "__main__":
    main()
