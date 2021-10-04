from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn.metrics.pairwise import rbf_kernel


_sl3_generators = np.array([[[ 0.,  0., +1.],
                             [ 0.,  0.,  0.],
                             [ 0.,  0.,  0.]],

                            [[ 0.,  0.,  0.],
                             [ 0.,  0., +1.],
                             [ 0.,  0.,  0.]],

                            [[ 0., +1.,  0.],
                             [ 0.,  0.,  0.],
                             [ 0.,  0.,  0.]],

                            [[ 0.,  0.,  0.],
                             [+1.,  0.,  0.],
                             [ 0.,  0.,  0.]],

                            [[+1.,  0.,  0.],
                             [ 0., -1.,  0.],
                             [ 0.,  0.,  0.]],

                            [[ 0.,  0.,  0.],
                             [ 0., -1.,  0.],
                             [ 0.,  0., +1.]],

                            [[ 0.,  0.,  0.],
                             [ 0.,  0.,  0.],
                             [+1.,  0.,  0.]],

                            [[ 0.,  0.,  0.],
                             [ 0.,  0.,  0.],
                             [ 0., +1.,  0.]],])


def expmap_sl3_to_SL3(vec8: np.array) -> np.array:
    """ Map an element of `sl(3)` to `SL(3)` using the exponential map.

    An element of `SL(3)` can be used to represent a homography; This
    homography is thus (minimally) parameterized by the 8 elements of
    `vec8`.

    Parameters
    ----------
    `vec8`: 8-element parameterization of the `sl(3)` element to be mapped onto
        an `SL(3)` element.
    the algebra of special linear matrices, `sl(3)`,
    parameterized using the 8 elements of `vec8`, to the corresponding
    element of the special linear group `SL(3)`.

    Notes
    -----
    `sl(3)` is the lie-algebra of special linear matrices.
    g ∈ sl(3) => trace(g) == 0

    `SL(3)` is the lie-group of special linear matrices.
    G ∈ SL(3) => Det(G) == 1
    """
    assert len(vec8.shape) == 1
    assert len(vec8) == 8, "An element of sl3 must have exactly 8 parameters"
    g = np.einsum('i,ijk->jk', vec8, _sl3_generators)
    return scipy.linalg.expm(g)


def _generate_grid_line_segments() -> np.array:
    """Generate x-y grid line segments for a unit-square grid centered
    at the origin.
    """
    grid = []
    for x in np.linspace(-.5, .5, 11):
        grid.append([x, -.5, 1.])
        grid.append([x, +.5, 1.])
    for y in np.linspace(-.5, .5, 11):
        grid.append([-.5, y, 1.])
        grid.append([+.5, y, 1.])
    return np.transpose(grid)


def _generate_grid_of_samples(height, width) -> np.ndarray:
    """Generate random samples in a grid such that the center of the
    grid is the identity sample (0, 0, 0, 0, 0, 0, 0, 0).
    """
    samples = np.zeros((height, width, 8))
    for j in range(height):
        for i in range(width):
            r, c = j - (height // 2), i - (width // 2)
            dist = np.linalg.norm([r, c])
            a = np.random.randn(8)
            a = (a / np.linalg.norm(a)) * (dist**2 / 88)
            samples[j, i] = a
    samples[height // 2, width // 2, :] *= 0.  # set center of grid to be the identity
    return samples


def _box_filter_sample_grid(samples: np.ndarray, num_iters=1) -> np.ndarray:
    """Smooth samples such that transitions between sampled homographies
    are gradual.
    """
    for _iters in range(num_iters):
        smoothed_samples = np.copy(samples)
        for j in range(1, smoothed_samples.shape[0] - 1):
            for i in range(1, smoothed_samples.shape[1] - 1):
                neighbors = samples[j-1:j+2, i-1:i+2].reshape((-1, 8))
                smoothed_samples[j, i] = (0.8 * samples[j, i]) + (0.2 * neighbors.mean(axis=0))
    return smoothed_samples


def _get_displacement_line_segment(homography_matrix_SL3: np.array, p=np.r_[0., 0., 1.]) -> np.array:
    q = homography_matrix_SL3 @ p
    q /= q[-1]
    return np.vstack([p, q]).T


SQRT_PIXEL_SAMPLES = 20
N_PIXEL_SAMPLES = SQRT_PIXEL_SAMPLES**2
pixel_grid = np.transpose([ (x, y) for x in np.linspace(-1, 1, SQRT_PIXEL_SAMPLES)
                                       for y in np.linspace(-1, 1, SQRT_PIXEL_SAMPLES) ])


def _sqrt_gram_matrix(characteristic_length_scale, noise_std):
    N = N_PIXEL_SAMPLES
    matrix = np.zeros((N, N))
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            z = (pixel_grid[:, j] - pixel_grid[:, i]) / characteristic_length_scale
            matrix[j, i] = np.exp(-z.dot(z))
    matrix += np.eye(N) * (noise_std*noise_std)
    return np.linalg.cholesky(matrix)


def sample_SL3_from_gp(tau: float):
    multi_gp = [ _sqrt_gram_matrix(tau, 1e-6) for _ in range(8) ]
    channelwise_samples = [ f @ (0.05*np.random.randn(N_PIXEL_SAMPLES)) for f in multi_gp ]
    N = SQRT_PIXEL_SAMPLES
    sl3_samples = np.dstack(tuple([ a.reshape(N, N) for a in channelwise_samples ]))
    SL3_samples = np.zeros((N, N, 3, 3))
    for j in range(N):
        for i in range(N):
            SL3_samples[j, i, :, :] = expmap_sl3_to_SL3(sl3_samples[j, i])
            assert np.isclose(1,  np.linalg.det(SL3_samples[j, i, :, :]))
    return SL3_samples

def main():
    plt.style.use('seaborn-white')
    plt.figure(figsize=(20, 20))

    for iter, tau in enumerate([.25, .32, .4, .6, .8, 1.6, 3.2, 6.4, 12.8]):
        H_grid = sample_SL3_from_gp(tau)
        distorted = np.zeros(H_grid.shape[:2] + (2,))
        for j in range(SQRT_PIXEL_SAMPLES):
            for i in range(SQRT_PIXEL_SAMPLES):
                p = pixel_grid[:, j * SQRT_PIXEL_SAMPLES + i]
                d = _get_displacement_line_segment(H_grid[j, i])
                distorted[j, i, :] = p + d[:2, 1]

        plt.subplot(3, 3, iter+1)
        for j in range(SQRT_PIXEL_SAMPLES):
            x, y = distorted[j, :, 0], distorted[j, :, 1]
            interpolator = scipy.interpolate.make_interp_spline(y, x)
            y_new = np.linspace(y.min(), y.max(), 100)
            x_new = interpolator(y_new)
            plt.plot(x_new, y_new, 'k:')
        for i in range(SQRT_PIXEL_SAMPLES):
            x, y = distorted[:, i, 0], distorted[:, i, 1]
            interpolator = scipy.interpolate.make_interp_spline(x, y)
            x_new = np.linspace(x.min(), x.max(), 100)
            y_new = interpolator(x_new)
            plt.plot(x_new, y_new, 'k:')

        plt.title('tau={:.2f}'.format(tau))
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig('/tmp/homographies/distortion.png')


def main_():
    grid = _generate_grid_line_segments()

    plt.style.use('seaborn-white')
    plt.figure(figsize=(32, 32))
    figH, figW = 17, 17
    # figH, figW = 5, 5

    samples = _generate_grid_of_samples(figH, figW)
    smoothed_samples = _box_filter_sample_grid(samples, num_iters=1)

    # Crop out edge, since edge is not smoothed
    samples = smoothed_samples[1:-1, 1:-1, :]
    figH, figW = figH - 2, figW - 2
    assert samples.shape == (figH, figW, 8), "{} != {}".format(samples.shape, (figH, figW, 8))
    samples[figH // 2, figW // 2, :] *= 0.  # Center of grid must be identity

    for i in range(figH * figW):
        r, c = (i // figH) - (figH // 2), (i % figW) - (figW // 2)
        dist = np.linalg.norm([r, c])
        H = expmap_sl3_to_SL3(samples[i // figH, i % figW])
        assert np.isclose(np.linalg.det(H), 1)
        q = H @ grid
        q = q / q[-1,:]

        plt.subplot(figH, figW, i+1)
        intensity = 1 - (dist / (1.5 * max(figH // 2, figW // 2)))
        intensity = np.clip(intensity, 0.1, 1)
        for i in range(0, len(q.T), 2):
            plt.plot(q[0, i:i+2], q[1, i:i+2], 'r-', linewidth=2*intensity, c='0.7')
            plt.plot(q[0, i:i+2], q[1, i:i+2], '.', c='0.1')
        d = _get_displacement_line_segment(H)[:2, :]
        plt.plot(d[0, :], d[1, :], 'r-')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.setp(plt.gca().spines.values(), color=(0.9, 0.9, 0.9))
        plt.gca().set_facecolor((0.9, 0.9, 0.9))

    plt.tight_layout()
    plt.savefig('/tmp/homographies/plots.png')


if __name__ == '__main__':
    main()
