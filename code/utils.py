import os
import numpy as np
import igl
import scipy
import matplotlib.pyplot as plt
import pygeodesic.geodesic as geodesic


def visualize_mat(mat, title, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.matshow(mat)
    ax.axis('equal')
    plt.colorbar()
    plt.title(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)


def load_landmark_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    vids = np.array([int(line.strip().split(',')[0]) for line in lines])
    names = [line.strip().split(',')[1] for line in lines]

    return vids, names

def energy_sample_generator(method, emin, emax, nsamples, variance=5):
    if method == 'log_linear':
        E = np.linspace(np.log(emin), np.log(emax) / 1.02, nsamples)
        sigma = (E[1] - E[0]) * variance
        return E, sigma
    elif method == 'log_sampled':
        tmin = np.abs(4 * np.log(10) / emax)
        tmax = np.abs(4 * np.log(10) / emin)
        E = np.exp(np.linspace(np.log(tmin), np.log(tmax), nsamples))
        return E
    else:
        assert 0, f'Unsupported sample method {method}'


def wave_kernel_signature(evecs, evals, energies, sigma):
    # Computes the wave kernel signature according to the spectrum of a graph
    # derived operator(e.g., the Cotangent Laplacian).
    # This signature was introduced in the paper of M.Aubry, U.Schlickewei and D.Cremers
    # "The Wave Kernel Signature: A Quantum Mechanical Approach To Shape Analysis"
    # http: // www.di.ens.fr / ~aubry / texts / 2011 - wave - kernel - signature.pdf
    #
    # Input: evecs - [n, k] Eigenvectors of a graph operator
    # n is the number of nodes of the graph.
    # evals - [k] Corresponding eigenvalues.
    # energies - [e] Energy values over which the kernel is
    # evaluated.
    # sigma - float Controls the variance of the fitted gausian.
    #
    # Output: signatures - [n, e] Matrix with the values of the WKS for
    # different energies in its columns.

    gaussian_kernel = np.exp(-((energies.reshape(1, -1) - np.log(evals).reshape(-1, 1)) ** 2) / (2 * sigma ** 2))
    signatures = np.matmul(evecs ** 2, gaussian_kernel)  # [n, k], [k, e] -> [n, e]
    scale = np.sum(gaussian_kernel, axis=0, keepdims=True)  # [1, e]
    signatures /= scale

    return signatures


def heat_kernel_signature(evecs, evals, T):
    # Computes the heat kernel signature according to the spectrum of a graph operator(e.g., Laplacian).
    # The signature was introduced in the paper of J.Sun, M.Ovsjanikov and L.Guibas(2009)
    # "A Concise and Provably Informative Multi-Scale Signature-Based on Heat Diffusion."
    #
    # Input: evecs - [n, k] Eigenvectors of a graph operator arranged as columns.n denotes the number of nodes of the graph.
    # evals - [k] corresponding eigenvalues
    # T - [t] time values over which the kernel is evaluated
    #
    # Output:
    # signatures - [n, t] matrix with the values of the HKS for different T in its columns

    low_pass_filter = np.exp(-np.matmul(evals.reshape(-1, 1), T.reshape(1, -1)))  # [k, t]
    signatures = np.matmul(evecs ** 2, low_pass_filter)   #  [n, t]
    scale = np.sum(low_pass_filter, axis=0, keepdims=True)  # [1, t]
    signatures /= scale

    return signatures


def compute_mesh_info(v, f, hks_samples=100, wks_samples=100, num_eigs=100):

    L = -igl.cotmatrix(v, f)
    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    evals, evecs = scipy.sparse.linalg.eigs(L, k=num_eigs, M=M, sigma=-1e-5)

    evals, evecs = np.real(evals[1:]), np.real(evecs[:, 1:])

    heat_time = energy_sample_generator('log_sampled', evals[0], evals[-1], hks_samples)
    hks_sig = heat_kernel_signature(evecs, evals, heat_time)

    energies, sigma = energy_sample_generator('log_linear', evals[0], evals[-1], wks_samples)
    wks_sig = wave_kernel_signature(evecs, evals, energies, sigma)

    return np.concatenate([hks_sig, wks_sig], axis=-1)


def load_mesh_info(path, compute_feature=True):
    v, f = igl.read_triangle_mesh(path)

    mesh_info = {'name': path.split('/')[-1].split('.')[-2],
                 'vertices': v,
                 'faces': f}

    if compute_feature:
        node_feature = compute_mesh_info(v, f)
        mesh_info['feature'] = node_feature

    return mesh_info

def compute_all_pair_normalized_geodesics(mesh, vids):

    geo = geodesic.PyGeodesicAlgorithmExact(mesh['vertices'], mesh['faces'])

    # Compute the longest distance on the mesh.
    # Find the farthest point 'p' from any point.
    dists, _ = geo.geodesicDistances([0], None)
    p_idx = np.argmax(dists)

    # Find the farthest point from point 'p'.
    dists, _ = geo.geodesicDistances([p_idx], None)
    max_dist = np.max(dists)

    num_vids = len(vids)
    all_dists = np.zeros((num_vids, num_vids))
    for i, vid in enumerate(vids):
        dists, _ = geo.geodesicDistances([vid], vids)
        all_dists[i] = dists / max_dist

    # Symmetrize.
    pair_geods = 0.5 * (all_dists + all_dists.T)

    # Ensure that the maximum normalized distance is zero.
    pair_geods[pair_geods > 1] = 1

    return pair_geods


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', type=str)
    return parser.parse_args()


def main(args):
    mesh_info = load_mesh_info(args.path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
