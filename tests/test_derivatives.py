"""
Test that the analytic expressions for derivatives and diagonal Hessian terms are correct by comparing to
numerical estimates.
"""

from __future__ import print_function
import numpy as np
import sys
sys.path.append('..')
import linear_corex as lc
import pylab
import matplotlib.pyplot as plt
from modules import gen_data_cap
from matplotlib import animation

max_iter = 1
N = 200
gpu = False
C = 16
seed = 1


def run_derivative(f, fprime, name='test', xmin=-10, xmax=10, dx=0.05):
    xs = np.arange(xmin, xmax, dx)
    fs = np.array([f(x) for x in xs])
    ds = np.array([fprime(x) for x in xs])
    num_ds = np.gradient(fs, dx)

    fig = plt.figure(figsize=(8,5))
    ax = plt.subplot(111)
    plt.xlim(xmin, xmax)

    plt.xlabel('x', fontsize=18, fontweight='bold')
    plt.plot(xs, fs, '-', lw=1.5, color='g', label='$f(x)$')
    plt.plot(xs, ds, '-', lw=1.5, color='r', label="$f'(x)$")
    plt.plot(xs, num_ds, '-', lw=1., color='b', label="$\hat f'(x)$")
    plt.legend()

    plt.savefig("{}.pdf".format(name), bbox_inches="tight")
    plt.close('all')


def test_test():
    def f(x):
        return x**2

    def fp(x):
        return 2 * x
    run_derivative(f, fp, 'test_test')


def test_first_derivative():
    np.random.seed(seed)
    x, z = gen_data_cap(n_sources=10, k=5, n_samples=N, capacity=C)
    # x = np.random.random((50, 100))
    out = lc.Corex(n_hidden=10, verbose=True, max_iter=max_iter, seed=seed, gpu=gpu).fit(x)
    print('TC', out.tc)
    x = out.preprocess(x, fit=False)
    M = np.cov(x.T)
    print(M, M.shape)

    # 1 / 1 + Si
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        return (1. / (1. + m['Si']))[0]
    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        return -(1. / (1. + m['Si'])**2 * syi * 2 * m['rhoinvrho'])[0,0]
    run_derivative(f, fp, 'test_si')

    # 1 / sqrt(Y_j^2)
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        return syi[0, 0]
    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        return -(m["rho"] * syi**2)[0,0]
    run_derivative(f, fp, 'test_syi')

    # sub-expression...looking for some numerical instability
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        # return (m["invrho"] * m["rhoinvrho"])[0,0]
        return (m["rho"])[0,0]
    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        J1 = syi * (1 - m["rho"]**2)
        return J1[0,0]
    run_derivative(f, fp, 'invrho')

    # "Reusable expression" page in Dec. 2016 notebook
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        O1D1 = out.noise**2 * syi * m["invrho"] * m["rhoinvrho"]
        return O1D1[0,0]
    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        J1 = out.noise**2 * syi**2 * m["invrho"]**2 * (1 + 2. * m["rho"]**2)
        return J1[0,0]
    run_derivative(f, fp, 'test_01d1_1+Si')

    # 01D1
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        O1D1 = out.noise**2 * syi * m["invrho"] * m["rhoinvrho"] / (1 + m["Si"])
        return O1D1[0,0]
    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        J1 = out.noise**2 * syi**2 / (1 + m["Si"]) * m["invrho"]**2 \
            * (1 + 2. * m["rho"]**2 * (1. - m["invrho"] / (1 + m["Si"])))
        return J1[0,0]
    run_derivative(f, fp, 'test_01d1')

    # 01D2
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        O1D1 = out.noise**2 * syi * m["invrho"] * m["rhoinvrho"] / (1 + m["Si"])
        O1D2 = out.noise**2 * syi * m["invrho"]**2 * \
               ((1 + m["rho"]**2) * m["Qij"] - 2 * m["rho"] * m["Si"]) / (1 - m["Si"]**2 + m["Qi"]) \
               - O1D1
        return O1D2[0,0]

    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        J1 = out.noise**2 * syi**2 / (1 + m["Si"]) * m["invrho"]**2 \
            * (1 + 2. * m["rho"]**2 * (1. - m["invrho"] / (1 + m["Si"])))
        U = m["Qij"] - m["rho"] * m["Si"]
        V = m["rho"] * m["Qij"] - m["Si"]
        D = (1 - m["Si"]**2 + m["Qi"])
        A = out.noise**2 * syi**2 * m["invrho"]**2 / D * (
             1 + V + 3 * m["rho"] * U - m["rho"]**2 - 2 * U * (U + m["rho"] * V) / D * m["invrho"]
        )
        J2 = -J1 + A
        return J2[0,0]
    run_derivative(f, fp, 'test_01d2')

    # qi
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        return m['Qi'][0]

    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        d = 2 * syi * (m["Qij"] + m["Si"] * m["rho"]) * m["invrho"]
        return d[0,0]
    run_derivative(f, fp, 'test_qi')

    # 1 / 1+qi - si^2
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        return (1 / (1 + m['Qi'] - m["Si"]**2))[0]

    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        d = -2 * syi * m['invrho'] / (1 - m["Si"]**2 + m["Qi"])**2 * (m["Qij"] - m["Si"] * m["rho"])
        return d[0,0]
    run_derivative(f, fp, 'test_s_qi_denom')

    # B
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        return (m["rhoinvrho"] / (1 + m['Qi'] - m["Si"]**2))[0,0]

    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        B = syi * m['invrho'] / (1 - m["Si"]**2 + m["Qi"]) * (1 + m['rho']**2
            - m['rhoinvrho'] * 2 / (1 - m["Si"]**2 + m["Qi"]) * (m["Qij"] - m["Si"] * m["rho"]))
        # B = syi**2 * np.dot(np.cov(x.T), B.T).T - m["rho"] * np.sum(m["rho"] * B, axis=1, keepdims=True)
        return B[0,0]
    run_derivative(f, fp, 'test_B_diagonal')

    # B off diagonal
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        return np.sum(m["rhoinvrho"][0] / (1 + m['Qi'] - m["Si"]**2))

    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        H = out.noise**2 * syi * syi.T * np.dot(m["rhoinvrho"] / (1 + m["Qi"] - m["Si"]**2), m["rhoinvrho"].T)
        np.fill_diagonal(H, 0.)
        B = syi * m['invrho'] / (1 - m["Si"]**2 + m["Qi"]) * (1 + m['rho']**2
            - m['rhoinvrho'] * 2 / (1 - m["Si"]**2 + m["Qi"]) * (m["Qij"] - m["Si"] * m["rho"]))
        B = syi**2 * np.dot(np.cov(x.T), B.T).T - m["rho"] * np.sum(m["rho"] * B, axis=1, keepdims=True)
        return B[0,0]
    def fpapprox(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        H = out.noise**2 * syi * syi.T * np.dot(m["rhoinvrho"] / (1 + m["Qi"] - m["Si"]**2), m["rhoinvrho"].T)
        np.fill_diagonal(H, 0.)
        B = syi * m['invrho']**2 / (1 - m["Si"]**2 + m["Qi"]) * (1 + m['rho']**2
            - m['rhoinvrho'] * 2 / (1 - m["Si"]**2 + m["Qi"]) * (m["Qij"] - m["Si"] * m["rho"]))
        B = np.sum(B, axis=1, keepdims=True) - m["rho"] * np.sum(m["rho"] * B, axis=1, keepdims=True)
        return B[0,0]

    run_derivative(f, fpapprox, 'test_B_off_diagonal')

    # 02D2
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        H = out.noise**2 * syi * syi.T * np.dot(m["rhoinvrho"] / (1 + m["Qi"] - m["Si"]**2), m["rhoinvrho"].T)
        np.fill_diagonal(H, 0.)
        O2D2 = np.dot(H, out.ws)
        return O2D2[0,0]

    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        H = out.noise**2 * syi * syi.T * np.dot(m["rhoinvrho"] / (1 + m["Qi"] - m["Si"]**2), m["rhoinvrho"].T)
        np.fill_diagonal(H, 0.)
        O2D2 = np.dot(H, out.ws)
        Gji = out.ws * m["rhoinvrho"] * syi
        Gi = np.sum(Gji, axis=0) - Gji
        B = m["invrho"] * m['invrho'] / (1 - m["Si"]**2 + m["Qi"]) * (
            1 + m["rho"]**2 - 2 * m['rhoinvrho'] * (m["Qij"] - m["Si"] * m["rho"]) / (1 - m["Si"]**2 + m["Qi"])
        )
        B = np.dot(np.cov(x.T), B.T).T - m["rho"] * np.sum(m["rho"] * B, axis=1, keepdims=True)
        J3 = - syi * m["rho"] * O2D2 + Gi * out.noise**2 * syi**2 * B
        return J3[0,0]
    run_derivative(f, fp, 'test_02d2')


    # Objective
    def f(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        return -m["TC"]  # Derived assuming we were minimizing an UB on minus TC.

    def fp(z):
        out.ws[0,0] = z
        m = out._calculate_moments(x, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        H = out.noise**2 * syi * syi.T * np.dot(m["rhoinvrho"] / (1 + m["Qi"] - m["Si"]**2), m["rhoinvrho"].T)
        np.fill_diagonal(H, 0.)

        O1D1 = out.noise**2 * syi * m["invrho"] * m["rhoinvrho"] / (1 + m["Si"])
        O1D2 = out.noise**2 * syi * m["invrho"]**2 * \
               ((1 + m["rho"]**2) * m["Qij"] - 2 * m["rho"] * m["Si"]) / (1 - m["Si"]**2 + m["Qi"]) \
               - O1D1
        O2D2 = np.dot(H, out.ws)
        grad = out.ws - (O1D1 - O1D2 - O2D2)
        # Surprise, where does the next line come from. LOL. Dec. 5 2016 notebook, heading starts "From Nov 2016"
        grad = grad - out.ws * np.sum(m['X_i Y_j'].T * grad, axis=1, keepdims=True) * syi**2
        grad = np.dot(M, grad[0])
        return grad[0]
    run_derivative(f, fp, 'test_objective')

def animate_objective():
    # Objective
    np.random.seed(seed)
    xmin = -10
    xmax = 10
    dx = 0.05
    x, z = gen_data_cap(n_sources=10, k=5, n_samples=1000, capacity=20)
    # x = np.random.random((50, 100))
    out = lc.Corex(n_hidden=10, verbose=True, max_iter=1, seed=1, gpu=False).fit(x)
    print('TC', out.tc)
    xp = out.preprocess(x, fit=False)
    M = np.cov(xp.T)

    def f(corex, z, j):
        tmp = corex.ws[j, 0]
        corex.ws[j, 0] = z
        m = corex._calculate_moments(xp, quick=True)
        corex.ws[j, 0] = tmp
        return -m["TC"]  # Derived assuming we were minimizing an UB on minus TC.

    def fp(corex, z, j):
        tmp = corex.ws[j, 0]
        corex.ws[j, 0] = z
        m = corex._calculate_moments(xp, quick=True)
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        H = corex.noise**2 * syi * syi.T * np.dot(m["rhoinvrho"] / (1 + m["Qi"] - m["Si"]**2), m["rhoinvrho"].T)
        np.fill_diagonal(H, 0.)

        O1D1 = corex.noise**2 * syi * m["invrho"] * m["rhoinvrho"] / (1 + m["Si"])
        O1D2 = corex.noise**2 * syi * m["invrho"]**2 * \
               ((1 + m["rho"]**2) * m["Qij"] - 2 * m["rho"] * m["Si"]) / (1 - m["Si"]**2 + m["Qi"]) \
               - O1D1
        O2D2 = np.dot(H, corex.ws)
        grad = corex.ws - (O1D1 - O1D2 - O2D2)
        # Surprise, where does the next line come from. LOL. Dec. 5 2016 notebook, heading starts "From Nov 2016"
        grad = grad - corex.ws * np.sum(m['X_i Y_j'].T * grad, axis=1, keepdims=True) * syi**2
        grad = np.dot(M, grad[j])
        corex.ws[j, 0] = tmp
        return grad[0]

    xs = np.arange(xmin, xmax, dx)

    fig = plt.figure(figsize=(8,5))
    ax = plt.subplot(111)
    plt.xlim(xmin, xmax)
    plt.ylim(-2, 2)
    title = fig.suptitle("")

    plt.xlabel('x', fontsize=18, fontweight='bold')
    artist1, = ax.plot([], [], '-', lw=1.5, color='g', label='$f(x)$')
    artist2, = ax.plot([], [], '-', lw=1.5, color='r', label="$f'(x)$")
    artist3, = ax.plot([], [], '-', lw=1., color='b', label="$\hat f'(x)$")
    plt.legend()

    def init():
        artist1.set_data([],[])
        artist2.set_data([],[])
        artist3.set_data([],[])
        title.set_text("")
        return [artist1, artist2, artist3, title]

    yj = 6
    def animate(j):
        out.fit(x)
        print(np.argmax(np.abs(out.ws[:,0])))
        fs = np.array([f(out, xl, yj) for xl in xs])
        artist1.set_data(xs, fs / 50.)
        artist2.set_data(xs, [fp(out, xl, yj) for xl in xs])
        artist3.set_data(xs, np.gradient(fs, dx))
        title.set_text("TC:{:.3f}".format(out.tc))
        return [artist1, artist2, artist3, title]

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, blit=True)
    anim.save("objective.mp4", fps=10, extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])


if __name__ == "__main__":
    animate_objective()