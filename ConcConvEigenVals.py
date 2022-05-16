import numpy as np
from scipy.stats import unitary_group
from matplotlib import pyplot as plt

def get_eigenvalues(t_i, tita, mu, w):
    N = len(tita)
    # Genero u_t
    u_t = np.matrix(np.diag(
        np.exp(1j*tita*t_i)
    ))
    # Genero v
    
    v = np.matrix(np.diag(
        np.exp(1j*mu)
    ))
    
    # Calculo Autovalores
    H = w * v * (w**-1) * u_t
    eigen_values = (1j*np.log(np.linalg.eigvals(H)))
    assert np.abs(eigen_values.imag).sum() < 1e-9
    eigen_values = eigen_values.real
    # eigen_values = (1j*np.log(np.linalg.eigvalsh(H)))
    return max(eigen_values), min(eigen_values), list(np.sort(eigen_values))

def iterate_through_t(tita, mu, w, t_step = 0.01, plot=True):
    t = [0]
    mx, mn, eigen_values = get_eigenvalues(0, tita, mu, w)
    max_evs = [mx]
    mix_evs = [mn]
    eigen_values_t = [eigen_values]
    t_i = t_step
    delta_t = 0
    while delta_t < np.pi/8:
        t_i = t_i + t_step
        mx, mn, eigen_values = get_eigenvalues(t_i, tita, mu, w)
        delta_t = abs(mx - max_evs[-1])
        t.append(t_i)
        max_evs.append(mx)
        mix_evs.append(mn)
        eigen_values_t.append(eigen_values)
    
    t_step = - t_step
    t_i = t_step
    delta_t = 0
    max_evs = max_evs[::-1]
    mix_evs = mix_evs[::-1]
    eigen_values_t = eigen_values_t[::-1]
    t = t[::-1]
    while delta_t < np.pi/8:
        t_i = t_i + t_step
        mx, mn, eigen_values = get_eigenvalues(t_i, tita, mu, w)
        
        delta_t = abs(mx - max_evs[-1])
        t.append(t_i)
        max_evs.append(mx)
        mix_evs.append(mn)
        eigen_values_t.append(eigen_values)
        
    max_evs = max_evs[-2:1:-1]
    mix_evs = mix_evs[-2:1:-1]
    t = t[-2:1:-1]
    eigen_values_t = eigen_values_t[-2:1:-1]
    pi_diff = np.ones((len(max_evs),3)).T *(np.array(max_evs) - np.array(mix_evs) < np.pi)
    pi_diff = (max(max_evs) - min(mix_evs))*(np.array(max_evs) - np.array(mix_evs) < np.pi) + min(mix_evs)
    if plot:
        f, ax = plt.subplots(1,1, figsize=(20, 5))
        plt.plot(t, eigen_values_t)
        ax.plot(t, pi_diff, linewidth=0.5, c='k')
        ax.set_xlabel('t')
        ax.set_ylabel('eigenvalues')
        # ax.legend()
        # ax.pcolorfast(ax.get_xlim(), ax.get_ylim(), pi_diff)
        print('thetas:')
        print(tita)
        print('mus:')
        print(mu)
        print("w:")
        print(w)
    return np.array(eigen_values_t)

def get_EV_and_plot(N=None, tita=None, mu=None, w=None):
    if tita is None:
        tita = np.pi * (np.random.rand(N) - 0.5)
    if mu is None:
        mu = np.pi * (np.random.rand(N) - 0.5)
    if w is None:
        w = np.matrix(unitary_group.rvs(N))
    ev = iterate_through_t(
        tita, mu, w, t_step=0.01
    )
    return ev, tita, mu, w