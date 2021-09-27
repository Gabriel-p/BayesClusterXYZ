
import numpy as np
import emcee
from astropy.io import ascii
import astropy.units as u
import astropy.coordinates as coord


def main(
    ra_c='RA_ICRS', de_c='DE_ICRS', plx_c='Plx', dmin=300, dmax=500,
        w_field=.5, w_cluster=.95, nruns=5000, nwalkers=10, ndim=1):
    """
    Estimate the distance to a cluster member using Bayesian inference. The
    likelihood and prior are defined as in Carrera et al. (2019)

    Parameters
    ----------
    ra_c, de_c, plx_c : names for the (RA, DEC, Plx) columns
    dmin, dmax : Range limit for the distances (used as a prior)
    w_field, w_cluster : weight values for the prior model. By default we use
      the same values used by Pang et al (2020)
    nruns, nwalkers, ndim : parameters for 'emcee'

    """

    # Load data for NGC2516
    data = ascii.read('input/ngc2516_pang.dat')
    print("Data loaded")

    d_0, sigma_d = initDist(data, ra_c, de_c, plx_c)
    print("Not corrected distance estimate: {:.1f} +/- {:.1f}".format(
        d_0, sigma_d))

    print("Running emcee...")
    d_bayes = bayesInference(
        data, d_0, sigma_d, dmin, dmax, w_field, w_cluster, nruns, nwalkers,
        ndim)

    # Store results to file
    ascii.write(d_bayes, "bayes_dists.dat", names=('bayes_dist',))

    print("Finished")


def initDist(data, ra_c, de_c, plx_c):
    """
    Estimate the (not corrected) distance to the cluster and its STDDEV.
    These values are used by the Gaussian prior
    """
    c = coord.SkyCoord(ra=data[ra_c] * u.degree,
                       dec=data[de_c] * u.degree,
                       distance=(1000 / data[plx_c]) * u.pc,
                       frame='icrs')
    coords = c.galactic.cartesian
    x, y, z = coords.x.value, coords.y.value, coords.z.value
    obs_d = np.sqrt(x**2 + y**2 + z**2)

    d_0, sigma_d = np.median(obs_d), np.std(obs_d)

    return d_0, sigma_d


def bayesInference(
    data, d_0, sigma_d, dmin, dmax, w_field, w_cluster, nruns, nwalkers,
        ndim):
    """
    Estimate the corrected distance for each star using 'emcee'
    """
    d_bayes = []
    for i, st in enumerate(data):
        # Parallax and its error
        plx, sigma_plx = st['Plx'], st['e_Plx']

        # Initial random starting positions
        pos = d_0 + sigma_d * np.random.randn(nwalkers, ndim)

        # Solve with emcee
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(plx, sigma_plx, d_0, sigma_d, dmin, dmax, w_field,
                  w_cluster))
        sampler.run_mcmc(pos, nruns, progress=False)
        samples = sampler.get_chain()
        # Discard the first half of the chains as burn-in
        post_bi = samples[int(nruns * .5):, :, 0]
        # Store the mean of the distribution
        d_bayes.append(np.mean(post_bi))
        print(i)

    return np.array([d_bayes]).T


def log_probability(
        d, plx, sigma_plx, d_0, sigma_d, dmin, dmax, w_field, w_cluster):
    """
    """
    # Restrict solutions to this range
    if d < dmin or d > dmax:
        return -np.inf
    lp = log_prior(d, d_0, sigma_d, w_field, w_cluster)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(d, plx, sigma_plx)


def log_likelihood(d, plx, sigma_plx):
    return -(plx - 1000. / d)**2. / (2 * sigma_plx**2)


def log_prior(d, d_0, sigma_d, w_field, w_cluster):
    """
    Logarithmic prior combining the exponentially decreasing density with
    the Gaussian density.
    """
    def gaussian(d, d_0, sigma_d):
        return np.exp(-(d - d_0)**2 / (2 * sigma_d**2)) / np.sqrt(2 * sigma_d)

    def exp_decr_prior(d, L=8000):
        """
        Exponentially decreasing space density prior
        L: Scale of the exponentially decreasing density
        """
        return np.piecewise(
            d, [d < 0, d >= 0], [0, 1 / (2 * L**3.) * d**2. * np.exp(-d / L)])

    return np.log(
        w_field * exp_decr_prior(d) + w_cluster * gaussian(d, d_0, sigma_d))


if __name__ == '__main__':
    main()
