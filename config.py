# config.py
import os
import socket
import numpy as np
import csky as cy
import getpass
import utils


hostname = socket.gethostname()
username = getpass.getuser()
print('Running as User: {} on Hostname: {}'.format(username, hostname))
job_base = 'fit_cutoff'
if 'condor00' in hostname or 'cobol' in hostname :
    repo = cy.selections.Repository(
        local_root='/data/i3store/users/ssclafani/data/analyses'.format(username))
    template_repo = repo
    ana_dir = cy.utils.ensure_dir(
        '/data/i3store/users/{}/data/analyses'.format(username))
    base_dir = cy.utils.ensure_dir(
        '/data/i3store/users/{}/data/analyses/{}'.format(username, job_base))
    job_basedir = '/data/i3home/{}/submitter_logs'.format(username)
else:
    repo = cy.selections.Repository()
    template_repo = cy.selections.Repository(
        local_root='/data/ana/analyses/NuSources/2021_DNNCascade_analyses')
    ana_dir = cy.utils.ensure_dir('/data/user/{}/data/analyses'.format(username))
    base_dir = cy.utils.ensure_dir('/data/user/{}/data/analyses/{}'.format(username, job_base))
    ana_dir = '{}/ana'.format (base_dir)
    job_basedir = '/scratch/{}/'.format(username) 

# path at which source catalogs are located
catalog_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'catalogs')

# Path to submit config file. This needs to be a relative path to $HOME
# Example content of this file:
#    eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/setup.sh`
#    source  ~/path/to/venv/bin/activate
submit_cfg_file = 'DNNCascade/submitter_config'


# ---------------------------------------------
# Define csky config settings for trial runners
# ---------------------------------------------

def get_ps_conf(src, gamma, cutoff_GeV=np.inf, sigsub=True):
    """Get csky trial runner config for Point Source Likelihood

    Parameters
    ----------
    src : csky.utils.sources
        The sources.
    gamma : float
        The spectral index gamma to use for the powerlaw flux.
    cutoff_GeV : float, optional
        The cutoff value for the powerlaw flux.

    Returns
    -------
    dict
        The config, which may be passed to csky.get_trial_runner
    """
    if sigsub is False:
        print(utils.bcolors.YELLOW)
        print('=========================================================')
        print('=== Warning: trial runner is using no sigsub!         ===')
        print('=========================================================')
        print(utils.bcolors.ENDC)

    conf = {
        'src': src,
        'flux': cy.hyp.PowerLawFlux(gamma, energy_cutoff=cutoff_GeV),
        'update_bg': True,
        'sigsub':  sigsub,
        'randomize': ['ra', cy.inj.DecRandomizer],
        'sindec_bandwidth': np.radians(5),
        'dec_rand_method': 'gaussian_fixed',
        'dec_rand_kwargs': dict(randomization_width=np.radians(3)),
        'dec_rand_pole_exlusion': np.radians(8)
    }

    return conf


def get_gp_conf(
        template_str, inj_gamma=2.7, fit_gamma=2.7, inj_cutoff_GeV=np.inf, fit_cutoff = np.inf,
        base_dir=base_dir, repo=template_repo):
    """Get csky trial runner config for Galactic Plane Template

    Parameters
    ----------
    template_str : str
        The name of the template to use. Must be one of:
        ['pi0', 'fermibubbles', 'kra5', 'kra50']
    gamma : float, optional
        The spectral index to use. This may only be set for Pi0 or
        Fermi Bubbles. Defaults to 2.7 for Pi0 and 2.0 for Fermi Bubbles.
    cutoff_GeV : float, optional
        The cutoff value for the powerlaw spectrum used for Fermi Bubble flux.
        This is only relevant for the Fermi Bubble template.
    base_dir : str, optional
        The path to the base directory. Will be used to cache the templates.
    repo : csky.selections.Repository, optional
        Csky data repository to use.

    Returns
    -------
    dict
        The config, which may be passed to csky.get_trial_runner

    Raises
    ------
    ValueError
        Description
    """

    # Print warning: GP templates have fixed gamma in our analysis.
    # Setting it to custom values should only be done for debugging/testing
    # purposes and the user should be aware of this.

    if template_str == 'pi0':

        # get default gamma
        if inj_gamma is None:
            inj_gamma = 2.7
        else:
            # Note analysis is run with default value of 2.7
            print('\tSetting Gamma for pi0 to: {:3.3f}!'.format(inj_gamma))

        template = repo.get_template('Fermi-LAT_pi0_map')
        template_cache_dir = cy.utils.ensure_dir(
            '{}/templates/pi0/gamma/{:.3f}'.format(base_dir, inj_gamma))
        fit_flux =  cy.hyp.PowerLawFlux(fit_gamma, energy_cutoff=fit_cutoff)
        gp_conf = {
            'template': template,
            #'flux': cy.hyp.PowerLawFlux(inj_gamma, energy_cutoff=inj_cutoff_GeV),
            'flux' : fit_flux,
            'randomize': ['ra'],
            #'fitter_args': dict(gamma=fit_gamma),
            cy.pdf.CustomFluxEnergyPDFRatioModel: dict(
                hkw=dict(bins=(
                       np.linspace(-1, 1, 20),
                       np.linspace(np.log10(500), 8.001, 20)
                       )),
                flux=fit_flux,
                features=['sindec', 'log10energy'],
                normalize_axes=([1])),
            'sigsub': True,
            'energy': 'customflux',
            'update_bg': True,
            'fast_weight': False,
            'dir': template_cache_dir,
            'keep_extra' : 'gamma',
        }
    elif template_str == 'fermibubbles':

        # get default gamma
        if inj_gamma is None:
            inj_gamma = 2.0
        else:
            # Note analysis is run with default value of 2.0
            print('\tSetting Gamma for Fermi Bubbles to: {:3.3f}!'.format(
                inj_gamma))

        template = repo.get_template('Fermi_Bubbles_simple_map')
        template_cache_dir = cy.utils.ensure_dir(
            '{}/templates/fermibubbles/gamma/{:.3f}/cutoff_GeV/{:.0f}'.format(
                base_dir, inj_gamma, fit_cutoff_GeV, inj_cutoff_GeV))

        flux = cy.hyp.PowerLawFlux(inj_gamma, energy_cutoff=cutoff_GeV)
        gp_conf = {
            'template': template,
            'flux': flux,
            'randomize': ['ra'],
            cy.pdf.CustomFluxEnergyPDFRatioModel: dict(
                hkw=dict(bins=(
                       np.linspace(-1, 1, 20),
                       np.linspace(np.log10(500), 8.001, 20)
                       )),
                flux=flux,
                features=['sindec', 'log10energy'],
                normalize_axes=([1])),
            'energy': 'customflux',
            'sigsub': True,
            'update_bg': True,
            'fast_weight': False,
            'dir': template_cache_dir,
        }
    elif 'kra' in template_str:

        # check that gamma isn't set
        if inj_gamma is not None:
            raise ValueError(
                'Gamma must not be specified for KRA, but is:', inj_gamma)

        if template_str == 'kra5':
            template, energy_bins = repo.get_template(
                'KRA-gamma_5PeV_maps_energies', per_pixel_flux=True)
            kra_flux = cy.hyp.BinnedFlux(
                bins_energy=energy_bins,
                flux=template.sum(axis=0))
            template_dir = cy.utils.ensure_dir(
                '{}/templates/kra5'.format(base_dir))
        elif template_str == 'kra50':
            template, energy_bins = repo.get_template(
                      'KRA-gamma_maps_energies', per_pixel_flux=True)
            kra_flux = cy.hyp.BinnedFlux(
                bins_energy=energy_bins,
                flux=template.sum(axis=0))
            template_dir = cy.utils.ensure_dir(
                '{}/templates/kra50'.format(base_dir))

        gp_conf = {
            'template': template,
            'bins_energy': energy_bins,
            'randomize': ['ra'],
            'update_bg': True,
            'fast_weight': False,
            'sigsub': True,
            cy.pdf.CustomFluxEnergyPDFRatioModel: dict(
                hkw=dict(bins=(
                       np.linspace(-1, 1, 20),
                       np.linspace(np.log10(500), 8.001, 20)
                       )),
                flux=kra_flux,
                features=['sindec', 'log10energy'],
                normalize_axes=([1])),
            'energy': False,
            'dir': template_dir,
        }
    else:
        raise ValueError('Unknown template name: {}'.format(template_str))

    return gp_conf
