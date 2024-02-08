#!/usr/bin/env python

import csky as cy
import numpy as np
import pandas as pd
import glob
import healpy as hp
import pickle, datetime, socket
import histlite as hl
now = datetime.datetime.now
import matplotlib.pyplot as plt
import click, sys, os, time
import config as cg
import utils
flush = sys.stdout.flush
hp.disable_warnings()

repo, ana_dir, base_dir, job_basedir = cg.repo, cg.ana_dir, cg.base_dir, cg.job_basedir

class State (object):
    def __init__ (self, ana_name, ana_dir, save, base_dir, job_basedir):
        self.ana_name, self.ana_dir, self.save, self.job_basedir = ana_name, ana_dir, save, job_basedir
        self.base_dir = base_dir
        self._ana = None

    @property
    def ana (self):
        if self._ana is None:
            repo.clear_cache()
            specs = cy.selections.DNNCascadeDataSpecs.DNNC_10yr
            ana = cy.get_analysis(repo, 'version-001-p01', specs, )
            if self.save:
                cy.utils.ensure_dir (self.ana_dir)
                ana.save (self.ana_dir)
            ana.name = self.ana_name
            self._ana = ana
        return self._ana

    @property
    def state_args (self):
        return '--ana {} --ana-dir {} --base-dir {}'.format (
            self.ana_name, self.ana_dir, self.base_dir)

pass_state = click.make_pass_decorator (State)

@click.group (invoke_without_command=True, chain=True)
@click.option ('-a', '--ana', 'ana_name', default='DNNC', help='Dataset title')
@click.option ('--ana-dir', default=ana_dir, type=click.Path ())
@click.option ('--job_basedir', default=job_basedir, type=click.Path ())
@click.option ('--save/--nosave', default=False)
@click.option ('--base-dir', default=base_dir,
               type=click.Path (file_okay=False, writable=True))
@click.pass_context
def cli (ctx, ana_name, ana_dir, save, base_dir, job_basedir):
    ctx.obj = State.state = State (ana_name, ana_dir, save, base_dir, job_basedir)


@cli.resultcallback ()
def report_timing (result, **kw):
    exe_t1 = now ()
    print ('c11: end at {} .'.format (exe_t1))
    print ('c11: {} elapsed.'.format (exe_t1 - exe_t0))

@cli.command ()
@pass_state
def setup_ana (state):
    state.ana

@cli.command()
@click.argument('temp')
@click.option('--n-trials', default=1000, type=int)
@click.option ('-n', '--n-sig', default=0, type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option('--injgamma', default=2.7, type=float)
@click.option('--fitgamma', default=2.7, type=float)
@click.option ('--fitc', '--fitcutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV) to fit')      
@click.option('--injc', '--injcutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV) to inject')
@pass_state
def do_gp_trials ( 
            state, temp, n_trials, n_sig, 
            poisson, seed, cpus, injgamma, fitgamma, 
            fitc, injc, logging=True):
    """
    Do trials for galactic plane templates including Fermi bubbles
    and save output in a structured directory based on parameters
    """
    temp = temp.lower()
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print('Seed: {}'.format(seed))
    ana = state.ana
    inj_cutoff_GeV = injc * 1e3
    fit_cutoff_GeV = fitc * 1e3
    inj_gamma = injgamma
    fit_gamma = fitgamma

    def get_tr(temp, inj_gamma, inj_cutoff, fit_gamma, fit_cutoff):
        #print(inj_cutoff, fit_cutoff, inj_gamma, fit_gamma)
        gp_conf = cg.get_gp_conf(
            template_str=temp, inj_gamma=inj_gamma, fit_gamma=fit_gamma, inj_cutoff_GeV=inj_cutoff, fit_cutoff = fit_cutoff_GeV, base_dir=base_dir)
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr
    tr = get_tr(temp, inj_gamma, inj_cutoff_GeV, fit_gamma, fit_cutoff_GeV)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()
    trials = tr.get_many_fits (
        n_trials, n_sig=n_sig, poisson=poisson, seed=seed, logging=logging)
    t1 = now ()
    print ('Finished trials at {} ...'.format (t1))
    print (trials if n_sig else cy.dists.Chi2TSD (trials))
    print (t1 - t0, 'elapsed.')
    flush ()
    if n_sig == 0:
        out_dir = cy.utils.ensure_dir (
                '{}/gp/trials/{}/{}/fitgamma/{:.3f}/fitcutoff/{:.0f}/bkg/'.format (
                    state.base_dir, state.ana_name,
                    temp, fit_gamma, fitc,
                    n_sig))
    else:
        out_dir = cy.utils.ensure_dir (
                '{}/gp/trials/{}/{}/{:3f}/fitgamma/{:3f}/fitcutoff/{:.0f}/nsig/{:08.3f}/'.format (
                    state.base_dir, state.ana_name,
                    temp, injgamma, fit_gamma, fitc,
                    n_sig))

    out_file = '{}/trials_{:07d}__seed_{:010d}.npy'.format (
        out_dir, n_trials, seed)
    print ('-> {}'.format (out_file))
    np.save (out_file, trials.as_array)


@cli.command()
@click.argument('temp')
@click.option('--n-trials', default=1000, type=int)
@click.option ('-n', '--n-sig', default=0, type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option('--injgamma', default=2.7, type=float)
@click.option('--injc', '--injcutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV) to inject')
@pass_state
def do_correlated_gp_trials ( 
            state, temp, n_trials, n_sig, 
            poisson, seed, cpus, injgamma, 
            injc, logging=True):
    """
    Do trials for galactic plane templates including Fermi bubbles
    and save output in a structured directory based on parameters
    """
    temp = temp.lower()
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print('Seed: {}'.format(seed))
    ana = state.ana
    inj_cutoff_GeV = injc * 1e3
    inj_gamma = injgamma
    inj_cutoff = injc

    fit_gammas = np.arange(2.5,3.51,.1)
    fit_cutoffs = np.logspace(1,4,12)

    def get_tr(temp, inj_gamma, inj_cutoff, fit_gamma, fit_cutoff):
        #print(inj_cutoff, fit_cutoff, inj_gamma, fit_gamma)
        gp_conf = cg.get_gp_conf(
            template_str=temp, inj_gamma=inj_gamma, fit_gamma=fit_gamma, inj_cutoff_GeV=inj_cutoff, fit_cutoff = fit_cutoff, base_dir=base_dir)
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr
    
    tr_inj = get_tr(temp, inj_gamma, inj_cutoff_GeV, inj_gamma, inj_cutoff_GeV)
    #bkg_dict= cy.bk.get_all('/data/user/ssclafani/data/analyses/fit_cutoff/gp/trials/DNNC/pi0/fitgamma/', '*.npy', merge=np.concatenate, 
    #              post_convert=(lambda x: cy.dists.Chi2TSD (cy.utils.Arrays (x))))
    bkg_dict= cy.bk.get_all('{}/gp/trials/DNNC/pi0/fitgamma/'.format(state.base_dir), '*.npy', merge=np.concatenate, 
                  post_convert=(lambda x: cy.dists.Chi2TSD (cy.utils.Arrays (x))), log=False) 
    
    def get_mtr(ana, temp, inj_gamma, inj_cutoff, fit_gammas, fit_cutoffs):
        trs = []
        bgs = []
        for fit_gamma in sorted(fit_gammas):
            for fit_cutoff in sorted(fit_cutoffs):
                trs.append(get_tr(temp, inj_gamma, inj_cutoff_GeV, fit_gamma, fit_cutoff))
                bgs.append(cy.bk.get_best(bkg_dict, fit_gamma, 'fitcutoff', fit_cutoff, 'bkg'))

        return trs, bgs
    
    trs, bgs = get_mtr(ana, 'pi0', inj_gamma, inj_cutoff_GeV, fit_gammas, fit_cutoffs)
    mtr =   cy.trial.MultiTrialRunner(
        # the Analysis
        ana,
        # bg+sig injection trial runner (produces trials)
        tr_inj=tr_inj,
        # llh test trial runners (perform fits given trials)
        trs=trs,
        # background distrubutions
        bgs=bgs,
        # use multiprocessing
        mp_cpus=cpus,
    )


    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    print('Gamma: {} Cutoff Energy {} GeV'.format(inj_gamma, inj_cutoff_GeV))
    flush ()
    trials = mtr.get_many_fits (
        n_trials, n_sig=n_sig, poisson=poisson, seed=seed, logging=logging)
    t1 = now ()
    print ('Finished trials at {} ...'.format (t1))
    print (t1 - t0, 'elapsed.')
    flush ()
    if n_sig == 0:
        out_dir = cy.utils.ensure_dir (
                '{}/gp/trials/{}/{}/injgamma/{:3f}/injcutoff/{:.0f}/bkg/'.format (
                    state.base_dir, state.ana_name,
                    temp, inj_gamma, inj_cutoff))
    else:
        out_dir = cy.utils.ensure_dir (
                '{}/gp/trials/{}/{}/correlated_trials/injgamma/{:3f}/injcutoff/{:.0f}/nsig/{:08.3f}/'.format (
                    state.base_dir, state.ana_name,
                    temp, inj_gamma, inj_cutoff,
                    n_sig))

    out_file = '{}/trials_{:07d}__seed_{:010d}.npy'.format (
        out_dir, n_trials, seed)
    print ('-> {}'.format (out_file))
    np.save (out_file, trials.as_array)


@cli.command()
@click.argument('temp')
@click.option('--n-trials', default=1000, type=int)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('--gamma', default=None, type=float)
@click.option ('--nsigma', default=0, type=int)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@pass_state
def do_gp_sens ( 
        state, temp, n_trials,  seed, cpus, gamma, nsigma,
        cutoff, logging=True):
    """
    Calculate for galactic plane templates including fermi bubbles
    Recommend to use do_gp_trials for analysis level mass trial calculation
    """
    temp = temp.lower()
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    dir = cy.utils.ensure_dir ('{}/templates/{}'.format (state.base_dir, temp))
    cutoff_GeV = cutoff * 1e3

    def get_tr(temp):
        gp_conf = cg.get_gp_conf(
            template_str=temp,
            gamma=gamma,
            cutoff_GeV=cutoff_GeV,
            base_dir=state.base_dir,
        )
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    tr = get_tr(temp)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()

    bg = cy.dists.Chi2TSD(tr.get_many_fits (
      n_trials, n_sig=0, poisson=False, seed=seed, logging=logging))
    t1 = now ()
    print ('Finished bg trials at {} ...'.format (t1))
    if nsigma == 0:
        template_sens = tr.find_n_sig(
                        bg.median(), 
                        0.9, #percent above threshold (0.9 for sens)
                        n_sig_step=10,
                        batch_size = n_trials / 3, 
                        tol = 0.02)
    else:
        template_sens = tr.find_n_sig(
                        bg.isf_nsigma(nsigma),
                        0.5, #percent above threshold (0.5 for dp)
                        n_sig_step=50,
                        batch_size = n_trials / 3, 
                        tol = 0.02)
    
    if temp == 'pi0':
        template_sens['fluxE2_100TeV'] = tr.to_E2dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
        template_sens['fluxE2_100TeV_GeV'] = tr.to_E2dNdE(template_sens['n_sig'], 
            E0 = 1e5 , unit = 1)
        template_sens['flux_100TeV'] = tr.to_dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
        out_dir = cy.utils.ensure_dir(
            '{}/gp/{}/gamma/{}'.format(
            state.base_dir,temp, gamma))
    elif temp == 'fermibubbles':
        template_sens['fluxE2_100TeV'] = tr.to_E2dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3, flux = cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV))
        template_sens['flux_100TeV'] = tr.to_dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3, flux = cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV))
        template_sens['flux_1TeV'] = tr.to_dNdE(template_sens['n_sig'], 
            E0 = 1 , unit = 1e3, flux = cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV))
        out_dir = cy.utils.ensure_dir(
            '{}/gp/{}/gamma/{}/cutoff/{}_TeV/'.format(
            state.base_dir,temp, gamma, cutoff))
    else:
        template_sens['model_norm'] = tr.to_model_norm(template_sens['n_sig'])
        out_dir = cy.utils.ensure_dir(
            '{}/gp/{}/'.format(
            state.base_dir,temp))

    flush ()
    print(cutoff_GeV) 
    if nsigma == 0:
        out_file = out_dir + 'sens.npy'
    else: 
        out_file = out_dir + 'dp_{}sigma.npy'.format(nsigma)

    print(template_sens)
    np.save(out_file, template_sens)
    print ('-> {}'.format (out_file))                                                          

@cli.command()
@click.argument('temp')
@click.option ('--seed', default=None, type=int)
@click.option ('--gamma', default=None, type=float)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--flux', default=6.115e-10, type=float)
@pass_state
def do_gp_scan (
        state, temp, seed, gamma, cutoff, flux):
        """
        inject events then scan in all paramater space
        """
        from scipy import stats
        temp = temp.lower()
        if seed is None:
            seed = int (time.time () % 2**32)
        random = cy.utils.get_random (seed)
        print('Seed: {}'.format(seed))
        ana = state.ana
        inj_cutoff_GeV = cutoff * 1e3
        inj_gamma = gamma
        inj_cutoff = cutoff
        template_cache_dir = '/data/user/ssclafani/csky_fit/template_caches'
 

        def get_conf(gamma, cutoff):                                                 
            template_repo = cy.selections.Repository(
                local_root='/data/ana/analyses/NuSources/2021_DNNCascade_analyses')
            template = template_repo.get_template('Fermi-LAT_pi0_map')
            fit_flux =  cy.hyp.PowerLawFlux(gamma, energy_cutoff=cutoff)
            gp_conf = {
                'ana' : ana,
                'template': template,
                'flux' : fit_flux,
                'randomize': ['ra'],
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
                'mp_cpus' : 5,
            }
            return gp_conf

        def get_tr(temp, gamma, cutoff):
            gp_conf = get_conf(
                gamma=gamma, cutoff = cutoff)
            tr = cy.get_trial_runner(gp_conf, ana=ana)
            return tr 

        inj_tr = get_tr(temp, gamma, cutoff)
        ns=inj_tr.to_ns(flux, E0=1, unit=1e3,
                         flux=cy.hyp.PowerLawFlux(gamma, energy_cutoff=cutoff))
        trial = inj_tr.get_one_trial(n_sig=ns, TRUTH=False, seed=seed)
        L = inj_tr.get_one_llh_from_trial(trial)
        fit = L.fit(**inj_tr.fitter_args)
        ts_fit, ns_fit, _ = fit
        print(-2*L.get_minus_llh_ratio(ns))
        print(ts_fit, ns_fit)

        conf = get_conf(inj_gamma, inj_cutoff)
        tr = cy.get_trial_runner(conf=conf)
        L = tr.get_one_llh_from_trial(trial)
        print(L.fit())

        fit_gammas = np.linspace(2.5,3.51,40)
        fit_cutoffs = np.logspace(4,8,40) #GeV
        trials = np.ndarray((len(fit_gammas), len(fit_cutoffs)))
        injs = np.ndarray((len(fit_gammas), len(fit_cutoffs)))
        def get_trials_from_flux (f):
            for i, g in enumerate(fit_gammas):
                for j, c in enumerate(fit_cutoffs):
                    print(g, c)
                    tr = get_tr(temp, g, c)
                    L = tr.get_one_llh_from_trial(trial)
                    ns=tr.to_ns(f, E0=1, unit=1e3,
                                     flux=cy.hyp.PowerLawFlux(g, energy_cutoff=c))
                    delta_2llh = -2*L.get_minus_llh_ratio(ns=ns)
                    print(ns)
                    trials[i][j] = delta_2llh
            return trials
        plot_dir = '/data/user/ssclafani/csky_fit/inj_gamma/{}/inj_cutoff/{}/inj_flux/{}/'.format(inj_gamma, inj_cutoff, flux)
        cy.utils.ensure_dir(plot_dir)
        print('Fixing flux to {}: scanning gamma and cutoff'.format(flux) )
        trials = get_trials_from_flux(flux)

        fig, ax = plt.subplots()
        gamma, cutoff = np.meshgrid( fit_gammas, np.log10(fit_cutoffs),)
        delta_t = -(trials - np.max(trials))
        plt.pcolormesh(gamma, cutoff, delta_t.T,
                   cmap='viridis_r')
        gamma, cutoff = np.meshgrid( fit_gammas, np.log10(fit_cutoffs),)
        contour_fracs = [0.68, .95]
        dLLHs = stats.chi2.ppf (contour_fracs, 3)
        contour_labels = {dLLH: r'{:.0f}$\%$'.format (100*frac) for (dLLH, frac) in zip (dLLHs, contour_fracs)}
        plt.scatter(inj_gamma, np.log10(inj_cutoff), marker='x', s=25, c='r', label='inj')
        plt.colorbar(label='-2$\Delta$ln($L$)')
        cont = plt.contour(fit_gammas, np.log10(fit_cutoffs), delta_t.T, dLLHs, colors='k', linestyles=['solid', 'dashed'])
        ax.clabel (cont, inline=True, fmt=contour_labels, fontsize=10, inline_spacing=1)
        plt.xlabel('Gamma')
        plt.ylabel('log E$_{C}$ ')
        plt.title('Inj Flux: {{}}x10${-10}$'.format(f*1e10))
        cy.plotting.saving(plot_dir, 'gamma_vs_cutoff')

        fit_fluxes = np.linspace(1,10,40) * 1e-10
        fit_gammas = np.linspace(2.5,3.5,40)
        trials = np.ndarray((len(fit_gammas), len(fit_fluxes)))
        injs = np.ndarray((len(fit_gammas), len(fit_fluxes)))
        def get_trials_from_cutoff (c):
            for i, g in enumerate(fit_gammas):
                for j, f in enumerate(fit_fluxes):
                    print(g, c)
                    tr = get_tr(temp, g, c)
                    L = tr.get_one_llh_from_trial(trial)
                    ns=tr.to_ns(f, E0=1, unit=1e3,
                                     flux=cy.hyp.PowerLawFlux(g, energy_cutoff=c))
                    delta_2llh = -2*L.get_minus_llh_ratio(ns=ns)
                    print(ns)
                    trials[i][j] = delta_2llh
            return trials
        trials = get_trials_from_cutoff(cutoff)
        fig, ax = plt.subplots()
        gamma, dNdE = np.meshgrid( fit_gammas, fit_fluxes*1e10,)
        delta_t = -(trials - np.max(trials))
        plt.pcolormesh(gamma, dNdE, delta_t.T,
                   cmap='viridis_r',vmax=10)

        contour_fracs = [0.68, .95]
        dLLHs = stats.chi2.ppf (contour_fracs, 3)
        contour_labels = {dLLH: r'{:.0f}$\%$'.format (100*frac) for (dLLH, frac) in zip (dLLHs, contour_fracs)}


        plt.scatter(inj_gamma, f1*1e10, marker='x', s=25, c='r', label='inj')
        plt.colorbar(label='-2$\Delta$ln($L$)')

        cont = plt.contour(fit_gammas, fit_fluxes*1e10, delta_t.T, dLLHs, colors='k', linestyles=['solid', 'dashed'])
        ax.clabel (cont, inline=True, fmt=contour_labels, fontsize=10, inline_spacing=1)
        plt.xlabel('Gamma')
        plt.ylabel('$\Phi$')
        plt.title('Inj Cutoff: {}'.format(cutoff))
        cy.plotting.saving(plot_dir, 'gamma_vs_flux')

if __name__ == '__main__':
    exe_t0 = now ()
    print ('start at {} .'.format (exe_t0))
    cli ()
