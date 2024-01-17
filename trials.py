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

    def get_tr(temp, inj_gamma, inj_cutoff, fit_gamma, fit_cutoff):
        #print(inj_cutoff, fit_cutoff, inj_gamma, fit_gamma)
        gp_conf = cg.get_gp_conf(
            template_str=temp, inj_gamma=inj_gamma, fit_gamma=fit_gamma, inj_cutoff_GeV=inj_cutoff, fit_cutoff = fit_cutoff_GeV, base_dir=base_dir)
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    fit_gammas = np.arange(2.5,3.51,.1)
    fit_cutoffs = np.logspace(4,6,12)

    def get_tr(temp, inj_gamma, inj_cutoff, fit_gamma, fit_cutoff):
        #print(inj_cutoff, fit_cutoff, inj_gamma, fit_gamma)
        gp_conf = cg.get_gp_conf(
            template_str=temp, inj_gamma=inj_gamma, fit_gamma=fit_gamma, inj_cutoff_GeV=inj_cutoff, fit_cutoff = fit_cutoff, base_dir=base_dir)
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr
    
    tr_inj = get_tr(temp, inj_gamma, inj_cutoff_GeV, inj_gamma, inj_cutoff_GeV)
    #bkg_dict= cy.bk.get_all('/data/user/ssclafani/data/analyses/fit_cutoff/gp/trials/DNNC/pi0/fitgamma/', '*.npy', merge=np.concatenate, 
    #              post_convert=(lambda x: cy.dists.Chi2TSD (cy.utils.Arrays (x)))) 
    #print(bkg_dict)
    bkg_dict= cy.bk.get_all('/data/user/ssclafani/data/analyses/fit_cutoff/gp/trials/DNNC/pi0/fitgamma/', '*.npy', merge=np.concatenate, 
                  post_convert=(lambda x: cy.dists.Chi2TSD (cy.utils.Arrays (x)))) 
    
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
        mp_cpus=8,
    )


    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    print('Gamma: {} Cutoff Energy {} GeV'.format(inj_gamma, inj_cutoff_GeV))
    flush ()
    trials = mtr.get_many_fits (
        n_trials, n_sig=n_sig, poisson=poisson, seed=seed, logging=logging)
    t1 = now ()
    print ('Finished trials at {} ...'.format (t1))
    print (trials if n_sig else cy.dists.Chi2TSD (trials))
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
@click.option('--n-trials', default=1000, type=int)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('--emin', default=500, type=float)
@click.option ('--emax', default=8.00, type=float)
@click.option ('--nsigma', default=0, type=int)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@pass_state
def do_gp_sens_erange ( 
        state, temp, n_trials,  seed, cpus, emin, emax, nsigma,
        cutoff, logging=True):
    """
    Same as do_gp_sens with an option to set the emin and emax, 
    Usefull if you want to calculate the relavant 90% enegy range by varying these paramaters
    """
    temp = temp.lower()
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    ana_lim = state.ana
    a = ana_lim[0]
    mask = (a.sig.true_energy > emin) & (a.sig.true_energy < emax)
    ana_lim[0].sig = a.sig[mask]
    dir = cy.utils.ensure_dir ('{}/templates/{}'.format (state.base_dir, temp))
    cutoff_GeV = cutoff * 1e3

    def get_tr(temp, ana):
        gp_conf = cg.get_gp_conf(
            template_str=temp, cutoff_GeV=cutoff_GeV, base_dir=state.base_dir)
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    tr_bg = get_tr(temp, ana)
    tr_lim = get_tr(temp, ana_lim)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()

    bg = cy.dists.Chi2TSD(tr_bg.get_many_fits (
      n_trials, n_sig=0, poisson=False, seed=seed, logging=logging))
    t1 = now ()
    print ('Finished bg trials at {} ...'.format (t1))
    if nsigma == 0:
        template_sens = tr_lim.find_n_sig(
                        bg.median(), 
                        0.9, #percent above threshold (0.9 for sens)
                        n_sig_step=10,
                        batch_size = n_trials / 3, 
                        tol = 0.02)
    else:
        template_sens = tr_lim.find_n_sig(
                        bg.isf_nsigma(nsigma),
                        0.5, #percent above threshold (0.9 for sens)
                        n_sig_step=15,
                        batch_size = n_trials / 3, 
                        tol = 0.02)
    
    if temp == 'pi0':
        template_sens['fluxE2_100TeV'] = tr.to_E2dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
        template_sens['flux_100TeV'] = tr.to_dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
    else:
        template_sens['model_norm'] = tr_bg.to_model_norm(template_sens['n_sig'])

        print(tr_bg.to_model_norm(template_sens['n_sig']))
        print(tr_lim.to_model_norm(template_sens['n_sig']))
    flush ()

    out_dir = cy.utils.ensure_dir(
        '{}/gp/{}/limited_Erange/'.format(
        state.base_dir,temp))
    if nsigma == 0:
        out_file = out_dir + 'Emin_{:.4}_Emax_{:.4}_sens.npy'.format(emin, emax)
    else: 
        out_file = out_dir + 'dp{}.npy'.format(nsigma)

    print(template_sens)
    np.save(out_file, template_sens)
    print ('-> {}'.format (out_file))                                                          

@cli.command ()
@click.option('--inputdir', default=None, type=str, help='Option to Define an input directory outside of default')
@pass_state
def collect_gp_trials (state, inputdir):
    """
    Collect all Background and Signal Trials and save in nested dict
    """
    templates = ['fermibubbles', 'pi0', 'kra5', 'kra50']
    for template in templates:
        print(template)
        if inputdir:
            indir = inputdir
        else: 
            indir = '{}/gp/trials/{}/{}/'.format(state.base_dir, state.ana_name, template) 
        bg = cy.bk.get_all (
            indir,
            'trials*npy',
            merge=np.concatenate, post_convert=cy.utils.Arrays)
        outfile = '{}/gp/trials/{}/{}/trials.dict'.format (state.base_dir, state.ana_name, template)
        print ('->', outfile)
        with open (outfile, 'wb') as f:
            pickle.dump (bg, f, -1)

@cli.command ()
@click.option ('--template', default=None, type=str, 
    help='Only calculate for a particular template, default is all')
@click.option ('--nsigma', default=None, type=float, help='Number of sigma to find')
@click.option ('--fit/--nofit', default=False, help = 'Fit the bkg dist to a chi2 or not?')
@click.option ('--verbose/--noverbose', default=False, help = 'Noisy Output')
@click.option('--inputdir', default=None, type=str, help='option to define an input directory outside of default')
@click.option('--UL/--noUL' , default=False, help='Read in Result and calculate UL to TS')
@pass_state
def find_gp_n_sig(state, template, nsigma, fit, verbose, inputdir, ul):
    """
    Calculate the Sensitivity or discovery potential once bg and sig files are collected
    Does all galactic plane templates
    """
    ana = state.ana
    flux = []
    def find_n_sig_gp(template, gamma=2.0, beta=0.9, nsigma=None, cutoff=None, verbose=False):
        # get signal trials, background distribution, and trial runner
        if cutoff == None:
            cutoff = np.inf
            cutoff_GeV = np.inf
        else:
            cutoff_GeV = 1e3 * cutoff
        if verbose:
            print(gamma, cutoff)
        if template == 'fermibubbles':
            sig_trials = cy.bk.get_best(sig, 'poisson', 'cutoff', cutoff,  'nsig')
        else:
            sig_trials = cy.bk.get_best(sig, 'poisson',  'nsig')
        b = sig_trials[0.0]['ts']
        if verbose:
            print(b)

        def get_tr(temp, cpus=1):
            gp_conf = cg.get_gp_conf(
                template_str=temp,
                cutoff_GeV=cutoff_GeV,
                base_dir=state.base_dir,
            )
            tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
            return tr

        tr = get_tr(template)
        if nsigma !=None:
            #print('sigma = {}'.format(nsigma))
            if fit:
                ts = cy.dists.Chi2TSD(b).isf_nsigma(nsigma)
            else:
                ts = cy.dists.TSD(b).isf_nsigma(nsigma)
        else:
            #print('Getting sensitivity')
            if ul:
                print('Loading Results...')
                results_dir = '{}/gp/results/{}'.format(
                    state.base_dir, template)
                results_file = '{}/{}_unblinded.npy'.format(                 
                    results_dir, template)
                unblinded_results = np.load(results_file, allow_pickle=True)
                if template == 'fermibubbles':
                    if cutoff == 50:
                        loc = 0
                    elif cutoff == 100:
                        loc = 1
                    elif cutoff == 500:
                        loc = 2
                    elif cutoff == np.inf:
                        loc = 3
                    ts = unblinded_results[loc][0]
                else:
                    ts = unblinded_results[0]
                print('Calculating UL for ts={}'.format(ts))
            else:
                ts = cy.dists.TSD(b).median()
        if verbose:
            print(ts)

        result = tr.find_n_sig(ts, beta, max_batch_size=0, logging=verbose, trials=sig_trials)
        if template == 'pi0':
            flux = tr.to_E2dNdE(result, E0 = 100 , unit = 1e3)
        elif template == 'fermibubbles':
            flux  = tr.to_dNdE(result, 
                E0 = 1 , unit = 1e3, flux = cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV))
        else:
            flux = tr.to_model_norm(result)
        # return flux
        if verbose:
            print(ts, beta, result['n_sig'], flux)
        return flux

    if nsigma:
        beta = 0.5
    else:
        beta = 0.9
    if template:
        templates = [template]
    else:
        templates = ['fermibubbles', 'pi0', 'kra5', 'kra50']
    for template in templates:
        if inputdir:
            indir = inputdir + '/gp/trials/{}/{}/'.format(state.ana_name, template)
        else:
            indir = state.base_dir + '/gp/trials/{}/{}/'.format(state.ana_name, template)
        base_dir = state.base_dir + '/gp/trials/{}/{}/'.format(state.ana_name, template)
        sigfile = '{}/trials.dict'.format (indir)
        sig = np.load (sigfile, allow_pickle=True)
        print('Template: {}'.format(template))
        if template == 'fermibubbles':
            for cutoff in [50,100,500,np.inf]:
                f = find_n_sig_gp(template, beta=beta, nsigma=nsigma, cutoff=cutoff, verbose=verbose)
                flux.append(f) 
                print('Cutoff: {} TeV'.format(cutoff))
                print('Flux: {:.8}'.format(f))    
            print(flux)
            if nsigma:
                np.save(base_dir + '/{}_dp_{}sigma_flux.npy'.format(template, nsigma), flux)
            else:
                if ul:
                    np.save(base_dir + '/{}_90UL_flux.npy'.format(template), flux)
                else:
                    np.save(base_dir + '/{}_sens_flux.npy'.format(template), flux)

        else:
            f = find_n_sig_gp(template, nsigma=nsigma,beta =beta, cutoff=cutoff, verbose=verbose)
            print('Flux: {:.8}'.format(f))     
            if nsigma:
                np.save(base_dir + '/{}_dp_{}sigma_flux.npy'.format(template, nsigma), f)
            else:
                if ul:
                    np.save(base_dir + '/{}_90UL_flux.npy'.format(template), f)
                else:
                    np.save(base_dir + '/{}_sens_flux.npy'.format(template), f)


@cli.command()
@click.option('--n-trials', default=1000, type=int)
@click.option ('-n', '--n-sig', default=0, type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--catalog',   default='snr' , type=str, help='Stacking Catalog, SNR, PWN or UNID')
@click.option ('--gamma', default=2.0, type=float, help = 'Spectrum to Inject')
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@pass_state
def do_stacking_trials (
        state, n_trials, gamma, cutoff, catalog,
        n_sig,  poisson, seed, cpus, logging=True):
    """
    Do trials from a stacking catalog
    """
    catalog = catalog.lower()
    print('Catalog: {}'.format(catalog))
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    catalog_file = os.path.join(
        cg.catalog_dir, '{}_ESTES_12.pickle'.format(catalog))
    cat = np.load(catalog_file, allow_pickle=True)
    src = cy.utils.Sources(dec=cat['dec_deg'], ra=cat['ra_deg'], deg=True)
    cutoff_GeV = cutoff * 1e3
    def get_tr(src, gamma, cpus):
        conf = cg.get_ps_conf(src=src, gamma=gamma, cutoff_GeV=cutoff_GeV)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus)
        return tr
    tr = get_tr(src, gamma, cpus)
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
    if n_sig:
        out_dir = cy.utils.ensure_dir (
            '{}/stacking/trials/{}/catalog/{}/{}/gamma/{:.3f}/cutoff_TeV/{:.0f}/nsig/{:08.3f}'.format (
                state.base_dir, state.ana_name, catalog,
                'poisson' if poisson else 'nonpoisson',
                 gamma, cutoff,  n_sig))
    else:
        out_dir = cy.utils.ensure_dir ('{}/stacking/trials/{}/catalog/{}/bg/'.format (
            state.base_dir, state.ana_name, catalog))
    out_file = '{}/trials_{:07d}__seed_{:010d}.npy'.format (
        out_dir, n_trials, seed)
    print ('-> {}'.format (out_file))
    np.save (out_file, trials.as_array)

@cli.command()
@click.option('--n-trials', default=10000, type=int)
@click.option ('--catalog',   default='snr' , type=str, help='Stacking Catalog, SNR, PWN or UNID')
@click.option ('--gamma', default=2.0, type=float, help = 'Spectrum to Inject')
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('--nsigma', default=None, type=float)
@pass_state
def do_stacking_sens (
        state, n_trials, gamma, cutoff, catalog,
        seed, cpus, nsigma,logging=True):
    """
    Do senstivity calculation for stacking catalog.  Useful for quick numbers, not for
    analysis level numbers of trials
    """

    catalog = catalog.lower()
    print('Catalog: {}'.format(catalog))
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    catalog_file = os.path.join(
        cg.catalog_dir, '{}_ESTES_12.pickle'.format(catalog))
    cat = np.load(catalog_file, allow_pickle=True)
    src = cy.utils.Sources(dec=cat['dec_deg'], ra=cat['ra_deg'], deg=True)
    cutoff_GeV = cutoff * 1e3
    out_dir = cy.utils.ensure_dir ('{}/stacking/sens/{}/'.format (state.base_dir, catalog))

    def get_tr(src, gamma, cpus):
        conf = cg.get_ps_conf(src=src, gamma=gamma, cutoff_GeV=cutoff_GeV)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus)
        return tr

    tr = get_tr(src, gamma, cpus)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()
    bg = cy.dists.Chi2TSD(tr.get_many_fits (
      n_trials, n_sig=0, poisson=False, seed=seed, logging=logging))
    t1 = now ()
    print ('Finished bg trials at {} ...'.format (t1))
    if nsigma:
        sens = tr.find_n_sig(
                        bg.isf_nsigma(nsigma), 
                        0.5, #percent above threshold (0.5 for dp)
                        n_sig_step=25,
                        batch_size = n_trials / 3, 
                        tol = 0.02,
                        seed =seed)
    else:
        sens = tr.find_n_sig(
                        bg.median(), 
                        0.9, #percent above threshold (0.9 for sens)
                        n_sig_step=5,
                        batch_size = n_trials / 3, 
                        tol = 0.02,
                        seed = seed)
    sens['flux'] = tr.to_E2dNdE(sens['n_sig'], E0=100, unit=1e3)
    print ('Finished sens at {} ...'.format (t1))
    print (t1 - t0, 'elapsed.')
    print(sens['flux'])
    flush ()
    if nsigma == 0:
        out_file = out_dir + 'sens.npy'
    else: 
        out_file = out_dir + 'dp{}.npy'.format(nsigma)
    np.save(out_file, sens)

@cli.command ()
@click.option ('--dist/--nodist', default=False)
@click.option('--inputdir', default=None, type=str, help='Option to Define an input directory outside of default')
@pass_state
def collect_stacking_bg (state, dist, inputdir):
    """
    Collect all background trials for stacking into one dictionary for calculation of sensitvity
    """
    bg = {'cat': {}}
    cats = ['snr' , 'pwn', 'unid']
    for cat in cats:
        if inputdir:
            bg_dir = inputdir
        else:
            bg_dir = cy.utils.ensure_dir ('{}/stacking/trials/{}/catalog/{}/bg/'.format (
                state.base_dir, state.ana_name, cat))
        print(bg_dir)
        print ('\r{} ...'.format (cat) + 10 * ' ', end='')
        flush ()
        if dist:
            bg = cy.bk.get_all (
                bg_dir, 'trials*npy',
                merge=np.concatenate, post_convert=(lambda x: cy.dists.Chi2TSD (cy.utils.Arrays (x))))
        else:
            bg = cy.bk.get_all (
                bg_dir, 'trials*npy',
                merge=np.concatenate, post_convert=cy.utils.Arrays )

        print ('\rDone.              ')
        flush ()
        if dist:
            outfile = '{}/stacking/{}_bg_chi2.dict'.format (
                state.base_dir,  cat)
        else:
            outfile = '{}/stacking/{}_bg.dict'.format (
                state.base_dir, cat)
        print ('->', outfile)
        with open (outfile, 'wb') as f:
            pickle.dump (bg, f, -1)

@cli.command ()
@click.option('--inputdir', default=None, type=str, help='Option to Define an input directory outside of default')
@pass_state
def collect_stacking_sig (state, inputdir):
    """
    Collect all signal trials for stacking into one dictionary for calculation of sensitvity
    """
    cats = 'snr pwn unid'.split ()
    for cat in cats:
        if inputdir:
            sig_dir = inputdir
        else:
            sig_dir = '{}/stacking/trials/{}/catalog/{}/poisson'.format (
                state.base_dir, state.ana_name, cat)
        sig = cy.bk.get_all (
            sig_dir, '*.npy', merge=np.concatenate, post_convert=cy.utils.Arrays)
        outfile = '{}/stacking/{}_sig.dict'.format (
            state.base_dir,  cat)
        with open (outfile, 'wb') as f:
            pickle.dump (sig, f, -1)
        print ('->', outfile)

@cli.command ()
@click.option ('--nsigma', default=None, type=float, help='Number of sigma to find')
@click.option ('--fit/--nofit', default=False, help='Use chi2fit')
@click.option('--inputdir', default=None, type=str, help='Option to Define an input directory outside of default')
@click.option ('--verbose/--noverbose', default=False, help = 'Noisy Output')
@click.option('--UL/--noUL' , default=False, help='Read in Result and calculate UL to TS')
@pass_state
def find_stacking_n_sig(state, nsigma, fit, inputdir, verbose, ul):
    """
    Calculate the Sensitvity or discovery potential once bg and sig files are collected
    Does all stacking catalogs
    """
    cutoff = None
    ana = state.ana

    def find_n_sig_cat(src, gamma=2.0, beta=0.9, nsigma=None, cutoff=None, verbose=False, unblinded_results=None):
        # get signal trials, background distribution, and trial runner
        if cutoff == None:
            cutoff = np.inf
            cutoff_GeV = np.inf
        else:
            cutoff_GeV = 1e3 * cutoff
        if verbose:
            print(gamma, cutoff)
        sig_trials = cy.bk.get_best(sig,  'gamma', gamma, 'cutoff_TeV', 
            cutoff, 'nsig')
        b = bg
        if verbose:
            print(b)
        conf = cg.get_ps_conf(src=src, gamma=gamma, cutoff_GeV=cutoff_GeV)
        tr = cy.get_trial_runner(ana=ana, conf=conf)
            # determine ts threshold
        if nsigma !=None:
            #print('sigma = {}'.format(nsigma))
            if fit:
                ts = cy.dists.Chi2TSD(b).isf_nsigma(nsigma)
            else:
                ts = cy.dists.TSD(b).isf_nsigma(nsigma)
        else:
            #print('Getting sensitivity')
            if ul:
                print(unblinded_results)
                ts = unblinded_results[0] 
            else:
                ts = cy.dists.TSD(b).median()
        if verbose:
            print(ts)

        # include background trials in calculation
        trials = {0: b}
        trials.update(sig_trials)

        result = tr.find_n_sig(ts, beta, max_batch_size=0, logging=verbose, trials=trials)
        flux = tr.to_E2dNdE(result['n_sig'], E0=100, unit=1e3)
        # return flux
        if verbose:
            print(ts, beta, result['n_sig'], flux)
        return flux 
    fluxs = []
    if nsigma:
        beta = 0.5
    else:
        beta = 0.9
    cats = ['snr', 'pwn', 'unid']
    for cat in cats:
        if inputdir:
            indir = inputdir + '/stacking'
        else:
            indir = state.base_dir + '/stacking/'
        base_dir = state.base_dir + '/stacking/'
        sigfile = '{}/{}_sig.dict'.format (indir, cat)
        sig = np.load (sigfile, allow_pickle=True)
        bgfile = '{}/{}_bg.dict'.format (indir, cat)
        bg = np.load (bgfile, allow_pickle=True)
        print('CATALOG: {}'.format(cat))
        catalog_file = os.path.join(
            cg.catalog_dir, '{}_ESTES_12.pickle'.format(cat))
        srcs = np.load(catalog_file, allow_pickle=True)
        src = cy.utils.Sources(ra = srcs['ra_deg'], dec=srcs['dec_deg'], deg=True)
        if ul: 
            print('Loading Results...')
            results_dir = '{}/stacking/results/{}'.format(
                state.base_dir, cat)
            results_file = '{}/{}_unblinded.npy'.format(                 
                results_dir, cat)
            unblinded_results = np.load(results_file, allow_pickle=True)
            print(unblinded_results)
        else:
            unblinded_results=None
        for gamma in sig['gamma'].keys():
            print ('Gamma: {}'.format(gamma))
            f = find_n_sig_cat(src, gamma=gamma, beta=beta, nsigma=nsigma, cutoff=cutoff, verbose=verbose, unblinded_results=unblinded_results)
            print('Sensitvity Flux: {:.8}'.format(f))     
            fluxs.append(f)
            if nsigma:
                np.save(base_dir + '/stacking_{}_dp_{}sigma_flux_E{}.npy'.format(cat, nsigma, int(gamma * 100)), fluxs)
            else:
                if ul:
                    np.save(base_dir + '/stacking_{}_90UL_flux_E{}.npy'.format(cat, int(gamma * 100)), fluxs)
                else:
                    np.save(base_dir + '/stacking_{}_sens_flux_E{}.npy'.format(cat, int(gamma * 100)), fluxs)


@cli.command()
@click.option('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option('-n', '--n-sig', default=0, type=float,
              help='Number of signal events to inject')
@click.option('--nside', default=128, type=int)
@click.option('--cpus', default=1, type=int)
@click.option('--seed', default=None, type=int)
@click.option('--poisson/--nopoisson', default=True,
              help='toggle possion weighted signal injection')
@click.option('--gamma', default=2.0, type=float,
              help='Gamma for signal injection.')
@click.option('--fit/--nofit', default=False,
              help='Use Chi2 Fit or not for the bg trials at each declination')
@pass_state
def do_sky_scan_trials(
        state, poisson, dec_deg, nside, n_sig, cpus, seed, gamma, fit):
    """
    Scan each point in the sky in a grid of pixels
    """

    if seed is None:
        seed = int(time.time() % 2**32)
    random = cy.utils.get_random(seed)
    print('Seed: {}'.format(seed))
    dec = np.radians(dec_deg)
    sindec = np.sin(dec)
    base_dir = state.base_dir + '/ps/trials/DNNC'
    if fit:
        bgfile = '{}/bg_chi2.dict'.format(base_dir)
        bgs = np.load(bgfile, allow_pickle=True)['dec']
    else:
        bgfile = '{}/bg.dict'.format(base_dir)
        bg_trials = np.load(bgfile, allow_pickle=True)['dec']
        bgs = {key: cy.dists.TSD(trials) for key, trials in bg_trials.items()}

    def ts_to_p(dec, ts):
        return cy.dists.ts_to_p(bgs, np.degrees(dec), ts, fit=fit)

    t0 = now()
    ana = state.ana
    conf = cg.get_ps_conf(src=None, gamma=gamma)
    conf.pop('src')
    conf.update({
        'ana': ana,
        'mp_cpus': cpus,
        'extra_keep': ['energy'],
    })

    inj_src = cy.utils.sources(ra=0, dec=dec_deg, deg=True)
    inj_conf = {
        'src': inj_src,
        'flux': cy.hyp.PowerLawFlux(gamma),
    }

    sstr = cy.get_sky_scan_trial_runner(conf=conf, inj_conf=inj_conf,
                                        min_dec=np.radians(-80),
                                        max_dec=np.radians(80),
                                        mp_scan_cpus=cpus,
                                        nside=nside, ts_to_p=ts_to_p)
    print('Doing one Scan with nsig = {}'.format(n_sig))
    trials = sstr.get_one_scan(n_sig, poisson=poisson, logging=True, seed=seed)

    base_out = '{}/skyscan/trials/{}/nside/{:04d}'.format(
        state.base_dir, state.ana_name, nside)
    if n_sig:
        out_dir = cy.utils.ensure_dir(
            '{}/{}/{}/gamma/{:.3f}/dec/{:+08.3f}/nsig/{:08.3f}'.format(
                base_out,
                'poisson' if poisson else 'nonpoisson',
                'fit' if fit else 'nofit',
                gamma,  dec_deg, n_sig))
    else:
        out_dir = cy.utils.ensure_dir('{}/bg/{}'.format(
            base_out, 'fit' if fit else 'nofit'))
    out_file = '{}/scan_seed_{:010d}.npy'.format(out_dir,  seed)
    print ('-> {}'.format(out_file))
    np.save(out_file, trials)


@cli.command()
@click.option('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option('-n', '--n-sig', default=0, type=float,
              help='Number of signal events to inject')
@click.option('--nside', default=128, type=int)
@click.option('--poisson/--nopoisson', default=True,
              help='toggle possion weighted signal injection')
@click.option('--gamma', default=2.0, type=float,
              help='Gamma for signal injection.')
@click.option('--overwrite/--nooverwrite', default=False,
              help='If True, existing files will be overwritten')
@click.option('--fit/--nofit', default=False,
              help='Use Chi2 Fit or not for the bg trials at each declination')
@click.option('--inputfit/--noinputfit', default=False,
              help='Use Chi2 Fit or not for the bg trials at each declination')
@pass_state
def recalculate_sky_scan_trials(
        state, poisson, dec_deg, nside, n_sig, gamma, fit, inputfit,
        overwrite):
    """
    Recalculate previous sky scan result based on given background trials.

    This can be used to update old sky-scans if more background trials become
    available at each declination value, or if one wants to change from
    `--fit` (estimate via Chi2 Fit) to `--nofit` (use trials directly) and
    vice versa.
    """

    dec = np.radians(dec_deg)
    print('Loading background trials...')
    base_dir = state.base_dir + '/ps/trials/DNNC'
    if fit:
        bgfile = '{}/bg_chi2.dict'.format(base_dir)
        bgs = np.load(bgfile, allow_pickle=True)['dec']
    else:
        bgfile = '{}/bg.dict'.format(base_dir)
        bg_trials = np.load(bgfile, allow_pickle=True)['dec']
        bgs = {key: cy.dists.TSD(trials) for key, trials in bg_trials.items()}

    def ts_to_p(dec, ts):
        return cy.dists.ts_to_p(bgs, np.degrees(dec), ts, fit=fit)

    # get input and output directories
    base_out = '{}/skyscan/trials/{}/nside/{:04d}'.format(
        state.base_dir, state.ana_name, nside)
    if n_sig:
        input_dir = '{}/{}/{}/gamma/{:.3f}/dec/{:+08.3f}/nsig/{:08.3f}'.format(
            base_out,
            'poisson' if poisson else 'nonpoisson',
            'fit' if inputfit else 'nofit',
            gamma,  dec_deg, n_sig)
        out_dir = cy.utils.ensure_dir(
            '{}/{}/{}/gamma/{:.3f}/dec/{:+08.3f}/nsig/{:08.3f}'.format(
                base_out,
                'poisson' if poisson else 'nonpoisson',
                'fit' if fit else 'nofit',
                gamma,  dec_deg, n_sig))
    else:
        input_dir = '{}/bg/{}'.format(base_out, 'fit' if inputfit else 'nofit')
        out_dir = cy.utils.ensure_dir('{}/bg/{}'.format(
            base_out, 'fit' if fit else 'nofit'))

    # collect sky scans that will be recalculated
    print('Collecting input files...')
    input_files = sorted(glob.glob(os.path.join(input_dir, 'scan_seed_*.npy')))

    print('Found {} files. Recalculating p-values...'.format(len(input_files)))
    for input_file in input_files:

        # load and recalculate scan
        scan = np.load(input_file, allow_pickle=True)
        new_scan = utils.recalculate_scan(scan=scan, ts_to_p=ts_to_p)

        out_file = os.path.join(out_dir,  os.path.basename(input_file))

        if not overwrite and os.path.exists(out_file):
            msg = 'File {} already exists. To overwrite, pass `--overwrite`.'
            raise IOError(msg.format(out_file))

        print('-> {}'.format(out_file))
        np.save(out_file, new_scan)


@cli.command()
@click.option('--nside', default=128, type=int)
@click.option('--cpus', default=1, type=int)
@click.option('--seed', default=None, type=int)
@click.option('--poisson/--nopoisson', default=True,
              help='toggle possion weighted signal injection')
@click.option('--gamma', default=2.7, type=float,
              help='Gamma for signal injection.')
@click.option('--fit/--nofit', default=False,
              help='Use Chi2 Fit or not for the bg trials at each declination')
@pass_state
def do_sky_scan_weaksources(
        state, poisson, n ,  nside, n_sig, cpus, seed, gamma, fit):
    """
    Scan each point in the sky in a grid of pixels
    """

    if seed is None:
        seed = int(time.time() % 2**32)
    random = cy.utils.get_random(seed)
    print('Seed: {}'.format(seed))
    template = np.load('/data/ana/analyses/NuSources/2021_DNNCascade_analyses/templates/Fermi-LAT_pi0_map.npy',allow_pickle=True)
    def get_sources(template, N_src):
        idx = np.random.choice(len(template), N_src, p=template/sum(template))
        theta, phi = hp.pix2ang(nside=128, ipix=idx)
        ra = np.rad2deg(phi)
        dec = np.rad2deg(np.pi/2 - theta)
        return ra, dec
    
    N_src = n 
    ras, decs = get_sources(template, N_src)
    inj_src = cy.utils.sources(ras, decs, deg=True)
    base_dir = state.base_dir + '/ps/trials/DNNC'
    n_inj_pi0 = 750
    if fit:
        bgfile = '{}/bg_chi2.dict'.format(base_dir)
        bgs = np.load(bgfile, allow_pickle=True)['dec']
    else:
        bgfile = '{}/bg.dict'.format(base_dir)
        bg_trials = np.load(bgfile, allow_pickle=True)['dec']
        bgs = {key: cy.dists.TSD(trials) for key, trials in bg_trials.items()}

    def ts_to_p(dec, ts):
        return cy.dists.ts_to_p(bgs, np.degrees(dec), ts, fit=fit)

    t0 = now()
    ana = state.ana
    conf = cg.get_ps_conf(src=None, gamma=gamma)
    conf.pop('src')
    conf.update({
        'ana': ana,
        'mp_cpus': cpus,
        'extra_keep': ['energy'],
    })

    inj_src = cy.utils.sources(ra=0, dec=dec_deg, deg=True)
    inj_conf = {
        'src': inj_src,
        'flux': cy.hyp.PowerLawFlux(gamma),
    }

    sstr = cy.get_sky_scan_trial_runner(conf=conf, inj_conf=inj_conf,
                                        min_dec=np.radians(-80),
                                        max_dec=np.radians(80),
                                        mp_scan_cpus=cpus,
                                        nside=nside, ts_to_p=ts_to_p)
    print('Doing one Scan with nsig = {}'.format(n_sig))
    trials = sstr.get_one_scan(n_sig, poisson=poisson, logging=True, seed=seed)

    base_out = '{}/skyscan/trials/{}/nside/{:04d}'.format(
        state.base_dir, state.ana_name, nside)
    if n_sig:
        out_dir = cy.utils.ensure_dir(
            '{}/{}/{}/gamma/{:.3f}/dec/{:+08.3f}/nsig/{:08.3f}'.format(
                base_out,
                'poisson' if poisson else 'nonpoisson',
                'fit' if fit else 'nofit',
                gamma,  dec_deg, n_sig))
    else:
        out_dir = cy.utils.ensure_dir('{}/bg/{}'.format(
            base_out, 'fit' if fit else 'nofit'))
    out_file = '{}/scan_seed_{:010d}.npy'.format(out_dir,  seed)
    print ('-> {}'.format(out_file))
    np.save(out_file, trials)


@cli.command()
@pass_state
def collect_sky_scan_trials_bg(state):
    """
    Collect hottest p-value from background sky scan trials
    """

    base_dir = '{}/skyscan/trials/{}/'.format(state.base_dir, state.ana_name)

    # pre-calculate mask for northern pixels for given nside
    nside_dirs = glob.glob(os.path.join(base_dir, 'nside', '*'))
    nside_list = [int(os.path.basename(nside_dir)) for nside_dir in nside_dirs]
    mask_north_dict = utils.get_mask_north_dict(nside_list=nside_list)

    trials = cy.bk.get_all(
        base_dir, 'scan_seed_*.npy',
        pre_convert=utils.extract_hottest_p_value,
    )

    with open('{}sky_scan_bg.npy'.format(base_dir), 'wb') as f:
        pickle.dump(trials, f, -1)


@cli.command()
@click.option ('--sourcenum', default=1, type=int, help='what source in the list')
@click.option ('--n-sig', default=0, type=float, help='number of events to inject')
@click.option ('--n-trials', default=1000, type=int, help='number of trials to run')
@click.option ('--poisson/--nopoisson', default=True, 
    help = 'toggle possion weighted signal injection')
@click.option ('--gamma', default=2.0, type=float, help='Spectral Index to inject')
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--cpus', default=1, type=int, help='Number of CPUs to use')
@click.option ('--seed', default=None, type=int, help='Trial injection seed')
@pass_state
def do_trials_sourcelist (
        state, sourcenum, n_sig, n_trials,  poisson, gamma, cutoff, cpus, seed, logging=True):
    """
    Do trials at the exact declination of each source.
    Used as an input for the MTR correlated trials to correctly calculate
    pre-trial pvalues
    """
    src_list_file = os.path.join(cg.catalog_dir, 'Source_List_DNNC.pickle')
    sourcelist = pd.read_pickle(src_list_file)
    ras = sourcelist.RA.values
    decs = sourcelist.DEC.values
    names = sourcelist.Names.values

    if seed is None:
        seed = int (time.time () % 2**32)

    t0 = now ()
    ana = state.ana
    def get_tr(dec, ra, cpus=cpus):
        src = cy.utils.sources(ra=ra, dec=dec, deg=True)
        conf = cg.get_ps_conf(src=src, gamma=gamma)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus )
        return tr
    dec = decs[sourcenum]
    ra = ras[sourcenum]
    name = names[sourcenum]

    tr = get_tr(dec=dec, ra=ra, cpus=cpus)

    print('Doing Background Trials for Source {} : {}'.format(sourcenum, name))
    print('DEC {} : RA {}'.format(dec, ra))
    print('Seed: {}'.format(seed))
    trials = tr.get_many_fits(n_trials, n_sig= n_sig, seed=seed, poisson=poisson)
    t1 = now ()
    flush ()
    if n_sig == 0:
        out_dir = cy.utils.ensure_dir ('{}/ps/correlated_trials/bg/source_{}'.format (
            state.base_dir, sourcenum))
        out_file = '{}/bkg_trials_{}_seed_{}.npy'.format (
            out_dir, n_trials, seed)
    else:
        out_dir = cy.utils.ensure_dir('{}/ps/correlated_trials/{}/source_{}/gamma/{:.3f}/cutoff_TeV/{:.0f}/nsig/{:08.3f}'.format (
                state.base_dir,
                'poisson' if poisson else 'nonpoisson',
                sourcenum,
                 gamma, cutoff,  n_sig))
        out_file = '{}/trials_{}_seed_{}.npy'.format (
            out_dir, n_trials, seed)

    print ('-> {}'.format (out_file))
    np.save (out_file, trials.as_array)




@cli.command()
@pass_state
def collect_bkg_trials_sourcelist( state ):
    """
    Collect all the background trials from do-bkg-trials-sourcelist into one list to feed into
    the MTR for calculation of pre-trial pvalues.  Unlike do-ps-trials, these are done at the exact
    declination of each source
    """
    bgs = []
    numsources = 109
    base_dir = '{}/ps/correlated_trials/bg/'.format(state.base_dir)
    for i in range(numsources):
        bg = cy.bk.get_all(base_dir +'source_{}/'.format(i), '*.npy',
            merge=np.concatenate, post_convert =  (lambda x: cy.dists.TSD(cy.utils.Arrays(x))))
        bgs.append(bg)
    np.save('{}/ps/correlated_trials/pretrial_bgs.npy'.format(state.base_dir), bgs)

@cli.command()
@pass_state
def collect_sig_trials_sourcelist( state ):
    """
    Collect all the background trials from do-bkg-trials-sourcelist into one list to feed into
    the MTR for calculation of pre-trial pvalues.  Unlike do-ps-trials, these are done at the exact
    declination of each source
    """
    sigs = []
    numsources = 109
    base_dir = '{}/ps/correlated_trials/poisson/'.format(state.base_dir)
    for i in range(numsources):
        s = cy.bk.get_all(base_dir +'source_{}/'.format(i), '*.npy',
        merge=np.concatenate, post_convert=cy.utils.Arrays)
        sigs.append(s)
    np.save('{}/ps/correlated_trials/sourcelist_sig.npy'.format(state.base_dir), sigs)

@cli.command()
@click.option ('--n-trials', default=10, type=int, help='number of trials to run')
@click.option ('--cpus', default=1, type=int, help='ncpus')
@click.option ('--seed', default=None, type=int, help='Trial injection seed')
@pass_state
def do_correlated_trials_sourcelist (
        state, n_trials, cpus, seed,  logging=True):
    """
    Use MTR for correlated background trials evaluating at each source in the sourcelist
    """
    src_list_file = os.path.join(cg.catalog_dir, 'Source_List_DNNC.pickle')
    sourcelist = pd.read_pickle(src_list_file)
    ras = sourcelist.RA.values
    decs = sourcelist.DEC.values
    if seed is None:
        seed = int (time.time () % 2**32)

    t0 = now ()
    ana = state.ana
    print('Loading Backgrounds')
    bgs = np.load('{}/ps/correlated_trials/pretrial_bgs.npy'.format(state.base_dir), allow_pickle=True)
    assert len(bgs) == len(sourcelist)

    def get_tr(dec, ra, cpus=cpus):
        src = cy.utils.sources(ra=ra, dec=dec, deg=True)
        conf = cg.get_ps_conf(src=src, gamma=2.0)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus)
        return tr
    print('Getting trial runners')
    trs = [get_tr(d,r) for d,r in zip(decs, ras)]
    assert len(trs) == len(bgs)

    tr_inj = trs[0]
    multr = cy.trial.MultiTrialRunner(
        ana,
        # bg+sig injection trial runner (produces trials)
        tr_inj,
        # llh test trial runners (perform fits given trials)
        trs,
        # background distrubutions
        bgs=bgs,
        # use multiprocessing
        mp_cpus=cpus,
    )
    trials = multr.get_many_fits(n_trials, seed=seed)
    t1 = now ()
    flush ()
    out_dir = cy.utils.ensure_dir ('{}/ps/correlated_trials/correlated_bg/'.format (
        state.base_dir, state.ana_name))
    out_file = '{}/correlated_trials_{:07d}__seed_{:010d}.npy'.format (
        out_dir, n_trials, seed)
    print ('-> {}'.format (out_file))
    np.save(out_file, trials.as_array)


@cli.command()
@pass_state
def collect_correlated_trials_sourcelist(state):
    """
    Collect all the correlated MultiTrialRunner background trials from the
    do-correlated-trials-sourcelist into one list. These will be used to
    trial-correct the p-value of the hottest source in the source list.
    """
    base_dir = '{}/ps/correlated_trials/'.format(state.base_dir)
    trials = cy.bk.get_all(
        base_dir + 'correlated_bg/', 'correlated_trials_*.npy',
        merge=np.concatenate,
        post_convert=(lambda x: cy.utils.Arrays(x)),
    )

    # find and add hottest source
    # shape: [n_trials, n_sources]
    mlog10ps = np.stack([
        trials[k] for k in trials.keys() if k[:8] == 'mlog10p_'], axis=1)

    trials['ts'] = np.max(mlog10ps, axis=1)
    trials['idx_hottest'] = np.argmax(mlog10ps, axis=1)

    with open('{}correlated_bg.npy'.format(base_dir), 'wb') as f:
        pickle.dump(trials, f, -1)


@cli.command()
@click.option(
    '--n-trials', default=100, type=int, help='number of trials to run')
@click.option('--cpus', default=1, type=int, help='ncpus')
@click.option('--seed', default=None, type=int, help='Trial injection seed')
@click.option(
    '--gamma', default=2, type=float, help='Signal injection spectrum.')
@click.option(
    '-c', '--cutoff', default=np.inf, type=float,
    help='Exponential cutoff energy (TeV) for signal injection.')
@click.option(
    '-n', '--n-sig', default=0, type=float,
    help='Number of signal events to inject. Note these are distributed among'
    ' selected sources for injection.'
    )
@click.option(
    '--poisson/--nopoisson', default=True,
    help='toggle possion weighted signal injection')
@click.option(
    '--sourcenum', multiple=True, default=None, type=int,
    help='The sources at which to inject. If None, all sources are used.')
@click.option(
    '--n-random-sources', default=None, type=int,
    help='If provided, this number of random sources will be draw from'
    'the provided list of sources without replacement.'
    )
@pass_state
def do_correlated_trials_sourcelist_sig(
        state, n_trials, cpus, seed, gamma, cutoff, n_sig, poisson, sourcenum,
        n_random_sources, logging=True):
    """
    Use MTR for correlated signal trials evaluating at each source in the
    source list.
    """

    src_list_file = os.path.join(cg.catalog_dir, 'Source_List_DNNC.pickle')
    sourcelist = pd.read_pickle(src_list_file)
    ras = sourcelist.RA.values
    decs = sourcelist.DEC.values
    if seed is None:
        seed = int(time.time() % 2**32)

    t0 = now()
    ana = state.ana
    print('Loading Backgrounds')
    bgs = np.load('{}/ps/correlated_trials/pretrial_bgs.npy'.format(
        state.base_dir), allow_pickle=True)
    assert len(bgs) == len(sourcelist)

    # -----------------------
    # define signal injection
    # -----------------------

    # collect sources
    if sourcenum:
        sources = sorted(list(sourcenum))
        source_dir_str = 'sources'
        for source_i in sources:
            source_dir_str += '_{:02d}'.format(source_i)

    else:
        source_dir_str = 'all_sources'
        nsources = 109
        sources = [int(source) for source in range(nsources)]

    if n_random_sources:
        rng = np.random.RandomState(seed=seed)
        msg = (
            'Specified number of random sources {} is larger than number '
            'of available sources {}!'
        )
        if len(sources) < n_random_sources:
            raise ValueError(msg.format(n_random_sources, len(sources)))

        sources = rng.choice(sources, size=n_random_sources, replace=False)

    n_sources = len(sources)

    cutoff_GeV = cutoff * 1e3
    inj_src = cy.utils.sources(ra=ras[sources], dec=decs[sources], deg=True)
    inj_conf = {
        'src': inj_src,
        'flux': cy.hyp.PowerLawFlux(gamma, energy_cutoff=cutoff_GeV),
    }
    # -----------------------

    def get_tr(dec, ra, cpus=cpus, inj_conf={}):
        src = cy.utils.sources(ra=ra, dec=dec, deg=True)
        conf = cg.get_ps_conf(src=src, gamma=2.0)
        tr = cy.get_trial_runner(
            ana=ana, conf=conf, mp_cpus=cpus, inj_conf=inj_conf)
        return tr

    print('Getting trial runners')
    trs = [get_tr(d, r, inj_conf=inj_conf) for d, r in zip(decs, ras)]
    assert len(trs) == len(bgs)

    tr_inj = trs[0]
    multr = cy.trial.MultiTrialRunner(
        ana,
        # bg+sig injection trial runner (produces trials)
        tr_inj,
        # llh test trial runners (perform fits given trials)
        trs,
        # background distrubutions
        bgs=bgs,
        # use multiprocessing
        mp_cpus=cpus,
    )
    trials = multr.get_many_fits(
        n_trials, n_sig=n_sig, poisson=poisson, seed=seed, logging=logging)
    t1 = now()
    flush()
    out_dir = cy.utils.ensure_dir(
        ('{}/ps/correlated_trials/correlated_sig/{}/{}/gamma/{:.3f}/'
            'cutoff_TeV/{:.0f}/{}/{}/n_sources/{:03d}/nsig/{:08.3f}').format(
            state.base_dir, state.ana_name,
            'poisson' if poisson else 'nonpoisson',
            gamma, cutoff, source_dir_str,
            'random_sources' if n_random_sources else 'non_random_sources',
            n_sources, n_sig))

    out_file = '{}/correlated_trials_{:07d}__seed_{:010d}.npy'.format(
        out_dir, n_trials, seed)
    print ('-> {}'.format(out_file))
    np.save(out_file, trials.as_array)


@cli.command()
@pass_state
def collect_correlated_trials_sourcelist_sig(state):
    """
    Collect all the correlated MultiTrialRunner signal trials from the
    do-correlated-trials-sourcelist-sig into one list.
    """
    def post_convert(x):
        trials = cy.utils.Arrays(x)

        # find and add hottest source
        # shape: [n_trials, n_sources]
        mlog10ps = np.stack([
            trials[k] for k in trials.keys() if k[:8] == 'mlog10p_'], axis=1)

        trials['ts'] = np.max(mlog10ps, axis=1)
        trials['idx_hottest'] = np.argmax(mlog10ps, axis=1)

        return trials

    base_dir = '{}/ps/correlated_trials/'.format(state.base_dir)
    trials = cy.bk.get_all(
        base_dir + 'correlated_sig/', 'correlated_trials_*.npy',
        merge=np.concatenate,
        post_convert=post_convert,
    )

    with open('{}correlated_sig.npy'.format(base_dir), 'wb') as f:
        pickle.dump(trials, f, -1)


@cli.command()
@click.option(
    '--n-trials', default=10000, type=int, help='Number of trials to run')
@click.option('--cpus', default=1, type=int, help='ncpus')
@click.option('--seed', default=None, type=int, help='Trial injection seed')
@pass_state
def do_correlated_trials_fermibubbles(
        state, n_trials, cpus, seed,  logging=True):
    """Correlated trials for Fermibubbles

    Use MTR for correlated background trials evaluating for each cutoff
    """
    cutoffs = [50, 100, 500, np.inf]
    if seed is None:
        seed = int(time.time() % 2**32)

    t0 = now()
    ana = state.ana
    print('Loading Backgrounds')
    fermi_dir = '{}/gp/trials/{}/fermibubbles'.format(
        state.base_dir, state.ana_name)
    trials = np.load('{}/trials.dict'.format(fermi_dir), allow_pickle=True)

    # collect list of background trials for each cutoff
    bgs = [
        cy.dists.TSD(trials['poisson']['cutoff'][cutoff]['nsig'][0.0])
        for cutoff in cutoffs
    ]

    def get_tr(temp, cutoff):
        cutoff_GeV = cutoff * 1e3
        gp_conf = cg.get_gp_conf(
            template_str=temp, cutoff_GeV=cutoff_GeV, base_dir=state.base_dir)
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    print('Getting trial runners')
    trs = [get_tr('fermibubbles', cutoff) for cutoff in cutoffs]
    assert len(trs) == len(bgs)

    tr_inj = trs[0]
    multr = cy.trial.MultiTrialRunner(
        ana,
        # bg+sig injection trial runner (produces trials)
        tr_inj,
        # llh test trial runners (perform fits given trials)
        trs,
        # background distributions
        bgs=bgs,
        # use multiprocessing
        mp_cpus=cpus,
    )
    trials = multr.get_many_fits(n_trials, seed=seed)
    t1 = now()
    flush()
    out_dir = cy.utils.ensure_dir('{}/correlated_trials/correlated_bg'.format(
        fermi_dir))
    out_file = '{}/correlated_trials_{:07d}__seed_{:010d}.npy'.format(
        out_dir, n_trials, seed)
    print ('-> {}'.format(out_file))
    np.save(out_file, trials.as_array)


@cli.command()
@pass_state
def collect_correlated_trials_fermibubbles(state):
    """
    Collect all the correlated MultiTrialRunner background trials from the
    do-correlated-trials-fermibubbles into one list. These will be used to
    trial-correct the p-value of the most significant Fermibubble
    energy cutoff.
    """
    base_dir = '{}/gp/trials/{}/fermibubbles/correlated_trials'.format(
        state.base_dir, state.ana_name)
    trials = cy.bk.get_all(
        base_dir + '/correlated_bg/', 'correlated_trials_*.npy',
        merge=np.concatenate,
        post_convert=(lambda x: cy.utils.Arrays(x)),
    )

    # find and add most significant cutoff
    # shape: [n_trials, n_cutoffs]
    mlog10ps = np.stack([
        trials[k] for k in trials.keys() if k[:8] == 'mlog10p_'], axis=1)

    trials['ts'] = np.max(mlog10ps, axis=1)
    trials['idx_hottest'] = np.argmax(mlog10ps, axis=1)

    with open('{}/correlated_bg.npy'.format(base_dir), 'wb') as f:
        pickle.dump(trials, f, -1)


if __name__ == '__main__':
    exe_t0 = now ()
    print ('start at {} .'.format (exe_t0))
    cli ()
