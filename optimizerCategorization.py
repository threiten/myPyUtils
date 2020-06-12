import numpy as np
import scipy.interpolate as interp
import root_pandas
import pickle as pkl
import gzip
import os
import re
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import ray
import copy
import matplotlib as mpl
import pandas as pd
import Categorizer
import argparse


def findCuts(dfSig, dfsBkg, initGuess, mvaCut, upLimIn=0.04, nAct=50, penEvts=1000., useAsimov=False, intLumi=None):

    remoteOptim = ray.remote(Categorizer.Optimizer)
    remSig = ray.put(dfSig)
    remBkg = []
    for df in dfsBkg:
        remBkg.append(ray.put(df))

    NetActSig = ray.util.ActorPool([remoteOptim.remote(
        dfSig, dfsBkg, debug=False, setZero=True) for _ in range(int(3*nAct/4))])
    NetActMva = ray.util.ActorPool([remoteOptim.remote(
        dfSig, [dfsBkg[1]], debug=False, setZero=True) for _ in range(int(nAct/4))])

    resIds = {}
    res = {}
    resSig = []
    nCats = len(initGuess) - 1
    currGuess = initGuess
    space = {}
    for i in range(nCats):
        resIds['cut_{}'.format(i)] = []
        upLim = currGuess[i+2] if (i < len(currGuess)-2) else upLimIn
        space['space_{}'.format(i)] = np.linspace(currGuess[i], upLim, 151)
        tmpGuess = copy.deepcopy(currGuess)
        tmpGuesses = []
        for nWork, cut in enumerate(space['space_{}'.format(i)]):
            tmpGuesses.append(tmpGuess[:i+1] + [cut] + tmpGuess[i+2:])

        resIt = NetActSig.map(lambda a, cats: a.sigMultCat.remote(
            cats, 'subleadmva>{0} and leadmva>{0}'.format(mvaCut), penEvts=penEvts, useAsimov=useAsimov, intLumi=intLumi), tmpGuesses)
        res['cut_{}'.format(i)] = np.array([res for res in resIt])
        res['cut_{}'.format(i)][np.isnan(res['cut_{}'.format(i)])] = 0.
        print(res['cut_{}'.format(i)])
        currGuess[i+1] = space['space_{}'.format(
            i)][res['cut_{}'.format(i)] == res['cut_{}'.format(i)].max()][0]
        resSig.append(res['cut_{}'.format(i)].max())

    print(currGuess)
    print(resSig)
    resSig = np.array(resSig)
    oldSig = -resSig[0]
    crit = 1
    nIt = 0

    while crit > (nCats/2.)*0.001 or nIt<2:
        print(nIt)
        print(currGuess)

        currGuess = np.around(currGuess, decimals=6)
        resSig = []

        for i in range(nCats):
            upLim = currGuess[i+2] if (i < len(currGuess)-2) else upLimIn
            space['space_{}'.format(i)] = np.linspace(
                np.around(currGuess[i], 4), np.around(upLim, 4), 300)

        for i in range(nCats):
            tmpGuess = copy.deepcopy(list(currGuess))
            resIds['cut_{}'.format(i)] = []
            tmpGuesses = []
            for cut in space['space_{}'.format(i)]:
                tmpGuesses.append(tmpGuess[:i+1] + [cut] + tmpGuess[i+2:])
            resIt = NetActSig.map(lambda a, cut: a.sigMultCat.remote(
                cut, 'subleadmva>{0} and leadmva>{0}'.format(mvaCut), penEvts=penEvts, useAsimov=useAsimov, intLumi=intLumi), tmpGuesses)
            res['cut_{}'.format(i)] = np.array([res for res in resIt])
            res['cut_{}'.format(i)][np.isnan(res['cut_{}'.format(i)])] = 0.
            print(res['cut_{}'.format(i)])
            currGuess[i+1] = space['space_{}'.format(
                i)][res['cut_{}'.format(i)] == res['cut_{}'.format(i)].max()][0]
            resSig.append(res['cut_{}'.format(i)].max())

        resIds['mvaCut'] = []
        tmpMvaCuts = []
        for cut in np.linspace(0.2, 0.6, 125):
            tmpMvaCuts.append(cut)

        mvaIt = NetActMva.map(lambda a, mva: a.sigMultCat.remote(
            currGuess, 'subleadmva>{0} and leadmva>{0}'.format(mva), penEvts=int(penEvts/2), useAsimov=useAsimov, intLumi=intLumi), tmpMvaCuts)
        res['mvaCut'] = np.array([res for res in mvaIt])
        res['mvaCut'][np.isnan(res['mvaCut'])] = 0.
        print(res['mvaCut'])
        mvaCut = np.linspace(0.2, 0.6, 125)[
            res['mvaCut'] == res['mvaCut'].max()]

        resSig = np.array(resSig)
        if resSig.shape[0] > 1:
            diffSig = np.abs(resSig[1:] - resSig[:-1])
        else:
            diffSig = np.array([np.abs(resSig[0]-oldSig)])
        if len(diffSig) > 1:
            crit = abs(max(diffSig)/min(resSig))
        else:
            crit = abs(diffSig/min(resSig))
        nIt += 1
        oldSig = resSig[0]
        print(mvaCut)
        print(currGuess)
        print(resSig)
        print(crit)

    return currGuess, res, space


def main(options):

    ray.init(address='auto', redis_password=options.redis_password, log_to_driver=False)

    if options.year == '2016':
        paths = {'signal': '/t3home/threiten/eos/Analysis/Differentials/2016ReReco/dev_differential_withJets_puTarget_decorr_r9Cut_signal_IA_16/allSig_125.root', 'dipho': '/t3home/threiten/eos/Analysis/Differentials/2016ReReco/dev_differential_background_decorrFix_diphoton_sherpa_16/diphoBackgroundSherpa.root', 'gjet': '/t3home/threiten/eos/Analysis/Differentials/2016ReReco/dev_differential_background_GJets_16/GJetBackground.root'}
    elif options.year == '2017':
        paths = {'signal': '/t3home/threiten/eos/Analysis/Differentials/2017ReReco/dev_differential_withJets_puTarget_decorr_r9Cut_signal_IA_17/allSig_125CN.root', 'dipho': '/t3home/threiten/eos/Analysis/Differentials/2017ReReco/dev_differential_background_decorrFix_diphoton_sherpa_17/diphoBackgroundSherpa.root', 'gjet': '/t3home/threiten/eos/Analysis/Differentials/2017ReReco/dev_differential_background_GJets_17/GJetBackground.root'}
    elif options.year == '2018':
        paths = {'signal': '/t3home/threiten/eos/Analysis/Differentials/2018ReABCPromptDReco/dev_differential_withJets_puTarget_decorr_r9Cut_signal_IA_18/allSig_125.root', 'dipho': '/t3home/threiten/eos/Analysis/Differentials/2018ReABCPromptDReco/dev_differential_background_decorrFix_diphoton_sherpa_18/diphoBackgroundSherpa.root', 'gjet': '/t3home/threiten/eos/Analysis/Differentials/2018ReABCPromptDReco/dev_differential_background_GJets_18/GJetBackground.root'}
    else:
        raise Exception('Year has to be one of 2016, 2017, 2018')

    df_signal = root_pandas.read_root(paths['signal'],"tagsDumper/trees/InsideAcceptance_125_13TeV_SigmaMpTTag_0",columns=['CMS_hgg_mass','weight','recoLeadFull5x5_r9','recoSubleadFull5x5_r9','leadmva','subleadmva','sigmaMoM_decorr','sigmarv'])
    df_DiphoBkg = root_pandas.read_root(paths['dipho'],"tagsDumper/trees/DiPhotonJets_80_Inf_Sherpa_13TeV_SigmaMpTTag_0",columns=['CMS_hgg_mass','weight','recoLeadFull5x5_r9','recoSubleadFull5x5_r9','leadmva','subleadmva','sigmaMoM_decorr','sigmarv'])
    df_GJBkg = root_pandas.read_root(paths['gjet'],"tagsDumper/trees/GJet_MGG_80_Inf_13TeV_SigmaMpTTag_0",columns=['CMS_hgg_mass','weight','recoLeadFull5x5_r9','recoSubleadFull5x5_r9','leadmva','subleadmva','sigmaMoM_decorr','sigmarv'])
    
    cutsRess = []
    for cats in [[0.0, 0.02],[0.0, 0.01, 0.02],[0.0, 0.01, 0.015, 0.025],[0.0, 0.01, 0.015, 0.02, 0.03]]:
        cutsRess.append(findCuts(df_signal, [df_DiphoBkg, df_GJBkg], cats, 0.35, nAct=options.nActors, penEvts=options.penEvts, useAsimov=True, intLumi=options.intLumi))

    try:
        pkl.dump(cutsRess,gzip.open('{}/optRes_{}.pkl.gz'.format(options.outfolder, options.year),'wb'))
    except OSError:
        pkl.dump(cutsRess,gzip.open('/t3home/threiten/optRes_{}.pkl.gz'.format(options.year),'wb'))

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredAgrs = parser.add_argument_group()
    requiredAgrs.add_argument('--year', '-y', action='store', type=str, required=True)
    requiredAgrs.add_argument('--outfolder', '-o', action='store', type=str, required=True)
    requiredAgrs.add_argument('--redis_password', '-r', action='store', type=str, required=True)
    requiredAgrs.add_argument('--nActors', '-n', action='store', type=int, required=True)
    requiredAgrs.add_argument('--penEvts', '-p', action='store', type=int, required=True)
    requiredAgrs.add_argument('--intLumi', '-i', action='store', type=float, required=True)
    options = parser.parse_args()
    main(options)

