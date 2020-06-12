import numpy as np
import matplotlib
matplotlib.use('Agg')

import scipy.optimize as opt
import pandas as pd
import dill
import argparse
import root_pandas
import iminuit
import probfit
import scipy.integrate as integr
from joblib import Parallel, delayed
from bayes_opt import BayesianOptimization

def pyxponential(x,x0,lambd):
    return lambd*np.exp(-lambd*(x-x0))

def getSignificance(sig, bkg, plots=False, minPeak=None, maxPeak=None, debug=False):
    
    if sig.size < 100. or bkg.size < 100. or bkg[:,1].sum()<1000.:
        print('Very low population in Category, setting its significance to ZERO !')
        return 0.

    xpon = probfit.functor.Extended(pyxponential, 'N')
    gauss = probfit.pdf.gaussian
    twogauss = probfit.functor.AddPdfNorm(gauss,gauss,prefix=['g1','g2'])
    combined_pdf =  probfit.functor.AddPdf(probfit.functor.Extended(twogauss,'G'), probfit.functor.Extended(pyxponential, 'N'))
    
    
#     bkg = bkg[np.logical_and(bkg[:,0]>minPeak, bkg[:,0]<maxPeak) ,:]
    data = np.vstack((bkg[np.logical_and(bkg[:,0]>110, bkg[:,0]<140),:],sig[np.logical_and(sig[:,0]>110, sig[:,0]<140),:]))
    sig = sig[np.logical_and(sig[:,0]>110, sig[:,0]<140),:]
#     
    
    binnedLH = probfit.BinnedLH(twogauss,sig[:,0],bins=120,weights=sig[:,1],use_w2=True, extended=False)
    minu = iminuit.Minuit(binnedLH, f_0=0.5, g1mean=125.,g2mean=125., g1sigma=2., g2sigma=2., limit_g1mean=(123,127), limit_g2mean=(123,127), error_f_0 = 0.01, error_g1mean=1., error_g2mean=1., error_g1sigma=0.1, error_g2sigma=0.1, errordef=0.5)
    minu.migrad()
    if plots:
        binnedLH.show(minu)
    if debug:
        minu.print_fmin()
    dynRange = np.sqrt((minu.fitarg['f_0']*minu.fitarg['g1sigma'])**2+((1-minu.fitarg['f_0'])*minu.fitarg['g2sigma'])**2)
    sigMean = minu.fitarg['f_0']*minu.fitarg['g1mean']+(1-minu.fitarg['f_0'])*minu.fitarg['g2mean']

    if np.any(np.isnan(minu.np_errors())) or not minu.get_fmin()['is_valid']:
        print("WARNING: Two gaussian fit failed using only one")
        
        binnedLH = probfit.BinnedLH(gauss,sig[:,0],bins=120,weights=sig[:,1],use_w2=True, extended=False)  
        minu = iminuit.Minuit(binnedLH, mean=125., sigma=1., limit_mean=(123,127), error_mean=1, error_sigma=0.1, errordef=0.5)
        res = minu.migrad()
        combined_pdf =  probfit.functor.AddPdf(probfit.functor.Extended(gauss,'G'), probfit.functor.Extended(pyxponential, 'N'))
        if plots:
            binnedLH.show(minu)
        if debug:
            minu.print_fmin()
        dynRange = minu.fitarg['sigma']
        sigMean = minu.fitarg['mean']

    binnedLH_bkg = probfit.BinnedLH(pyxponential,bkg[:,0],bins=160,weights=bkg[:,1],use_w2=True, extended=False)
    minu_bkg = iminuit.Minuit(binnedLH_bkg, lambd=0.03, x0=100, error_lambd=0.003, error_x0=10., errordef=0.5)
    minu_bkg.migrad()
    if plots:
        binnedLH_bkg.show(minu_bkg)
    if debug:
        minu_bkg.print_fmin()
        
    fix_dict = {}
    for par in minu.parameters + minu_bkg.parameters:
        fix_dict['fix_{}'.format(par)]=True
    
    binnedLH_comb = probfit.BinnedLH(combined_pdf,data[:,0],bins=60,weights=data[:,1],use_w2=True, extended=True)
    minu_comb = iminuit.Minuit(binnedLH_comb, errordef=0.5, N=bkg[:,1].sum(), G=sig[:,1].sum(), error_N=bkg[:,1].sum()/10., error_G=sig[:,1].sum()/10., limit_G = (0.,2 * sig[:,1].sum()), **dict(minu.values), **dict(minu_bkg.values), **fix_dict)
    minu_comb.migrad()
    if plots:
        binnedLH_comb.show(minu_comb)
    if debug:
        minu_comb.print_fmin()
        
    if minPeak is None:
        minPeak = sigMean - dynRange
        if debug:
            print(minPeak)
    if maxPeak is None:
        maxPeak = sigMean + dynRange
        if debug:
            print(maxPeak)

    return (integr.quad(lambda x: combined_pdf(x, *minu_comb.np_values()), minPeak, maxPeak)[0] - integr.quad(lambda x: dict(minu_comb.values)['N'] * pyxponential(x, *minu_bkg.np_values()), minPeak, maxPeak)[0])/np.sqrt(integr.quad(lambda x: dict(minu_comb.values)['N'] * pyxponential(x, *minu_bkg.np_values()), minPeak, maxPeak)[0])

def selectSigBkg(dfSignal, dfBkg,IdMVACut, minSigMoM, maxSigMoM, debug=False):

    cut = 'leadmva > {0:0.5f} and subleadmva > {0:0.5f} and sigmaMoM_decorr > {1:0.5f} and sigmaMoM_decorr < {2:0.5f}'.format(IdMVACut, minSigMoM, maxSigMoM)
    if debug:
        print(cut)
    bkg = np.array(dfBkg.query(cut,engine='python').loc[:,['CMS_hgg_mass','weight']].values,dtype=np.float64)
    sig = np.array(dfSignal.query(cut,engine='python').loc[:,['CMS_hgg_mass','weight']].values, dtype=np.float64)
    return sig, bkg

def overallSig(dfSig, dfBkg, IdMVACut, sigMoMBins, debug=False, setZero=True):
    
    ovSigSq = 0.
    signals = []
    bkgs = []
    for i, binn in enumerate(sigMoMBins[:-1]):
        sig,bkg = selectSigBkg(dfSig, dfBkg, IdMVACut, binn, sigMoMBins[i+1],debug=debug)
        signals.append(sig)
        bkgs.append(bkg)

    try:
        sigfs = Parallel(n_jobs=len(signals))(delayed(getSignificance)(signals[i], bkgs[i], minPeak=None, maxPeak=None, debug=debug, plots=debug) for i in range(len(signals)))
    except:
        return 0.

    if not np.all(np.isfinite(sigfs)):
        return 0.
    
    for sigf in sigfs:
        if debug:
            print(sigf)
        ovSigSq += sigf**2
        
    if np.any(np.array(sigfs)<0.005*np.array(sigfs).max()) and setZero:
        print('Significance of one Category very low, setting overall significance to ZERO !')
        return 0.

    return np.sqrt(ovSigSq)
        

def getParams(n, vals, IdMVAC=None):
    
    # if len(vals) != n:
    #     raise Exception('vals needs to have length n')
    
    abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    argsS = ''
    callS = ''
    minuitDic = {}
    
    errs = [0.001] + (n-1) * [0.1]
    limits = [(0.005,0.05), (0., 3.)] + (n-2) * [(0., 5.)]
    for i, arg in enumerate(abc[:n]):
        argsS += '{}, '.format(arg)
        callS += '{0}={0}, '.format(arg)
        minuitDic['{}'.format(arg)] = vals[i]
        minuitDic['error_{}'.format(arg)] = errs[i]
        minuitDic['limit_{}'.format(arg)] = limits[i]

    argsS += 'IdMVACut, '
    callS += 'IdMVACut=IdMVACut, '
    minuitDic['IdMVACut'] = 0.2 + (0.9 - 0.2) * np.random.rand()
    minuitDic['error_IdMVACut'] = 0.05
    minuitDic['limit_IdMVACut'] = (0.2, 0.9)
        
    argsS = argsS[:-2]
    callS = callS[:-2]
    
    return argsS, callS, minuitDic

def overallSigCats(vec = None, **kwargs):
    
    if vec is None:
        vect = [0.0]
        for i, key in enumerate(kwargs.keys()):
            if i == 0:
                if len(key) == 1:
                    vect.append(kwargs[key])
            else:
                if len(key) == 1:
                    vect.append(vect[i]*(1+kwargs[key]))

    elif vec is not None:
        vect = [0.0]
        for i, arg in enumerate(vec):
            if i == 0:
                vect.append(arg)
            else:
                vect.append(vect[i]*(1+arg))
            
    debug=False
    if 'debug' in kwargs.keys():
        debug=kwargs['debug']

    if debug:
        print(vect)

    if 'IdMVACut' in kwargs.keys():
        IdMVACut = kwargs['IdMVACut']
    else:
        IdMVACut = 0.5
                
    if 'setZero' in kwargs.keys():
        setZero=kwargs['setZero']
    else:
        setZero=True
        
    ret = -overallSig(df_signal, df_bkg, IdMVACut, vect, setZero=setZero)

    if debug:
        print('OverallSignificance: {}'.format(ret))
        print('--------------------------------------------------------------')
        
    return ret

def getRangesBayOpt(n):

    abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    argsS = ''
    callS = ''
    bayOptDic = {}
    
    ranges = [(0.005,0.02), (0.1, 1)] + (n-2) * [(0.1, 2.5)]
    for i, arg in enumerate(abc[:n]):
        argsS += '{}, '.format(arg)
        callS += '{0}={0}, '.format(arg)
        bayOptDic['{}'.format(arg)] = ranges[i]
        
    argsS += 'IdMVACut, '
    callS += 'IdMVACut=IdMVACut, '
    bayOptDic['IdMVACut'] = (0.2, 0.9)
    
    argsS = argsS[:-2]
    callS = callS[:-2]

    return argsS, callS, bayOptDic

def getRanges(n, first):

    if n >= 2:
        ret = [(first,2*first), (0.1,0.5)] + (n-2) * [(0.1,1.5)]
    elif n == 1:
        ret = [(first,2*first)]
    print(ret)
    return ret

def getRandomGrid(ndim, ranges, Ns=None):
    
    if Ns is None:
        Ns = ndim
    
    ranGrid = np.random.rand(Ns**ndim, ndim)
    for i in range(ndim):
        ranGrid[:,i] = ranGrid[:,i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
        
    return ranGrid

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-o','--outdir', action='store', type=str, required=True)
    requiredArgs.add_argument('-s','--signal', action='store', type=str, required=True)
    requiredArgs.add_argument('-d','--DiphoBackground', action='store', type=str, required=True)
    requiredArgs.add_argument('-g','--GJetBackground', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--numCats', action='store', type=int, required=True)
    requiredArgs.add_argument('-f','--firstCat', action='store', type=float, required=True)
    requiredArgs.add_argument('-m','--method', action='store', default='bayOpt', type=str, required=True)
    options = parser.parse_args()
    
    print('Loading {}'.format(options.signal))
    df_signal = root_pandas.read_root(options.signal,"tagsDumper/trees/InsideAcceptance_125_13TeV_SigmaMpTTag_0",columns=['CMS_hgg_mass','weight','recoLeadFull5x5_r9','recoSubleadFull5x5_r9','leadmva','subleadmva','sigmaMoM_decorr'])
    
    print('Loading {}'.format(options.DiphoBackground))
    df_DiphoBkg = root_pandas.read_root(options.DiphoBackground,"tagsDumper/trees/DiPhotonJets_80_Inf_Sherpa_13TeV_SigmaMpTTag_0", columns=['CMS_hgg_mass','weight','recoLeadFull5x5_r9','recoSubleadFull5x5_r9','leadmva','subleadmva','sigmaMoM_decorr'])

    print('Loading {}'.format(options.GJetBackground))
    df_GJetsBkg = root_pandas.read_root(options.GJetBackground,"tagsDumper/trees/GJet_MGG_80_Inf_13TeV_SigmaMpTTag_0",columns=['CMS_hgg_mass','weight','recoLeadFull5x5_r9','recoSubleadFull5x5_r9','leadmva','subleadmva','sigmaMoM_decorr'])

    df_bkg = pd.concat([df_DiphoBkg,df_GJetsBkg], ignore_index=True)
    
    # bestP, fval, grid, jout = opt.brute(overallSigCats, getRanges(options.numCats, options.firstCat), full_output=True, workers=-1)
    # dill.dump(grid, open('{}/bruteGrid_{}_{}.dill'.format(options.outdir, options.numCats, options.firstCat), mode='wb'))
    # dill.dump(jout, open('{}/bruteJout_{}_{}.dill'.format(options.outdir, options.numCats, options.firstCat), mode='wb'))
    
    if options.method == 'randMigrad':
        print('Using method migrad with random starting point')
        randP = getRandomGrid(options.numCats, getRanges(options.numCats, options.firstCat), Ns=1)
        randP = randP.flatten()
        argsS, callS, minuitDic = getParams(options.numCats, list(randP))
        print(argsS)
        print(callS)
        print(minuitDic)
        exec('minuCategorization = iminuit.Minuit(lambda {}: overallSigCats({}, setZero=False), **minuitDic)'.format(argsS, callS))
        minuCategorization.migrad()
 
        paramsFval = (minuCategorization.np_values(),np.array([minuCategorization.fval]))
        dill.dump(paramsFval, open('{}/migradCat_{}_{}_{}.dill'.format(options.outdir, options.numCats, options.firstCat, randP), mode='wb'))
        minuCategorization.print_fmin()
        minuCategorization.print_param()

    elif options.method == 'bayOpt':
        print('Using method bayesian optimization')
        argsB, callB, rangesB = getRangesBayOpt(options.numCats)
        print(argsB)
        print(callB)
        print(rangesB)
        exec('bOpt = BayesianOptimization(lambda {}: -overallSigCats({}), pbounds=rangesB, verbose=2)'.format(argsB, callB))
        try:
            bOpt.maximize(n_iter=5000, init_points=10)
        except:
            dill.dump(bOpt, open('{}/bayOpt_{}_{}_inc.dill'.format(options.outdir, options.numCats, options.firstCat), mode='wb'))
            raise
        
        dill.dump(bOpt, open('{}/bayOpt_{}_{}.dill'.format(options.outdir, options.numCats, options.firstCat), mode='wb'))
