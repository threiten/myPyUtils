import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

import scipy.optimize as opt
import pandas as pd
import dill
import argparse
import root_pandas
import iminuit
import probfit
import scipy.integrate as integr
# import asyncio
from joblib import Parallel, delayed
from bayes_opt import BayesianOptimization


def pyxponential(x,lambd):
    return lambd*np.exp(-lambd*x)

def powerLaw(x, b):
    return np.power(x,-b)
    
def getSignificance(sig, bkg, minPeak=None, maxPeak=None, debug=False, plots=False, penEvts=1000., intLumi=None, retErr=False, savePlots=None, useAsimov=False):
        
    if sig.size < 10. or bkg.size < 10. or bkg[:,1].sum()<penEvts/2.:
        if debug:
            print('Very low population in Category, setting its significance to ZERO !')
        return 0.
    
    if savePlots is not None:
        plots=False

    prefct = 1/(1+np.exp(-0.05*(bkg[:,1].sum()-penEvts)))
    if debug:
        print('Prefactor: ', prefct)
        print('Sig weight: ', sig[:,1].sum())
        print('Bkg weight: ', bkg[:,1].sum())

    if intLumi is not None:
        bkg[:,1] = intLumi * bkg[:,1]
        sig[:,1] = intLumi * sig[:,1]

    Nsig = sig[:,1].sum()
    Nbkg = bkg[:,1].sum()
    Ntot = Nsig + Nbkg
    print(Nsig/Ntot)

    xpon = probfit.functor.Extended(pyxponential, 'N')
    gauss = probfit.pdf.gaussian
    cBall = probfit.functor.Normalized(probfit.pdf.crystalball, bound=(105.,145.))
    powLNorm = probfit.functor.Normalized(powerLaw, bound=(100.,180.))
    twogauss = probfit.functor.AddPdfNorm(gauss,gauss,prefix=['g1','g2'])
    # combined_pdf =  probfit.functor.AddPdf(probfit.functor.Extended(twogauss,'G'), probfit.functor.Extended(pyxponential, 'N'))
    combinedPdf = probfit.functor.AddPdfNorm(twogauss, powLNorm, facname=['G'])

    bkg = bkg[np.logical_and(bkg[:,0]>100, bkg[:,0]<180) ,:]
    data = np.vstack((bkg[np.logical_and(bkg[:,0]>100, bkg[:,0]<180),:],sig[np.logical_and(sig[:,0]>100, sig[:,0]<180),:]))
    sig = sig[np.logical_and(sig[:,0]>105, sig[:,0]<145),:]
    
    binnedLH = probfit.BinnedLH(twogauss,sig[:,0],bins=80,weights=sig[:,1],use_w2=True, extended=False)
    minu = iminuit.Minuit(binnedLH, f_0=0.2, g1mean=125.,g2mean=125., g1sigma=4., g2sigma=2., limit_g1mean=(123,127), limit_g2mean=(123,127), limit_f_0=(0.,1.), limit_g1sigma=(1.,6.), limit_g2sigma=(0.5,4.), error_f_0 = 0.05, error_g1mean=1., error_g2mean=1., error_g1sigma=0.5, error_g2sigma=0.25, errordef=0.5)
    minu.migrad()
    fSig = twogauss
    if debug>1:
        minu.print_fmin()
    if plots:
        binnedLH.show(minu)
    if savePlots is not None:
        binnedLH.draw(minu)
        plt.savefig('{}_Sig.pdf'.format(savePlots))
        plt.savefig('{}_Sig.png'.format(savePlots))
        plt.close()
        
    
    dynRange = 2*np.sqrt(minu.fitarg['f_0']*minu.fitarg['g1sigma']**2+(1-minu.fitarg['f_0'])*minu.fitarg['g2sigma']**2)
    sigMean = minu.fitarg['f_0']*minu.fitarg['g1mean']+(1-minu.fitarg['f_0'])*minu.fitarg['g2mean']

    if np.any(np.isnan(minu.np_errors())):
        return 0.

    # binnedLH = probfit.BinnedLH(cBall,sig[:,0],bins=80,weights=sig[:,1],use_w2=True, extended=False)
    # minu = iminuit.Minuit(binnedLH, alpha=2., n=10., mean=125., sigma=1., fix_alpha=True, error_n=1., error_mean=1., error_sigma=0.1, errordef=0.5)
    # minu.migrad()
    # fSig = cBall
    # combinedPdf = probfit.functor.AddPdfNorm(cBall, powLNorm, facname=['G'])
    # if debug>1:
    #     minu.print_fmin()
    # if plots:
    #     binnedLH.show(minu)
    # if savePlots is not None:
    #     binnedLH.draw(minu)
    #     plt.savefig('{}_Sig.pdf'.format(savePlots))
    #     plt.savefig('{}_Sig.png'.format(savePlots))
    #     plt.close()
        
    
    # dynRange = 2*minu.fitarg['sigma']
    # sigMean = minu.fitarg['mean']

    # if np.any(np.isnan(minu.np_errors())):
    #     return 0.
    # if np.any(np.isnan(minu.np_errors())) or not minu.get_fmin()['is_valid']:
    #     print("WARNING: Two gaussian fit failed using only one")
        
    # binnedLH = probfit.BinnedLH(gauss,sig[:,0],bins=80,weights=sig[:,1],use_w2=True, extended=False)  
    # minu = iminuit.Minuit(binnedLH, mean=125., sigma=1., limit_mean=(123,127), error_mean=1, error_sigma=0.1, errordef=0.5)
    # res = minu.migrad()
    # fSig = gauss
    # combinedPdf = probfit.functor.AddPdfNorm(gauss, powLNorm, facname=['G'])
    # if plots:
    #     binnedLH.show(minu)
    # if savePlots is not None:
    #     binnedLH.draw(minu)
    #     plt.savefig('{}_Sig.pdf'.format(savePlots))
    #     plt.savefig('{}_Sig.png'.format(savePlots))
    #     plt.close()
    # if debug>1:
    #     minu.print_fmin()
    # dynRange = minu.fitarg['sigma']
    # sigMean = minu.fitarg['mean']

    # if np.any(np.isnan(minu.np_errors())):
    #     return 0.
    #     if np.any(np.isnan(minu.np_errors())) or not minu.get_fmin()['is_valid']:
    #         print("WARNING: One gaussian fit failed using two again")
            
    #         binnedLH = probfit.BinnedLH(twogauss,sig[:,0],bins=80,weights=sig[:,1],use_w2=True, extended=False)
    #         minu = iminuit.Minuit(binnedLH, f_0=0.5, g1mean=125.,g2mean=125., g1sigma=2., g2sigma=2., limit_g1mean=(123,127), limit_g2mean=(123,127), error_f_0 = 0.01, error_g1mean=1., error_g2mean=1., error_g1sigma=0.1, error_g2sigma=0.1, errordef=0.5)
    #         minu.migrad()
    #         fSig = twogauss
    #         if debug>1:
    #             minu.print_fmin()
    #         if plots:
    #             binnedLH.show(minu)
    #         if savePlots is not None:
    #             binnedLH.draw(minu)
    #             plt.savefig('{}_Sig.pdf'.format(savePlots))
    #             plt.savefig('{}_Sig.png'.format(savePlots))
    #             plt.close()

    #         dynRange = 2*np.sqrt(minu.fitarg['f_0']*minu.fitarg['g1sigma']**2+(1-minu.fitarg['f_0'])*minu.fitarg['g2sigma']**2)
    #         sigMean = minu.fitarg['f_0']*minu.fitarg['g1mean']+(1-minu.fitarg['f_0'])*minu.fitarg['g2mean']
    #         combinedPdf = probfit.functor.AddPdfNorm(twogauss, powLNorm, facname=['G'])

    #         if np.any(np.isnan(minu.np_errors())):
    #             raise RuntimeError('Signal fit failed. Result would be useless.')

    binnedLH_bkg = probfit.BinnedLH(powLNorm,bkg[:,0],bins=160,weights=bkg[:,1],use_w2=True, extended=False)
    minu_bkg = iminuit.Minuit(binnedLH_bkg, b=1., error_b=0.1, errordef=0.5)
    # minu_bkg = iminuit.Minuit(binnedLH_bkg, lambd=0.05, x0=105, error_lambd=0.005, errordef=0.5, fix_x0=True)
    minu_bkg.migrad()
    if plots:
        binnedLH_bkg.show(minu_bkg)
    if savePlots is not None:
        binnedLH_bkg.draw(minu_bkg)
        plt.savefig('{}_Bkg.pdf'.format(savePlots))
        plt.savefig('{}_Bkg.png'.format(savePlots))
        plt.close()
    if debug>1:
        minu_bkg.print_fmin()
        
    fix_dict = {}
    for par in minu.parameters + minu_bkg.parameters:
        fix_dict['fix_{}'.format(par)]=True
    
    # binnedLH_comb = probfit.BinnedLH(combinedPdf, data[:,0], bins=100, weights=data[:,1], use_w2=True, extended=False)
    # minu_comb = iminuit.Minuit(binnedLH_comb, errordef=0.5, N=bkg[:,1].sum(), G=sig[:,1].sum(), error_N=bkg[:,1].sum()/10., error_G=sig[:,1].sum()/10., limit_G = (0.,2 * sig[:,1].sum()), **dict(minu.values), **dict(minu_bkg.values), **fix_dict)

    if useAsimov:
        bins = np.linspace(100.,180.,161)
        xc = 0.5*(bins[1:]+bins[:-1])
        binw = xc[1] - xc[0]
        histBkg = bkg[:,1].sum() * binw * np.array([powLNorm(ent, *minu_bkg.np_values()) for ent in xc])
        histSig = sig[:,1].sum() * binw * np.array([fSig(ent, *minu.np_values()) for ent in xc])
        histAsi = histSig + histBkg
        binnedLH_comb = probfit.BinnedLH(combinedPdf, xc, bins=160, weights=histAsi, extended=False)
    else:
        binnedLH_comb = probfit.BinnedLH(combinedPdf, data[:,0], bins=160, weights=data[:,1], use_w2=True, extended=False)

    minu_comb = iminuit.Minuit(binnedLH_comb, errordef = 0.5, G = Nsig/Ntot, error_G = Nsig/(Ntot*10), limit_G = (0., 1.), **dict(minu.values), **dict(minu_bkg.values), **fix_dict)
    minu_comb.migrad()
    if retErr:
        print('Running Minos!')
        minu_comb.minos()
    if plots:
        binnedLH_comb.show(minu_comb)
    if savePlots is not None:
        binnedLH_comb.draw(minu_comb)
        plt.savefig('{}_Comb.pdf'.format(savePlots))
        plt.savefig('{}_Comb.png'.format(savePlots))
        plt.close()
    if debug>1:
        minu_comb.print_fmin()
            
    if minPeak is None:
        minPeak = sigMean - dynRange
        if debug:
            print(minPeak)
    if maxPeak is None:
        maxPeak = sigMean + dynRange
        if debug:
            print(maxPeak)

        
    BInt = integr.quad(lambda x: (1 - dict(minu_comb.values)['G']) * powLNorm(x, *minu_bkg.np_values()), minPeak, maxPeak)[0]
    if retErr:
        print(minu_comb.get_merrors())
        err = np.sqrt((1/BInt) * (Ntot * dict(minu_comb.errors)['G']**2 + dict(minu_comb.values)['G']**2))

    sig = np.sqrt(Ntot) * prefct * (integr.quad(lambda x: combinedPdf(x, *minu_comb.np_values()), minPeak, maxPeak)[0] - integr.quad(lambda x: (1 - dict(minu_comb.values)['G']) * powLNorm(x, *minu_bkg.np_values()), minPeak, maxPeak)[0])/np.sqrt(BInt)
    
    if retErr:
        return sig, err
    else:
        return sig


    
class Optimizer(object):

    def __init__(self, dfSig, dfsBkg, cutVar='sigmaMoM_decorr', addVars=['leadmva', 'subleadmva'], debug=False, plotFits=False, setZero=True):

        self.dfSig = dfSig.loc[:, [cutVar, 'CMS_hgg_mass', 'weight'] + addVars ]
        self.dfBkg = dfsBkg[0].loc[:, [cutVar, 'CMS_hgg_mass', 'weight'] + addVars ]
        for dfBackground in dfsBkg[1:]:
            self.dfBkg = pd.concat([self.dfBkg, dfBackground.loc[:, [cutVar, 'CMS_hgg_mass', 'weight'] + addVars ]], ignore_index=True)

        self.cutVar = cutVar
        self.debug = debug
        self.plots = plotFits
        self.setZero = setZero

    
    def selectSigBkg(self, cutLow, cutHigh, addCut=None):

        if len(addCut) is not None:
            cut = addCut + ' and {0} > {1} and {0} < {2}'.format(self.cutVar, cutLow, cutHigh)
        else:
            cut = '{0} > {1} and {0} < {2}'.format(self.cutVar, cutLow, cutHigh)
        if self.debug:
            print(cut)
            
        bkg = np.array(self.dfBkg.query(cut,engine='python').loc[:,['CMS_hgg_mass','weight']].values,dtype=np.float64)
        sig = np.array(self.dfSig.query(cut,engine='python').loc[:,['CMS_hgg_mass','weight']].values, dtype=np.float64)
        
        return sig, bkg

    
    def sigOneCat(self, cutLow, cutHigh, addCut):

        sig, bkg = self.selectSigBkg(cutLow, cutHigh, addCut)
        ret = self.getSignificance(sig, bkg, minPeak=None, maxPeak=None)
        if self.debug:
            print(ret)
        return ret

    
    def sigMultCat(self, cuts, addCut, retErr=False, savePlots=None, **kwargs):

        ovSigSq = 0.
        signals = []
        bkgs = []
        for i, binn in enumerate(cuts[:-1]):
            sig,bkg = self.selectSigBkg(binn, cuts[i+1], addCut)
            signals.append(sig)
            bkgs.append(bkg)
    
        # sigfs = Parallel(n_jobs=len(signals))(delayed(getSignificance)(signals[i], bkgs[i]) for i in range(len(signals)))
        if savePlots is not None:
            sigfs = [getSignificance(signals[i], bkgs[i], debug=self.debug, plots=self.plots, retErr=retErr, savePlots='{}_Cat{}'.format(savePlots,i), **kwargs) for i in range(len(signals))]
        else:
            sigfs = [getSignificance(signals[i], bkgs[i], debug=self.debug, plots=self.plots, retErr=retErr, **kwargs) for i in range(len(signals))]
        if retErr:
            errs = [sigf[1] for sigf in sigfs]
            sigfs = [sigf[0] for sigf in sigfs]

        for sigf in sigfs:
            if self.debug:
                print(sigf)
            ovSigSq += sigf**2
        
        ovSig = np.sqrt(ovSigSq)

        if retErr:
            ovErrSq = 0.
            for i, err in enumerate(errs):
                if self.debug:
                    print(err)
                ovErrSq += (sigfs[i]**2/ovSigSq)*err**2
            ovErr = np.sqrt(ovErrSq)

        if np.any(np.array(sigfs)<0.005*np.array(sigfs).max()) and self.setZero:
            if self.debug:
                print('Significance of one Category very low, setting overall significance to ZERO !')
            return 0.

        if self.debug:
            print(ovSig)
            if retErr:
                print(ovErr)
        
        if retErr:
            return ovSig, ovErr
        else:
            return ovSig

    # async def sigMultCatAsync(self, cuts, addCut, **kwargs):
        
    #     task = asyncio.ensure_future(self.sigMultCat(cuts, addCut, **kwargs))
    #     res = await task
    #     return res

    def optimizeOneCut(self, start, cutOpt, indOpt, addCut):

        cuts = start
        cuts[indOpt] = cutOpt
        return self.sigMultCat(cuts, addCut)
        
    
    def iterativeOptim(self, nCats, addCut, start):

        currMin = 0.
        ovSigSq = 0.
        nCat = 0
        cats = []
        currCut = start
        while nCat < nCats:
            minuCategorization = iminuit.Minuit(lambda cut: -self.optimizeOneCut(currCut, cut, nCat+1, addCut=addCut), cut=currCut[nCat+1], error_cut=0.5*currCut[nCat+1], limit_cut=(currCut[nCat], None))
            minuCategorization.migrad()
            optimCut = minuCategorization.np_values()
            fval = np.array([minuCategorization.fval])
            ovSigSq += fval**2
            # cats.append(optimCut)
            currMin = optimCut[0]
            currCut[nCat + 1] = currMin
            if self.debug:
                print('Significance: ', fval)
                print('Cut Value: ', optimCut[0])
            nCat += 1

        ovSig = self.sigMultCat(currCut, addCut)

        return currCut, ovSig

        
