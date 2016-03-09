"""

Usage:
test_empca [-hvx] [-i ITER] [-s SEED] [-e ELEMS] [-m MAXVEC] [-n NVECS]

Options:
    -h, --help
    -v, --verbose
    -x, --hidefigs                      Option to hide figures
    -i ITER, --iter ITER                Number of iterations to perform [default: 1]
    -s SEED, --seed SEED                Random seed initializer [default: 1]
    -e ELEMS, --elems ELEMS             List of elements to construct false residuals [default: C]
    -n NVECS, --nvecs NVECS             Number of eigenvectors to use [default: 5]
    -m MAXVEC, --maxvec MAXVEC          Maximum number of vectors to display [default: 1]
"""

import matplotlib.pyplot as plt
import numpy as np
import access_spectrum as acs
import polyfit as pf
import os
import run_empca
reload(run_empca)
from run_empca import *
from comp_empca import gen_colours

windowinfo = 'pickles/windowinfo.pkl'
elemwindows,window_all,window_peak,windowPeaks,windowPixels,tophats = acs.pklread(windowinfo)


def make_specs(specs,errs,elemlist,proportion='default',scaling=None):
    SNR = specs/errs
    if proportion == 'default':
        proportion = np.linspace(0.9,1.01,len(specs))
    if proportion == 'real':
        proportion = np.ma.median(specs,axis=1)
    vec = np.zeros(aspcappix)
    for ind in range(len(elemlist)):
        if not scaling:
            vec += elemwindows[elemlist[ind]]
        elif scaling:
            vec += elemwindows[elem[ind]]*scaling[ind]
    newspecs = np.ma.masked_array(np.outer(proportion,vec),specs.mask)
    newspecs = (1-newspecs)
    noise =errs# newspecs/SNR
    drawn_noise = noise*np.random.randn(noise.shape[0],noise.shape[1])
    newspecs += drawn_noise
    newspecs.mask[np.where(noise<1e-10)] = True
    noise.mask[np.where(noise<1e-10)] = True
    return newspecs,noise

def vec_weight(elem,vec):
    w = elemwindows[elem]
    nw = np.ma.masked_array(pf.normweights(w))
    return np.ma.sum(nw*vec)
    
def arr_weight(elem,arr):
    w = elemwindows[elem]
    nw = np.ma.masked_array(pf.normweights(w))
    nws = np.tile(nw,(arr.shape[0],1))
    return np.ma.sum(nws*arr,axis=1)
    
def test_run(specs,noise,deltR2=2e-3,nvecs=5,mad=False,maxvec=5,seed=1):
    m1,m2,w1,w2 = pix_empca(None,specs.T,noise,'test.pkl',nvecs=nvecs,deltR2=2e-3,gen=True,usemad=mad,randseed=seed)
    R2_1 = R2(m1) #must be here (and not below resize) to avoid error
    R2_2 = R2(m2)
    print R2_1,R2_2
    R2_noise2 = R2noise(w2,m2,usemad=mad)[0]
    resize_pix_eigvecs(specs.T,m1,nstars=5,nvecs=nvecs)
    resize_pix_eigvecs(specs.T,m2,nstars=5,nvecs=nvecs)
    m1elem = np.zeros((nvecs,len(elems)))
    m2elem = np.zeros((nvecs,len(elems)))
    specs_weight = np.ma.masked_array(np.zeros((len(elems),specs.shape[0])))
    noise_weight = np.ma.masked_array(np.zeros((len(elems),specs.shape[0])))
    for ind in range(len(elems)):
        specs_weight[ind] = arr_weight(elems[ind],specs)
        noise_weight[ind] = arr_weight(elems[ind],noise)
        for vec in range(nvecs):
            m1elem[vec][ind] = vec_weight(elems[ind],m1.eigvec[vec])
            m2elem[vec][ind] = vec_weight(elems[ind],m2.eigvec[vec])
    specs_weight.mask[np.where(noise_weight<1e-10)] = True
    noise_weight.mask[np.where(noise_weight<1e-10)] = True
    m3,m4,w3,w4 = elem_empca(None,specs_weight,noise_weight,'test2.pkl',nvecs=nvecs,gen=True,deltR2=2e-3,usemad=mad,randseed=seed)        
    R2_3 = R2(m3)
    R2_4 = R2(m4)
    print R2_3,R2_4
    R2_noise4 = R2noise(w4,m4,usemad=mad)[0]
    resize_pix_eigvecs(specs_weight,m3,nstars=5,dim2=len(elems),nvecs=nvecs)
    resize_pix_eigvecs(specs_weight,m4,nstars=5,dim2=len(elems),nvecs=nvecs)
    for n in range(maxvec):
        plt.figure(figsize=(12,3))
        plt.axhline(0,linestyle='--',color='k',linewidth=3)
        plt.plot(norm_eigvec(m1elem[n]),'o',markersize=8,label='Pixel unweighted')
        plt.plot(norm_eigvec(m2elem[n]),'o',markersize=8,label='Pixel weighted')
        plt.plot(norm_eigvec(m3.eigvec[n]),'o',markersize=8,label='Element unweighted')
        plt.plot(norm_eigvec(m4.eigvec[n]),'o',markersize=8,label='Element weighted')
        plt.xticks(range(len(elems)),elems)
        plt.legend(loc='best',fontsize=10)
        plt.ylabel('Eigenvenctor {0}'.format(n+1))
        plt.xlim(-1,len(elems)+1)
        plt.ylim(-1,1)
    plt.figure(figsize=(12,3))
    plt.plot(R2_1,marker='o',linewidth = 3,markersize=8,label='Pixel unweighted')
    plt.plot(R2_2,marker='o',linewidth = 3,markersize=8,label='Pixel weighted')
    plt.axhline(R2_noise2,linestyle='--',color='b',linewidth=3,label='R2n_pix = {0:2f}'.format(R2_noise2))
    plt.fill_between(range(nvecs+1),R2_noise2,1,color='b',alpha=0.2)
    plt.plot(R2_3,marker='o',linewidth = 3,markersize=8,label='Element unweighted')
    plt.plot(R2_4,marker='o',linewidth = 3,markersize=8,label='Element weighted')
    plt.axhline(R2_noise4,linestyle='--',color='r',linewidth=3,label='R2n_elem = {0:2f}'.format(R2_noise4))
    plt.fill_between(range(nvecs+1),R2_noise4,1,color='r',alpha=0.2)
    plt.ylim(0,1)
    plt.legend(loc='best',fontsize=10)
    
def test_run_comp(specs,noise,iteration,axs,colours,seed=1,deltR2=2e-3,nvecs=5,mad=True,maxvec=5):
    m1,m2,w1,w2 = pix_empca(None,specs.T,noise,'test.pkl',nvecs=nvecs,deltR2=2e-3,gen=True,usemad=mad,randseed=seed)
    R2_1 = R2(m1) #must be here (and not below resize) to avoid error
    R2_2 = R2(m2)
    R2_noise2 = R2noise(w2,m2,usemad=mad)[0]
    resize_pix_eigvecs(specs.T,m1,nstars=5,nvecs=nvecs)
    resize_pix_eigvecs(specs.T,m2,nstars=5,nvecs=nvecs)
    m1elem = np.zeros((nvecs,len(elems)))
    m2elem = np.zeros((nvecs,len(elems)))
    specs_weight = np.ma.masked_array(np.zeros((len(elems),specs.shape[0])))
    noise_weight = np.ma.masked_array(np.zeros((len(elems),specs.shape[0])))
    for ind in range(len(elems)):
        specs_weight[ind] = arr_weight(elems[ind],specs)
        noise_weight[ind] = arr_weight(elems[ind],noise)
        for vec in range(nvecs):
            m1elem[vec][ind] = vec_weight(elems[ind],m1.eigvec[vec])
            m2elem[vec][ind] = vec_weight(elems[ind],m2.eigvec[vec])
    specs_weight.mask[np.where(noise_weight<1e-10)] = True
    noise_weight.mask[np.where(noise_weight<1e-10)] = True
    m3,m4,w3,w4 = elem_empca(None,specs_weight,noise_weight,'test2.pkl',nvecs=nvecs,gen=True,deltR2=2e-3,usemad=mad,randseed=seed)        
    R2_3 = R2(m3)
    R2_4 = R2(m4)
    R2_noise4 = R2noise(w4,m4,usemad=mad)[0]
    resize_pix_eigvecs(specs_weight,m3,nstars=5,dim2=len(elems),nvecs=nvecs)
    resize_pix_eigvecs(specs_weight,m4,nstars=5,dim2=len(elems),nvecs=nvecs)
    ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8 = axs
    for n in range(maxvec):
    
        ax1.plot(norm_eigvec(m1elem[n]),'o',color=colours[iteration],markersize=8,label=iteration)

    
        ax2.axhline(0,linestyle='--',color='k',linewidth=3)
        ax2.plot(norm_eigvec(m2elem[n]),'o',color=colours[iteration],markersize=8,label=iteration)
        ax2.set_title('Pixel weighted')
        ax2.set_xticks(range(len(elems)),elems)
        ax2.legend(loc='best',fontsize=10)
        ax2.set_ylabel('Eigenvenctor {0}'.format(n+1))
        ax2.set_xlim(-1,len(elems)+1)
        ax2.set_ylim(-1,1)
    
        ax3.axhline(0,linestyle='--',color='k',linewidth=3)
        ax3.plot(norm_eigvec(m3.eigvec[n]),'o',color=colours[iteration],markersize=8,label=iteration)
        ax3.set_title('Element unweighted')
        ax3.set_xticks(range(len(elems)),elems)
        ax3.legend(loc='best',fontsize=10)
        ax3.set_ylabel('Eigenvenctor {0}'.format(n+1))
        ax3.set_xlim(-1,len(elems)+1)
        ax3.set_ylim(-1,1)
    
        ax4.axhline(0,linestyle='--',color='k',linewidth=3)
        ax4.plot(norm_eigvec(m4.eigvec[n]),'o',color=colours[iteration],markersize=8,label=iteration)
        ax4.set_title('Element weighted')
        ax4.set_xticks(range(len(elems)),elems)
        ax4.legend(loc='best',fontsize=10)
        ax4.set_ylabel('Eigenvenctor {0}'.format(n+1))
        ax4.set_xlim(-1,len(elems)+1)
        ax4.set_ylim(-1,1)

    ax5.set_title('Pixel unweighted')
    ax5.plot(R2_1,marker='o',color=colours[iteration],linewidth = 3,markersize=8)
    ax5.axhline(R2_noise2,linestyle='--',color='b',linewidth=3,label='i {0} R2n_pix = {1:2f}'.format(iteration,R2_noise2))
    ax5.fill_between(range(nvecs+1),R2_noise2,1,color='b',alpha=0.2)
    ax5.set_ylabel('R2')
    ax5.set_xlabel('Number of EMPCA vectors')
    ax5.legend(loc='best',fontsize=10)

    ax6.set_title('Pixel weighted')
    ax6.plot(R2_2,marker='o',color=colours[iteration],linewidth = 3,markersize=8)
    ax6.axhline(R2_noise2,linestyle='--',color='b',linewidth=3,label='i {0} R2n_pix = {1:2f}'.format(iteration,R2_noise2))
    ax6.fill_between(range(nvecs+1),R2_noise2,1,color='b',alpha=0.2)
    ax6.set_ylabel('R2')
    ax6.set_xlabel('Number of EMPCA vectors')
    ax6.legend(loc='best',fontsize=10)

    ax7.set_title('Element unweighted')
    ax7.plot(R2_3,marker='o',color=colours[iteration],linewidth = 3,markersize=8)
    ax7.axhline(R2_noise4,linestyle='--',color='r',linewidth=3,label='i {0} R2n_elem = {1:2f}'.format(iteration,R2_noise4))
    ax7.fill_between(range(nvecs+1),R2_noise4,1,color='r',alpha=0.2)
    ax7.set_ylabel('R2')
    ax7.set_xlabel('Number of EMPCA vectors')
    ax7.legend(loc='best',fontsize=10)

    ax8.set_title('Element weighted')
    ax8.plot(R2_4,marker='o',color=colours[iteration],linewidth = 3,markersize=8)
    ax8.axhline(R2_noise4,linestyle='--',color='r',linewidth=3,label='i {0} R2n_elem = {1:2f}'.format(iteration,R2_noise4))
    ax8.fill_between(range(nvecs+1),R2_noise4,1,color='r',alpha=0.2)
    ax8.set_ylabel('R2')
    ax8.set_xlabel('Number of EMPCA vectors')
    ax8.legend(loc='best',fontsize=10)
    return ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8

if __name__=='__main__':

    arguments = docopt.docopt(__doc__)

    verbose = arguments['--verbose']
    hide = arguments['--hidefigs']
    iters = int(arguments['--iter'])
    nvecs = int(arguments['--nvecs'])
    maxvec = int(arguments['--maxvec'])
    elemlist = arguments['--elems']
    elemlist = elemlist.split(',')
    seeds = arguments['--seed']
    seeds = seeds.split(',')
    if len(seeds) == 1:
        seeds = [int(seeds[0])]*iters
    else:
        seeds = np.array(seeds).astype(int)
    
    plt.ion()

    specs = acs.pklread('red_clump/pickles/spectra_FE_H_u-0.4_d-0.5.pkl')[0]
    errs = acs.pklread('red_clump/pickles/errs_FE_H_u-0.4_d-0.5.pkl')

    if iters==1:
        falsespecs,noise = make_specs(specs,errs,elemlist)
        falsespecs -= np.mean(falsespecs,axis=0)
        # switched from all statement because it wasn't working - why?
        if np.sum(falsespecs.mask)==falsespecs.shape[0]*falsespecs.shape[1]:
            print 'All masked'
        elif not np.sum(falsespecs.mask)==falsespecs.shape[0]*falsespecs.shape[1]:
            test_run(falsespecs,noise,maxvec=maxvec,nvecs=nvecs,seed=seeds[0])

    elif iters != 1:
        f1 = plt.figure(1,figsize=(12,3))
        ax1 = f1.gca()
        ax1.axhline(0,linestyle='--',color='k',linewidth=3)
        f2 = plt.figure(2,figsize=(12,3))
        ax2 = f2.gca()
        ax2.axhline(0,linestyle='--',color='k',linewidth=3)
        f3 = plt.figure(3,figsize=(12,3))
        ax3 = f3.gca()
        ax3.axhline(0,linestyle='--',color='k',linewidth=3)
        f4 = plt.figure(4,figsize=(12,3))
        ax4 = f4.gca()
        ax4.axhline(0,linestyle='--',color='k',linewidth=3)
        f5 = plt.figure(5,figsize=(12,3))
        ax5 = f5.gca()
        f6 = plt.figure(6,figsize=(12,3))
        ax6 = f6.gca()
        f7 = plt.figure(7,figsize=(12,3))
        ax7 = f7.gca()
        f8 = plt.figure(8,figsize=(12,3))
        ax8 = f8.gca()
        axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
        colours = gen_colours(iters)
        for i in range(iters):
            falsespecs,noise = make_specs(specs,errs,elemlist)
            falsespecs -= np.mean(falsespecs,axis=0)
            if np.sum(falsespecs.mask)==falsespecs.shape[0]*falsespecs.shape[1]:
                print 'All masked'
            elif not np.sum(falsespecs.mask)==falsespecs.shape[0]*falsespecs.shape[1]:
                axs = test_run_comp(falsespecs,noise,i,axs,colours,maxvec=maxvec,nvecs=nvecs,seed=seeds[i])
        ax1.legend(loc='best',fontsize=10)
        ax1.set_ylabel('Eigenvenctor {0}'.format(maxvec+1))
        ax1.set_xlim(-1,len(elems)+1)
        ax1.set_ylim(-1,1)
        ax1.set_title('Pixel unweighted')
        ax1.set_xticks(range(len(elems)),elems)

        ax2.legend(loc='best',fontsize=10)
        ax2.set_ylabel('Eigenvenctor {0}'.format(maxvec+1))
        ax2.set_xlim(-1,len(elems)+1)
        ax2.set_ylim(-1,1)
        ax2.set_title('Pixel weighted')
        ax2.set_xticks(range(len(elems)),elems)

        ax3.legend(loc='best',fontsize=10)
        ax3.set_ylabel('Eigenvenctor {0}'.format(maxvec+1))
        ax3.set_xlim(-1,len(elems)+1)
        ax3.set_ylim(-1,1)
        ax3.set_title('Element unweighted')
        ax3.set_xticks(range(len(elems)),elems)

        ax4.legend(loc='best',fontsize=10)
        ax4.set_ylabel('Eigenvenctor {0}'.format(maxvec+1))
        ax4.set_xlim(-1,len(elems)+1)
        ax4.set_ylim(-1,1)
        ax4.set_title('Element weighted')
        ax4.set_xticks(range(len(elems)),elems)
    plt.ioff()
    if hide:
        plt.close('all')

