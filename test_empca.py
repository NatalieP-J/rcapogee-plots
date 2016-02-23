"""

Usage:
comp_empca [-hvx] [-i ITER] [-s SEED] [-e ELEMS] [-m MAXVEC] [-n NVECS]

Options:
    -h, --help
    -v, --verbose
    -x, --hidefigs                      Option to hide figures
    -i ITER, --iter ITER                Number of iterations to perform [default:1]
    -s SEED, --seed SEED                Random seed initializer [default:1]
    -e ELEMS, --elems ELEMS             List of elements to construct false residuals [default:C]
    -n NVECS, --nvecs NVECS             Number of eigenvectors to use
    -m MAXVEC, --maxvec MAXVEC          Maximum number of vectors to display
"""

import matplotlib.pyplot as plt
import numpy as np
import access_spectrum as acs
import polyfit as pf
import os

windowinfo = 'pickles/windowinfo.pkl'
elemwindows,window_all,window_peak,windowPeaks,windowPixels,tophats = acs.pklread(windowinfo)

def make_specs(specs,errs,elemlist,proportion=None):
    SNR = specs/errs
    vec = np.zeros(aspcappix)
    for ind in range(len(elemlist)):
        if not proportion:
            vec += elemwindows[elemlist[ind]]
        elif proportion:
            vec += elemwindows[elem[ind]]*proportion[ind]
    newspecs = np.ma.masked_array(np.tile(vec,(specs.shape[0],1)),specs.mask)
    noise = newspecs/SNR
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
    
def test_run(specs,noise,deltR2=2e-3,nvecs=5,mad=True,maxvec=5):
    m1,m2,w1,w2 = pix_empca(None,specs.T,noise,'test.pkl',nvecs=nvecs,deltR2=2e-3,gen=True,usemad=mad)
    R2_1 = R2(m1) #must be here (and not below resize) to avoid error
    R2_2 = R2(m2)
    R2_noise2 = R2noise(w2,m2,usemad=mad)
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
    m3,m4,w3,w4 = elem_empca(None,specs_weight,noise_weight,'test2.pkl',nvecs=nvecs,gen=True,deltR2=2e-3,usemad=mad)        
    R2_3 = R2(m3)
    R2_4 = R2(m4)
    R2_noise4 = R2noise(w4,m4,usemad=mad)
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
    plt.legend(loc='best',fontsize=10)
    
def test_run_comp(specs,noise,deltR2=2e-3,nvecs=5,mad=True,maxvec=5):
    m1,m2,w1,w2 = pix_empca(None,specs.T,noise,'test.pkl',nvecs=nvecs,deltR2=2e-3,gen=True,usemad=mad)
    R2_1 = R2(m1) #must be here (and not below resize) to avoid error
    R2_2 = R2(m2)
    R2_noise2 = R2noise(w2,m2,usemad=mad)
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
    m3,m4,w3,w4 = elem_empca(None,specs_weight,noise_weight,'test2.pkl',nvecs=nvecs,gen=True,deltR2=2e-3,usemad=mad)        
    R2_3 = R2(m3)
    R2_4 = R2(m4)
    R2_noise4 = R2noise(w4,m4,usemad=mad)
    resize_pix_eigvecs(specs_weight,m3,nstars=5,dim2=len(elems),nvecs=nvecs)
    resize_pix_eigvecs(specs_weight,m4,nstars=5,dim2=len(elems),nvecs=nvecs)
    for n in range(maxvec):
        plt.figure(1,figsize=(12,3))
        plt.axhline(0,linestyle='--',color='k',linewidth=3)
        plt.plot(norm_eigvec(m1elem[n]),'o',markersize=8)
        plt.title('Pixel unweighted')
        plt.xticks(range(len(elems)),elems)
        plt.legend(loc='best',fontsize=10)
        plt.ylabel('Eigenvenctor {0}'.format(n+1))
        plt.xlim(-1,len(elems)+1)
        plt.ylim(-1,1)
        plt.figure(2,figsize=(12,3))
        plt.axhline(0,linestyle='--',color='k',linewidth=3)
        plt.plot(norm_eigvec(m2elem[n]),'o',markersize=8)
        plt.title('Pixel weighted')
        plt.xticks(range(len(elems)),elems)
        plt.legend(loc='best',fontsize=10)
        plt.ylabel('Eigenvenctor {0}'.format(n+1))
        plt.xlim(-1,len(elems)+1)
        plt.ylim(-1,1)
        plt.figure(3,figsize=(12,3))
        plt.axhline(0,linestyle='--',color='k',linewidth=3)
        plt.plot(norm_eigvec(m3.eigvec[n]),'o',markersize=8)
        plt.title('Element unweighted')
        plt.xticks(range(len(elems)),elems)
        plt.legend(loc='best',fontsize=10)
        plt.ylabel('Eigenvenctor {0}'.format(n+1))
        plt.xlim(-1,len(elems)+1)
        plt.ylim(-1,1)
        plt.figure(4,figsize=(12,3))
        plt.axhline(0,linestyle='--',color='k',linewidth=3)
        plt.plot(norm_eigvec(m4.eigvec[n]),'o',markersize=8)
        plt.title('Element weighted')
        plt.xticks(range(len(elems)),elems)
        plt.legend(loc='best',fontsize=10)
        plt.ylabel('Eigenvenctor {0}'.format(n+1))
        plt.xlim(-1,len(elems)+1)
        plt.ylim(-1,1)
    plt.figure(5,figsize=(12,3))
    plt.title('Pixel unweighted')
    plt.plot(R2_1,marker='o',linewidth = 3,markersize=8)
    plt.axhline(R2_noise2,linestyle='--',color='b',linewidth=3,label='R2n_pix = {0:2f}'.format(R2_noise2))
    plt.fill_between(range(nvecs+1),R2_noise2,1,color='b',alpha=0.2)
    plt.ylabel('R2')
    plt.xlabel('Number of EMPCA vectors')
    plt.legend(loc='best',fontsize=10)
    plt.figure(6,figsize=(12,3))
    plt.title('Pixel weighted')
    plt.plot(R2_2,marker='o',linewidth = 3,markersize=8)
    plt.axhline(R2_noise2,linestyle='--',color='b',linewidth=3,label='R2n_pix = {0:2f}'.format(R2_noise2))
    plt.fill_between(range(nvecs+1),R2_noise2,1,color='b',alpha=0.2)
    plt.ylabel('R2')
    plt.xlabel('Number of EMPCA vectors')
    plt.legend(loc='best',fontsize=10)
    plt.figure(7,figsize=(12,3))
    plt.title('Element unweighted')
    plt.plot(R2_3,marker='o',linewidth = 3,markersize=8,label='Element unweighted')
    plt.axhline(R2_noise4,linestyle='--',color='r',linewidth=3,label='R2n_elem = {0:2f}'.format(R2_noise4))
    plt.fill_between(range(nvecs+1),R2_noise4,1,color='r',alpha=0.2)
    plt.ylabel('R2')
    plt.xlabel('Number of EMPCA vectors')
    plt.legend(loc='best',fontsize=10)
    plt.figure(8,figsize=(12,3))
    plt.title('Element weighted')
    plt.plot(R2_4,marker='o',linewidth = 3,markersize=8,label='Element weighted')
    plt.axhline(R2_noise4,linestyle='--',color='r',linewidth=3,label='R2n_elem = {0:2f}'.format(R2_noise4))
    plt.fill_between(range(nvecs+1),R2_noise4,1,color='r',alpha=0.2)
    plt.ylabel('R2')
    plt.xlabel('Number of EMPCA vectors')
    plt.legend(loc='best',fontsize=10)
    

if __name__=='__main__':

    arguments = docopt.docopt(__doc__)

    verbose = arguments['--verbose']
    hide = arguments['--hidefigs']
    iters = arguments['--iter']
    seed = arguments['--seed']
    nvecs = arguments['--nvecs']
    maxvec = arguments['--maxvec']
    elemlist = arguments['--elems']
    elemlist = elemlist.split(',')

    specs = acs.pklread('red_clump/pickles/spectra_FE_H_u-0.4_d-0.5.pkl')[0]
    errs = acs.pklread('red_clump/pickles/errs_FE_H_u-0.4_d-0.5.pkl')

    if iters==1:
        falsespecs,noise = make_specs(specs,errs,elemlist)
        if all(falsespecs.mask==True):
            print 'All masked'
        elif not all(falsespecs.mask==True):
            test_run(falsespecs,noise,maxvec=maxvec,nvecs=nvecs)

    elif iters != 1:
        for i in range(iters):
            falsespecs,noise = make_specs(specs,errs,elemlist)
        if all(falsespecs.mask==True):
            print 'All masked'
        elif not all(falsespecs.mask==True):
            test_run_comp(falsespecs,noise,maxvec=maxvec,nvecs=nvecs)

