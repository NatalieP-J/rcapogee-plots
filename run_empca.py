"""

Usage:
run_empca [-hvgxups] [-m FNAME] [-d DELTR] [-n NVECS]

Options:
    -h, --help
    -v, --verbose
    -s, --silent
    -g, --generate                  Option to generate everything from scratch.
    -u, --usemad                    Option to use M.A.D. instead of variance.
    -p, --plot                      Option to generate plots
    -x, --hidefigs                  Option to hide figures.
    -m FNAME, --model FNAME         Provide a pickle file containing a Sample model (see residuals.py)
    -d DELTR, --delt DELTR          Provide an R2 difference at which to cutoff iteration [default: 0]
    -n NVECS, --nvecs NVECS         Specify number of empca vectors [default: 5]

"""

import docopt
import empca
reload(empca)
from empca import empca
import numpy as np
import matplotlib.pyplot as plt
from run_residuals import timeIt,weight_residuals
from residuals import doubleResidualHistPlot,elemwindows
import access_spectrum as acs
import os
import polyfit as pf

silent=False
verbose=True
elems = ['Al','Ca','C','Fe','K','Mg','Mn','Na','Ni','N','O','Si','S','Ti','V']
aspcappix = 7214
default_colors = {0:'b',1:'g',2:'r',3:'c'}
default_markers = {0:'o',1:'s',2:'v',3:'D'}
default_sizes = {0:10,1:8,2:10,3:8}

def pix_empca(model,residual,errs,empcaname,nvecs=5,gen=False,verbose=False,nstars=5,deltR2=0,usemad=True,randseed=1):
    """
    Runs EMPCA on a set of residuals with dimension of aspcappix.
    """
    if os.path.isfile(empcaname) and not gen:
        empcamodel,empcamodel_weight,basicweights = acs.pklread(empcaname)
    elif not os.path.isfile(empcaname) or gen:
        # Identify pixels at which there are more than nstars stars
        goodpix = ([i for i in range(aspcappix) if np.sum(residual[i].mask) < residual.shape[1]-nstars],)
        if verbose:
            print 'number of unusable pixels, ',len([i for i in range(aspcappix) if i not in goodpix[0]])
        # Create new array to feed to EMPCA with only good pixels
        empca_res = residual[goodpix].T
        # Create a set of weights for EMPCA, setting weight to zero if value is masked
        mask = (empca_res.mask==False)
        basicweights = mask.astype(float)
        empcamodel,runtime1 = timeIt(empca,empca_res.data,weights = basicweights,nvec=nvecs,deltR2=deltR2,mad=usemad,randseed=randseed)
        # Change weights to incorporate flux uncertainties
        sigmas = errs.T[goodpix].T
        weights=basicweights
        weights[mask] = 1./sigmas[mask]**2
        if verbose:
            print 'nans ',np.where(np.isnan(weights)==True)
        empcamodel_weight,runtime2 = timeIt(empca,empca_res.data,weights = weights,nvec=nvecs,deltR2=deltR2,mad=usemad,randseed=randseed)
        if verbose:
            print 'Pixel runtime (unweighted):\t', runtime1/60.,' min'
            print 'Pixel runtime (weighted):\t', runtime2/60.,' min'
        acs.pklwrite(empcaname,[empcamodel,empcamodel_weight,basicweights,weights])
    return empcamodel,empcamodel_weight,basicweights,weights

def resize_pix_eigvecs(residual,empcamodel,dim2=aspcappix,nstars=5,nvecs=5):
    good = ([i for i in range(dim2) if np.sum(residual[i].mask) < residual.shape[1]-nstars],)
    # Resize eigenvectors appropriately and mask missing elements
    #empcamodel.eigvec.resize((nvecs,aspcappix))
    empcamodel.neweigvec = np.ma.masked_array(np.zeros((nvecs,dim2)))
    for ind in range(len(elems)):
        for vec in range(nvecs):
            newvec = np.ma.masked_array(np.zeros((dim2)),mask = np.ones((dim2)))
            newvec[good] = empcamodel.eigvec[vec][:len(good[0])]
            newvec.mask[good] = 0
            empcamodel.neweigvec[vec] = newvec
    empcamodel.eigvec=empcamodel.neweigvec
    plt.show()

def elem_empca(model,residual,errs,empcaname,nvecs=5,gen=False,verbose=False,deltR2=0,usemad=True,nstars=5,randseed=1):
    if nvecs > len(elems):
        nvecs = len(elems) - 1
    if os.path.isfile(empcaname) and not gen:
        empcamodel,empcamodel_weight,basicweights,weights = acs.pklread(empcaname)
    elif not os.path.isfile(empcaname) or gen:
        goodelem = ([i for i in range(len(elems)) if np.sum(residual[i].mask) < residual.shape[1]-nstars])
        empca_res = residual[goodelem].T
        noise = errs[goodelem].T
        mask = (empca_res.mask==False)
        basicweights = mask.astype(float)
        empcamodel,runtime1 = timeIt(empca,empca_res,weights = basicweights,nvec=nvecs,deltR2=deltR2,mad=usemad,randseed=randseed)
        weights=basicweights
        weights[mask] = 1./noise[mask]**2
        if verbose:
            print 'nan ',np.where(np.isnan(weights)==True)
        empcamodel_weight,runtime2 = timeIt(empca,empca_res,weights = weights,nvec=nvecs,deltR2=deltR2,mad=usemad,randseed=randseed)
        if verbose:
            print 'Element runtime (unweighted):\t', runtime1/60.,' min'
            print 'Element runtime (weighted):\t', runtime2/60.,' min'
        acs.pklwrite(empcaname,[empcamodel,empcamodel_weight,basicweights,weights])
    return empcamodel,empcamodel_weight,basicweights,weights

def R2noise(weights,empcamodel,usemad=True):
    """
    Calculate the fraction of variance due to noise.
    """
    if usemad:
        var = empcamodel._unmasked_data_mad2*1.4826**2.
    elif not usemad:
        var = empcamodel._unmasked_data_var
    Vnoise = np.mean(1./(weights[weights!=0]))
    if verbose:
        print 'var, Vnoise ',var,Vnoise
    return 1-(Vnoise/var)

def R2(empcamodel,usemad=True):
    """
    For a given EMPCA model object, fill an array with R2 values for a set number of eigenvectors
    """
    vecs = len(empcamodel.eigvec)
    R2_arr = np.zeros(vecs+1)
    for vec in range(vecs+1):
        R2_arr[vec] = empcamodel.R2(vec,mad=usemad)
    return R2_arr

def weight_eigvec(model,nvecs,empcamodel):
    """
    Construct new eigenvectors weighted by element windows.
    """
    neweigvecs = np.zeros((nvecs,len(elems)))
    for ind in range(len(elems)):
        for vec in range(nvecs):
            neweigvecs[vec][ind] = model.weighting(empcamodel.eigvec[vec],elems[ind])
    return neweigvecs

def weight_residual(model,numstars,plot=True,subgroup=False):
    # Create output arrays
    weighted = np.ma.masked_array(np.zeros((len(elems),numstars)))
    weightedsigs = np.ma.masked_array(np.zeros((len(elems),numstars)))
    i=0
    # Cycle through elements
    for elem in elems:
        if subgroup != False:
            match = np.where(model.data[model.subgroup]==subgroup)
            residual = model.residual[subgroup]
            sigma = model.errs[match].T
        elif not subgroup:
            residual = model.residual
            sigma = model.errs.T
        # Weight residuals and sigma values
        weightedr = model.weighting_stars(residual,elem,
                                          model.outName('pkl','resids',elem=elem,
                                                        order = model.order,
                                                        subgroup=subgroup,
                                                        cross=model.cross))
        weighteds = np.sqrt(np.ma.sum(sigma**2*np.tile(pf.normweights(elemwindows[elem])**2,(sigma.shape[1],1)).T,axis=0))
        weighteds = np.ma.masked_array(weighteds,mask=weightedr.mask)
        savename = model.outName('pkl',content='sigma',elem=elem,order=model.order,subgroup=subgroup,cross=model.cross)
        acs.pklwrite(savename,weighteds)
        if plot:
            doubleResidualHistPlot(elem,weightedr[weightedr.mask==False],weighteds[weighteds.mask==False],
                                   model.outName('res','residhist',elem = elem,
                                                 order = model.order,
                                                 cross=model.cross,seed = model.seed,
                                                 subgroup = subgroup),
                                   bins = 50)
        weighted[i] = weightedr
        weightedsigs[i] = weighteds
        i+=1
    return weighted,weightedsigs

def plot_R2(empcamodels,weights,ptitle,savename,labels=None,nvecs=5,usemad=True,hide=True):
    R2noiseval = R2noise(weights,empcamodels[1],usemad=usemad)
    vec_vals = range(0,nvecs+1)
    plt.figure(figsize=(12,10))
    plt.xlim(0,nvecs)
    plt.ylim(0,1)
    plt.fill_between(vec_vals,R2noiseval,1,color='r',alpha=0.2)
    if R2noiseval > 0:
        plt.text(1,0.9,'R2_noise = {0:2f}'.format(R2noiseval))
    elif R2noiseval <= 0:
        plt.text(1,0.9,'R2_noise = {0:2f}'.format(R2noiseval))
    plt.axhline(R2noiseval,linestyle='--',color='k',label='Noise Threshold')
    plt.xlabel('Number of eigenvectors')
    plt.ylabel('Variance explained')
    plt.title(ptitle)
    for e in range(len(empcamodels)):
        R2_vals = R2(empcamodels[e],usemad=usemad)
        if not labels:
            plt.plot(vec_vals,R2_vals,marker='o',linewidth = 3,markersize=8)
        else:
            plt.plot(vec_vals,R2_vals,label=labels[e],marker='o',linewidth = 3,markersize=8)
    if not labels:
        plt.savefig(savename)
    else:
        plt.legend(loc='best')
        plt.savefig(savename)
    if hide:
        plt.close()

def norm_eigvec(eigvec):
    return eigvec/np.sqrt(np.sum(eigvec**2))

def plot_element_eigvec(eigvecs,savenames,mastercolors=default_colors,markers=default_markers,sizes=default_sizes,labels=None,hidefigs=False,nvecs=5):
    
    assert len(eigvecs) <= len(mastercolors)
    assert len(markers) == len(mastercolors)
    assert len(sizes) == len(mastercolors)

    for vec in range(nvecs):
        plt.figure(figsize=(12,10))
        plt.xticks(range(len(elems)),elems)
        plt.axhline(0,color='k')
        plt.xlim(-1,len(elems))
        plt.ylim(-1,1)
        plt.xlabel('Elements')
        plt.ylabel('Eigenvector')
        plt.title('{0} eigenvector, weighted by element'.format(vec))
        vectors = np.zeros((len(eigvecs),len(elems)))
        e = 0
        for eigvec in eigvecs: 
            vectors[e] = norm_eigvec(eigvec[vec])
            if not labels:
                plt.plot(vectors[e],color=mastercolors[e],marker=markers[e],linestyle='None',markersize=sizes[e])
            else:
                plt.plot(vectors[e],color=mastercolors[e],marker=markers[e],linestyle='None',markersize=sizes[e],label=labels[e])
            e+=1
        for i in range(len(elems)):
            colors = {}
            for v in range(len(vectors)):
                colors[vectors[v][i]] = mastercolors[v]
            order = colors.keys()
            order = sorted(order,key = abs)[::-1]
            for val in order:
                plt.plot([i,i],[0,val],color=colors[val],linewidth=3)
        for v in range(len(vectors)):
            plt.axhline(max(vectors[v],key=abs),color=mastercolors[v],linestyle='--',linewidth=3)
        if not labels:
            plt.savefig(savename)
            plt.close()
        else:
            plt.savefig(savenames[vec])
            plt.legend(loc='best')
            plt.close()



if __name__=='__main__':

    # Read in command line arguments
    arguments = docopt.docopt(__doc__)

    verbose = arguments['--verbose']
    silent = arguments['--silent']
    gen = arguments['--generate']
    usemad = arguments['--usemad']
    doplot = arguments['--plot']
    hide = arguments['--hidefigs']
    modelname = arguments['--model']
    deltR2 = float(arguments['--delt'])
    nvecs = int(arguments['--nvecs'])

    model=acs.pklread(modelname)

    nstars = 5

    if model.subgroups[0] != False:
        for subgroup in model.subgroups:
            
            if verbose:
                print subgroup
            
            match = np.where(model.data[model.subgroup]==subgroup)
            
            if verbose:
                print 'PIXEL SPACE'
            empcaname = model.outName('pkl',content = 'empca',subgroup=subgroup,order = model.order,seed = model.seed,cross=model.cross,nvecs=nvecs,mad=usemad)
            m1,m2,w1,w2 = pix_empca(model,model.residual[subgroup],model.errs[match],empcaname,nvecs=nvecs,gen=gen,verbose=verbose,nstars=nstars,deltR2=deltR2,usemad=usemad)
            
            R2noiseval1 = R2noise(w1,m1,usemad=usemad)
            R2noiseval2 = R2noise(w2,m2,usemad=usemad)

            R2vals1 = R2(m1,usemad=usemad)
            R2vals2 = R2(m2,usemad=usemad)
            
            try:
                nvecs_required1 = np.where(R2vals1 > R2noiseval1)[0][0]
                if nvecs_required1 > 0:
                    nvecs_required1 -= 1
            except IndexError:
                nvecs_required1 = 'more than '+str(nvecs)
            try:
                nvecs_required2 = np.where(R2vals2 > R2noiseval2)[0][0]
                if nvecs_required2 > 0:
                    nvecs_required2 -= 1
            except IndexError:
                nvecs_required2 = 'more than '+str(nvecs)

            if verbose:
                print 'ELEMENT SPACE'
            residual,errs = weight_residual(model,model.numstars[subgroup],plot=True,subgroup=subgroup)
            empcaname = model.outName('pkl',content = 'empca_element',order = model.order,seed = model.seed,cross=model.cross,subgroup=subgroup,nvecs=nvecs,mad=usemad)
            m3,m4,w3,w4 = elem_empca(model,residual,errs,empcaname,nvecs=nvecs,gen=gen,verbose=verbose,deltR2=deltR2,usemad=usemad)
            
            R2noiseval3 = R2noise(w3,m3,usemad=usemad)
            R2noiseval4 = R2noise(w4,m4,usemad=usemad)

            R2vals3 = R2(m3,usemad=usemad)
            R2vals4 = R2(m4,usemad=usemad)
            
            try:
                nvecs_required3 = np.where(R2vals3 > R2noiseval3)[0][0]
                if nvecs_required3 > 0:
                    nvecs_required3 -= 1
            except IndexError:
                nvecs_required3 = 'more than '+str(nvecs)
            try:
                nvecs_required4 = np.where(R2vals4 > R2noiseval4)[0][0]
                if nvecs_required4 > 0:
                    nvecs_required4 -= 1
            except IndexError:
                nvecs_required4 = 'more than '+str(nvecs)

            if verbose:
                print 'space \t weight type \t R2_noise \t number of vectors'
                print 'pixel \t basic weight \t ',R2noiseval1,' \t ',nvecs_required1
                print 'pixel \t error weight \t ',R2noiseval2,' \t ',nvecs_required2
                print 'elem \t basic weight \t ',R2noiseval3,' \t ',nvecs_required3
                print 'elem \t error weight \t ',R2noiseval4,' \t ',nvecs_required4

            if doplot:
                labels = ['Unweighted EMPCA - raw','Weighted EMPCA - raw']
                savename = model.outName('pca',content='pix_empcaR2',subgroup=subgroup,order=model.order,seed=model.seed,cross=model.cross,nvecs=nvecs,mad=usemad)
                ptitle = 'R2 for {0} from pixel space'.format(subgroup)
                plot_R2([m1,m2],w2,ptitle,savename,labels=None,nvecs=nvecs,usemad=usemad,hide=hide)
                labels = ['Unweighted EMPCA - proc','Weighted EMPCA - proc']
                savename = model.outName('pca',content='elem_empcaR2',subgroup=subgroup,order=model.order,seed=model.seed,cross=model.cross,nvecs=nvecs,mad=usemad)
                ptitle = 'R2 for {0} from element space'.format(subgroup)
                plot_R2([m3,m4],w4,ptitle,savename,labels=None,nvecs=nvecs,usemad=usemad,hide=hide)

            resize_pix_eigvecs(model.residual[subgroup],m1,nstars=nstars,nvecs=nvecs)
            resize_pix_eigvecs(model.residual[subgroup],m2,nstars=nstars,nvecs=nvecs)

            newm1 = weight_eigvec(model,nvecs,m1)
            newm2 = weight_eigvec(model,nvecs,m2)
            
            if doplot:
                savenames = []
                for vec in range(nvecs+1):
                    savenames.append(model.outName('pca',content='empca',subgroup=subgroup,order=model.order,seed=model.seed,cross=model.cross,nvecs=nvecs,eigvec=vec,mad=usemad))
                labels = ['Unweighted EMPCA - raw','Weighted EMPCA - raw','Unweighted EMPCA - proc','Weighted EMPCA - proc']
                eigvecs = [newm1,newm2,m3.eigvec,m4.eigvec]
                plot_element_eigvec(eigvecs,savenames,labels=labels,hidefigs=hide,nvecs=nvecs)

    elif model.subgroups[0] == False:
        
        if verbose:
            print 'PIXEL SPACE'
        empcaname = model.outName('pkl',content = 'empca',order = model.order,seed = model.seed,cross=model.cross,nvecs=nvecs,mad=usemad)
        m1,m2,w1,w2 = pix_empca(model,model.residual,model.errs,empcaname,nvecs=nvecs,gen=gen,verbose=verbose,nstars=nstars,deltR2=deltR2,usemad=usemad)
        
        R2noiseval1 = R2noise(w1,m1,usemad=usemad)
        R2noiseval2 = R2noise(w2,m2,usemad=usemad)

        R2vals1 = R2(m1,usemad=usemad)
        R2vals2 = R2(m2,usemad=usemad)
    
        try:
            nvecs_required1 = np.where(R2vals1 > R2noiseval1)[0][0]
            if nvecs_required1 > 0:
                nvecs_required1 -= 1
        except IndexError:
            nvecs_required1 = 'more than '+str(nvecs)
        try:
            nvecs_required2 = np.where(R2vals2 > R2noiseval2)[0][0]
            if nvecs_required2 > 0:
                nvecs_required2 -= 1
        except IndexError:
            nvecs_required2 = 'more than '+str(nvecs)

        if verbose:
            print 'ELEMENT SPACE'
        residual,errs = weight_residual(model,model.numstars,plot=True)
        empcaname = model.outName('pkl',content = 'empca_element',order = model.order,seed = model.seed,cross=model.cross,nvecs=nvecs,mad=usemad)
        m3,m4,w3,w4 = elem_empca(model,residual,errs,empcaname,nvecs=nvecs,gen=gen,verbose=verbose,deltR2=deltR2,usemad=usemad)
        
        R2noiseval3 = R2noise(w3,m3,usemad=usemad)
        R2noiseval4 = R2noise(w4,m4,usemad=usemad)

        R2vals3 = R2(m3,usemad=usemad)
        R2vals4 = R2(m4,usemad=usemad)
        
        try:
            nvecs_required3 = np.where(R2vals3 > R2noiseval3)[0][0]
            if nvecs_required3 > 0:
                nvecs_required3 -= 1
        except IndexError:
            nvecs_required3 = 'more than '+str(nvecs)
        try:
            nvecs_required4 = np.where(R2vals4 > R2noiseval4)[0][0]
            if nvecs_required4 > 0:
                nvecs_required4 -= 1
        except IndexError:
            nvecs_required4 = 'more than '+str(nvecs)
            
        if verbose:
            print 'space \t weight type \t R2_noise \t number of vectors'
            print 'pixel \t basic weight \t ',R2noiseval1,' \t ',nvecs_required1
            print 'pixel \t error weight \t ',R2noiseval2,' \t ',nvecs_required2
            print 'elem \t basic weight \t ',R2noiseval3,' \t ',nvecs_required3
            print 'elem \t error weight \t ',R2noiseval4,' \t ',nvecs_required4


        if doplot:
            labels = ['Unweighted EMPCA - raw','Weighted EMPCA - raw']
            savename = model.outName('pca',content='pix_empcaR2',order=model.order,seed=model.seed,cross=model.cross,nvecs=nvecs,mad=usemad)
            ptitle = 'R2 from pixel space'
            plot_R2([m1,m2],w1,ptitle,savename,labels=None,nvecs=nvecs,usemad=usemad,hide=hide)
            labels = ['Unweighted EMPCA - proc','Weighted EMPCA - proc']
            savename = model.outName('pca',content='elem_empcaR2',order=model.order,seed=model.seed,cross=model.cross,nvecs=nvecs,mad=usemad)
            ptitle = 'R2 from element space'
            plot_R2([m3,m4],w2,ptitle,savename,labels=None,nvecs=nvecs,usemad=usemad,hide=hide)

        resize_pix_eigvecs(model.residual,m1,nstars=nstars,nvecs=nvecs)
        resize_pix_eigvecs(model.residual,m2,nstars=nstars,nvecs=nvecs)

        newm1 = weight_eigvec(model,nvecs,m1)
        newm2 = weight_eigvec(model,nvecs,m2)
 
        if doplot:      
            savenames = []
            for vec in range(nvecs+1):
                savenames.append(model.outName('pca',content='empca',order=model.order,seed=model.seed,cross=model.cross,nvecs=nvecs,eigvec=vec+1,mad=usemad))
            labels = ['Unweighted EMPCA - raw','Weighted EMPCA - raw','Unweighted EMPCA - proc','Weighted EMPCA - proc']
            eigvecs = [newm1,newm2,m3.eigvec,m4.eigvec]
            plot_element_eigvec(eigvecs,savenames,labels=labels,hidefigs=hide,nvecs=nvecs)







