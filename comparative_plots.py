import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from empca_residuals import *
import os, glob
from scipy.optimize import leastsq
from matplotlib.ticker import MultipleLocator,AutoMinorLocator
from ncells_calculation import calculate_Ncells,consth
import access_spectrum as acs

font = {'family': 'serif',
        'weight': 'normal',
        'size'  :  18 
}

matplotlib.rc('font',**font)

default_cmap = 'plasma'
datadir = '/geir_data/scr/price-jones/Data/apogee_dim_reduction/'
figdir = '/home/price-jones/Documents/rc_dim_paper'

def factors(n):
    """
    Return factors of n.

    Found on stackoverflow: https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
    
    """
    return np.array(list(set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))))


def contrastR2_methods(direcs,models,labels,colours=None,titles=[], 
                       savename=None,figsize=(15,6),subsamples=False,seeds=[]):
    """
    Create a plot to compare R^2 values for a given list of models.
    
    direcs:        List of strings that name directories where model files 
                   are stored.
    models:        List of strings that name model files to access
    labels:        Labels for each model
    colours:       List of colours for each model
    titles:        List of strings as titles for each directory
    savename:      String file name to save the plot
    figsize:       Tuple that deterimines the size of a figure
    subsamples:    Toggle to the number of subsamples in the jackknife analysis
                   to show errorbars on the number of eigenvectors
    seeds:         A list with a seed for each directory that indicates which
                   random jackknife result to use.
    
    """
    # Initialize figure
    plt.figure(1,figsize=figsize)
    # If colours not given, use default colourmap
    if not isinstance(colours,(list,np.ndarray)):
        colours = plt.get_cmap('plasma')(np.linspace(0,0.85,len(models)))
    # Find point outline colours
    edgecolours = plt.get_cmap('Greys')(np.linspace(0.2,0.85,len(models)))
    
    # DIRECTORIES
    for d in range(len(direcs)):
        # Create subplot for each directory
        ax=plt.subplot(1,len(direcs),d+1)
        plt.ylim(0,1)
        plt.xlabel('number of components',fontsize=18)
        # If we're on the first directory, add y-axis information
        if d==0:
            plt.ylabel(r'$R^2$',fontsize=25)
            plt.yticks(fontsize=20)
            yminorlocator = MultipleLocator(0.05)
            ax.yaxis.set_minor_locator(yminorlocator)
        # If not on first directory, just add y-ticks
        if d!=0:
            emptys = ['']*5
            plt.yticks(np.arange(0,1.2,0.2),emptys)
            yminorlocator = MultipleLocator(0.05)
            ax.yaxis.set_minor_locator(yminorlocator)
        # Set colour index to zero
        c = 0
        
        # MODELS
        for m in range(len(models)):
            # Read model from file
            model = acs.pklread('{0}/{1}'.format(direcs[d],models[m]))
            # Constrain x-axis from size of R^2
            plt.xlim(-1,len(model.R2Array))
            # Indexes whether appropriate jackknife files are found
            found = True
            # If using jackknife technique, read from file to determine 
            # R^2-R^2_noise intersection
            if subsamples and found:
                func = models[m].split('_')
                func = func[-1].split('.')[0]
                # If seed not specified, use results from all random seeds
                if seeds == []:
                    matchfiles = glob.glob('{0}/subsamples{1}*{2}*numeigvec.npy'.format(direcs[d],subsamples,func))
                # If seed specified, only find results that match seed
                if seeds !=[]:
                    matchfiles = glob.glob('{0}/subsamples{1}*{2}*seed{3}*numeigvec.npy'.format(direcs[d],subsamples,func,seeds[d]))
                # If no files, flag to derive the intersection later
                if matchfiles == []:
                    found=False
                elif matchfiles != []:
                    # If multiple seeds found, take the average of results
                    avgs = np.zeros(len(matchfiles))
                    sigs = np.zeros(len(matchfiles))
                    for f in range(len(matchfiles)):
                        avgs[f],sigs[f] = np.fromfile(matchfiles[f])
                    avg = np.mean(avgs)
                    sig = np.mean(sigs)
                    # Put vertical lines at location of average R^2-R^2_noise 
                    # intersection
                    if avg != -1 and abs(len(model.R2Array)-1-avg)>1:
                        plt.axvline(avg,0,1,color=colours[c],lw=3)
                        plt.axvline(avg-sig,0,color=colours[c],lw=1.5)
                        plt.axvline(avg+sig,0,color=colours[c],lw=1.5)
                        plt.fill_between(np.array([avg-sig,avg+sig]),0,1,
                                         alpha=0.08,color=colours[c])
                        
            # If not using jackknife technique, derive R^2-R^2_noise 
            # intersection
            if not found or not subsamples:
                crossvec = np.where(model.R2Array > model.R2noise)
                if crossvec[0] != []:
                    crossvec = crossvec[0][0] - 1
                    if crossvec < 0:
                        crossvec = 0
                    # Mark intersection
                    plt.axvline(crossvec,0,model.R2Array[crossvec],
                                color=colours[c],lw=3)
                    # Label intersection
                    if crossvec != 0:
                        plt.text(crossvec+0.03*len(model.R2Array),0.02,
                                 '{0}'.format(crossvec),color=colours[c],
                                 weight='bold',fontsize=15)
                    elif crossvec == 0:
                        plt.text(crossvec-0.06*len(model.R2Array),0.02,
                                 '{0}'.format(crossvec),color=colours[c],
                                 weight='bold',fontsize=15)
            # If you're on the first model, add title
            if m==0:
                if titles != []:
                    plt.text(-1+0.05*(len(model.R2Array)),0.95,titles[d],
                             fontsize=15,va='top',backgroundcolor='w')
            
            # If you're on the last subplot, plot R2 with labels
            if d==len(direcs)-1:
                plt.plot(model.R2Array,'-',color=colours[c],lw=4,
                         label=labels[m])
                # Add dots to the curve but not at every point if there's too 
                # many principal components
                if len(model.R2Array)<=10:
                    plt.plot(model.R2Array,'o',color=colours[c],markersize=11,
                             markeredgecolor=edgecolours[c],
                             markeredgewidth=1.5)
                if len(model.R2Array)>10:
                    xvals = np.arange(0,len(model.R2Array))
                    plt.plot(xvals[0::len(model.R2Array)/7],
                             model.R2Array[0::len(model.R2Array)/7],'o',
                             color=colours[c],markersize=11,
                             markeredgecolor=edgecolours[c],
                             markeredgewidth=1.5)
                # If this is the last model, make lable for R^2_noise
                if m==len(models)-1:
                    plt.axhline(np.NaN,np.NaN,color='k',ls='--',lw=3,
                                label=r'$R^2_{\mathrm{noise}}$')
            # If you're on any other subplot, plot R2 without labels
            elif d!=len(direcs)-1:
                plt.plot(model.R2Array,'-',color=colours[c],lw=4)
                # Add dots to the curve but not at every point if there's too  
                # many principal components 
                if len(model.R2Array)<=10:
                    plt.plot(model.R2Array,'o',color=colours[c],markersize=11,
                             markeredgecolor=edgecolours[c],
                             markeredgewidth=1.5)
                if len(model.R2Array)>10:
                    xvals = np.arange(0,len(model.R2Array))
                    plt.plot(xvals[0::len(model.R2Array)/7],
                             model.R2Array[0::len(model.R2Array)/7],'o',
                             color=colours[c],markersize=11,
                             markeredgecolor=edgecolours[c],
                             markeredgewidth=1.5)
            # Plot R^2_noise
            plt.axhline(model.R2noise,color=colours[c],ls='--',lw=3)
            # Move to the next colour
            c+=1

        # Reduce the number of xticks if there are many eigenvectors
        if len(model.R2Array) > 10:
            steps = np.linspace(0,len(model.R2Array),5,dtype=int)[:-1]
            plt.xticks(steps,fontsize=20)
            stepsize = (steps[1]-steps[0])/2.
            xminorlocator = MultipleLocator(stepsize)
            ax.xaxis.set_minor_locator(xminorlocator)
        elif len(model.R2Array) < 10:
            plt.xticks(np.arange(0,len(model.R2Array),1),fontsize=20)
            xminorlocator = MultipleLocator(0.5)
            ax.xaxis.set_minor_locator(xminorlocator)

        # Tweak tick parameters
        ax.yaxis.set_tick_params(width=2,which='major',size=7)
        ax.yaxis.set_tick_params(width=2,which='minor',size=4)
        ax.xaxis.set_tick_params(width=2,which='major',size=7)
        ax.xaxis.set_tick_params(width=2,which='minor',size=4)

        # Add legend
        if d==len(direcs)-1:
            legend = plt.legend(loc='best',fontsize=20)
            legend.get_frame().set_linewidth(0.0)

    # Remove space between subplots
    plt.subplots_adjust(wspace=0.05)
    # Save the plot
    if savename:
        plt.savefig('{0}/{1}'.format(figdir,savename))

def sample_compare_nvec(direcs,models,labels,colours=None,savename=None,figsize=(15,6),subsamples=5,seeds = [],rotation=30,ha='right',bottom_margin=0.25):
    """
    Create a plot to compare R^2-R^2_noise intersection across models
 
    direcs:        List of strings that name directories where model files     
                   are stored.                                                 
    models:        List of strings that name model files to access             
    labels:        Labels for each model                  
    colours:       List of colours for each model
    savename:      String file name to save the plot                           
    figsize:       Tuple that deterimines the size of a figure                 
    subsamples:    Toggle to the number of subsamples in the jackknife analysis
                   to show errorbars on the number of eigenvectors             
    seeds:         A list with a seed for each directory that indicates which  
                   random jackknife result to use.
    rotation:      Angle of rotation for plot x-labels
    ha:            Alignment position for plot x-labels
    bottom_margin: Position of bottom margin for plot

    Return intersection points for each model.
   
    """
    # Initialize figure
    plt.figure(1,figsize=figsize)
    # Create subplot
    ax = plt.subplot(111)
    # Choose colours
    if not isinstance(colours,(list,np.ndarray)):
        colours = plt.get_cmap('inferno')(np.linspace(0,0.8,len(models)*len(direcs)))
    # Create arrays to store intersection points for each model
    points = np.zeros((len(direcs)*len(models)))
    errorbars = np.zeros((len(direcs)*len(models)))
    # Set model index to zero
    k = 0
    
    # DIRECTORIES
    for d in range(len(direcs)):
        
        # MODELS
        for m in range(len(models)):
            # Read model from file
            model = acs.pklread('{0}/{1}'.format(direcs[d],models[m]))
            # Indexes whether appropriate jackknife files are found
            found = True
            # If using jackknife technique, read from file to determine
            # R^2-R^2_noise intersection
            if subsamples:
                func = models[m].split('_')
                func = func[-1].split('.')[0]
                # If seed not specified, use results from all random seeds
                if seeds == []:
                    matchfiles = glob.glob('{0}/subsamples{1}*{2}*numeigvec.npy'.format(direcs[d],subsamples,func))
                # If seed specified, only find results that match seed
                if seeds !=[]:
                    matchfiles = glob.glob('{0}/subsamples{1}*{2}*seed{3}*numeigvec.npy'.format(direcs[d],subsamples,func,seeds[d]))
                # If no files, flag with disallowed value for intersection
                if matchfiles == []:
                    points[k] = -1
                    errorbars[k] = 0
                elif matchfiles != []:
                    # If multiple seeds found, take the average of results
                    avgs = np.zeros(len(matchfiles))
                    sigs = np.zeros(len(matchfiles))
                    for f in range(len(matchfiles)):
                        avgs[f],sigs[f] = np.fromfile(matchfiles[f])
                    avg = np.mean(avgs)
                    sig = np.mean(sigs)
                    if abs(avg-(len(model.R2Array)-1)) < 1:
                        points[k] = -1
                        errorbars[k] = 0
                    else:
                        points[k] = avg
                        errorbars[k] = sig
                    k+=1
    # Set equally separated positions for model labels
    xvals = np.arange(0,len(points))*2
    for i in range(len(points)):
        # Plot all valid points with shaded errorbars
        if points[i] != -1:
            plt.errorbar(xvals[i],points[i],yerr=errorbars[i],fmt='o',elinewidth=3.5,
                         ecolor=colours[i],color=colours[i],markersize=11,capthick=6,
                         markeredgewidth=2,markeredgecolor='k')
            plt.fill_between(np.arange(-2,np.max(xvals)+3),points[i]-errorbars[i],
                             points[i]+errorbars[i],color=colours[i],alpha=0.1)
            # Set transparency as a function of colour (lighter colours less transparent)
            alph = 0.7*((i+1.)/len(points))+0.2
            plt.axhline(points[i]-errorbars[i],color=colours[i],alpha=alph,lw=1)
            plt.axhline(points[i]+errorbars[i],color=colours[i],alpha=alph,lw=1)
        # Plot invalid points a lower bounds
        if points[i] == -1:
            plt.plot(xvals[i],len(model.R2Array)-1,'o',color=colours[i],markersize=8)
            plt.arrow(xvals[i],len(model.R2Array)-1,0,1,head_length=0.4,head_width=0.05,
                      color=colours[i])
    # Add model labels
    plt.xticks(xvals,labels,rotation=rotation,ha=ha,fontsize=20)
    
    # Tweak axes
    yminorlocator = MultipleLocator(2.5)
    ax.yaxis.set_minor_locator(yminorlocator)
    ax.yaxis.set_tick_params(width=2,which='major',size=7)
    ax.yaxis.set_tick_params(width=2,which='minor',size=4)
    ax.xaxis.set_tick_params(width=2,which='major',size=7)
    plt.ylabel('number of components')
    plt.xlim(-2,np.max(xvals)+2)
    # Adjust subplot position
    plt.margins(0.2)
    plt.subplots_adjust(bottom=bottom_margin,top=0.93,left=0.18,right=0.96)
    if savename:
        plt.savefig('{0}/{1}'.format(figdir,savename))
    return points


def Ncells_model(p,n,ncent=10):
    """
    Model the number of cells as a function of the number of principal
    components:

    p:     Holds model parameter a and b
    n:     Array of the number of principal components
    ncent: Central value for the number of principal components 
    
    Returns array with model values.
    """
    a,b = p
    return (10**a)*(b**(n-10))

def Ncells_res(p,n,Ncells_true,ncent=10):
    """
    Calculate the residuals between the model for the number of cells and 
    the number of cells.

    p:              Holds model parameter a and b                                   
    n:              Array of the number of principal components                     
    ncent:          Central value for the number of principal components
    Ncells_true:    Actual number of cells

    Returns array of residuals.
    """
    return Ncells_true-Ncells_model(p,n,ncent)

def contrast_Ncells(direcs,models,labels,colours,titles=[],savename=None,
                    figsize=(15,6),subsamples=False,seeds=[],generate=False,
                    denom=consth,ybounds=(1,1e15),givecvc=False,
                    makemodel=True,ncent=10,**kwargs):
    """
    Create a plot to compare R^2 values for a given list of models.
   
    direcs:        List of strings that name directories where model files     
                   are stored.                                                 
    models:        List of strings that name model files to access             
    labels:        Labels for each model                                       
    colours:       List of colours for each model                              
    titles:        List of strings as titles for each directory                
    savename:      String file name to save the plot                           
    figsize:       Tuple that deterimines the size of a figure                 
    subsamples:    Toggle to the number of subsamples in the jackknife analysis
                   to show errorbars on the number of eigenvectors
    seeds:         A list with a seed for each directory that indicates which  
                   random jackknife result to use.
    generate:      Keyword for calculate_Ncells that requires Ncells object to
                   be constructed from scratch if True
    denom:         Keyword for calculate_Ncells that specifies cell size
    ybounds:       Range of y-axis
    givecvc:       Allows user to specify R^2-R^2_noise intersection
    makemodel:     Toggle to make a model for Ncells
    ncent:         Central value for the number of prinicipal components if modeling
    **kwargs:      Keywords for chemical cell size function
    
    Returns model parameters if model requested.

    """
    # Intialize figure
    plt.figure(len(direcs),figsize=figsize)
    # If colours not given, use default colourmap                              
    if not isinstance(colours,(list,np.ndarray)):
        colours = plt.get_cmap('plasma')(np.linspace(0,0.85,len(models)))
    # Find point outline colours
    edgecolours = plt.get_cmap('Greys')(np.linspace(0.2,0.85,len(models)))

    ymin,ymax=np.floor(np.log10(ybounds[0])),np.ceil(np.log10(ybounds[1]))

    if makemodel:
        # Prepare for Ncells model
        fitparams = np.zeros((len(direcs)*len(models),2))
        # Set fit parameter index to zero
        a = 0

    # DIRECTORIES
    for d in range(len(direcs)):
        # Create subplot for each directory
        ax=plt.subplot(1,len(direcs),d+1)
        plt.ylim(ybounds[0],ybounds[1])
        plt.xlabel('number of components',fontsize=18)
        # Set colour index to zero
        c = 0
        
        # MODELS
        for m in range(len(models)):
            # Read model from file
            model = acs.pklread('{0}/{1}'.format(direcs[d],models[m]))
            # Constrain x-axis from size of R^2
            plt.xlim(-1,len(model.R2Array))
            # Indexes whether apprpriate jackknife files are found
            found = True
            # If using jackknife technique, read from file to determine 
            # R^2-R^2_noise intersection
            if subsamples and found:
                func = models[m].split('_')
                func = func[-1].split('.')[0]
                # If seed not specified, use results from all random seeds
                if seeds == []:
                    matchfiles = glob.glob('{0}/subsamples{1}*{2}*numeigvec.npy'.format(direcs[d],subsamples,func))
                # If seed specified, only find results that match seed
                if seeds !=[]:
                    matchfiles = glob.glob('{0}/subsamples{1}*{2}*seed{3}*numeigvec.npy'.format(direcs[d],subsamples,func,seeds[d]))
                # If not files, flad to derive the intersection later
                if matchfiles == []:
                    found = False
                elif matchfiles != []:
                    # If multiple seed found, take the average of results
                    avgs = np.zeros(len(matchfiles))
                    sigs = np.zeros(len(matchfiles))
                    for f in range(len(matchfiles)):
                        avgs[f],sigs[f] = np.fromfile(matchfiles[f])
                    avg = np.mean(avgs)
                    sig = np.mean(sigs)
                    # Put vertical lines at location of average R^2-R^2_noise
                    # intersection
                    if avg != -1 and abs(len(model.R2Array)-1-avg)>1:
                        plt.axvline(avg,color=colours[c],lw=3)
                        plt.axvline(avg-sig,color=colours[c],lw=1.5)
                        plt.axvline(avg+sig,color=colours[c],lw=1.5)
                        plt.fill_between(np.array([avg-sig,avg+sig]),10**ymin,
                                         10**ymax,alpha=0.08,color=colours[c])
                
            # If not using jackknife technique, derive R^2-R^2_noise           
            # intersection 
            elif not subsamples or not found:
                # Find the point of intersection between R^2 and R^2_noise
                crossvec = np.where(model.R2Array > model.R2noise)
                if crossvec[0] != []:
                    crossvec = crossvec[0][0] - 1
                    if crossvec < 0:
                        crossvec = 0
                    # Mark intersection
                    plt.axvline(crossvec,0,Ncells[crossvec],color=colours[c],
                                lw=3)
            # If intersection not found and no intersection specified
            if not givecvc or 'avg' not in locals():        
                Ncells = calculate_Ncells(direcs[d],model,models[m],
                                          denom=denom,generate=generate,
                                          **kwargs)

            # If intersection specified, pick top, middle or bottom of possible
            # range from jackknife, or use number given
            elif givecvc == 'max':
                Ncells = calculate_Ncells(direcs[d],model,models[m],
                                          denom=denom,generate=generate,
                                          cvc=avg+sig)
            elif givecvc == 'mid':
                Ncells = calculate_Ncells(direcs[d],model,models[m],
                                          denom=denom,generate=generate,
                                          cvc=avg)
            elif givecvc == 'min':
                Ncells = calculate_Ncells(direcs[d],model,models[m],
                                          denom=denom,generate=generate,
                                          cvc=avg-sig)
            elif givecvc == 'cvc':
                Ncells = calculate_Ncells(direcs[d],model,models[m],
                                          denom=denom,generate=generate,
                                          cvc=crossvec)
            elif isinstance(givecvc,(int)):
                Ncells = calculate_Ncells(direcs[d],model,models[m],
                                          denom=denom,generate=generate,
                                          cvc=givecvc)
        
            # Create independent values for model
            xvals = np.arange(0,len(model.R2Array)-1)+1
            # Store measured Ncells at those values
            plotcells = Ncells(xvals)
            
            if makemodel:
                # Model Ncells with least-squares
                p0 = [9,7]
                pnew = leastsq(Ncells_res,p0,args=(xvals,plotcells,ncent))
                fitparams[a] = pnew[0]
                a+=1
            
            # If you're on the last subplot, plot Ncells with labels
            if d==len(direcs)-1:
                plt.semilogy(xvals,plotcells,color=colours[c],lw=4,
                             label=labels[m])
                # Add dots to the curve but not at every point if there's too  
                # many principal components 
                if len(model.R2Array)<=10:
                    plt.semilogy(xvals,plotcells,'o',color=colours[c],
                                 markersize=11,markeredgecolor=edgecolours[c],
                                 markeredgewidth=1.5)
                if len(model.R2Array) > 10:
                    plt.semilogy(xvals[0::len(model.R2Array)/7],
                                 plotcells[0::len(model.R2Array)/7],'o',
                                 color=colours[c],markersize=11,
                                 markeredgecolor=edgecolours[c],
                                 markeredgewidth=1.5)
                # Show model
                if makemodel:
                    plt.semilogy(xvals,Ncells_model(pnew[0],xvals,10),
                                 ls = '--',lw=3,color='r')

            # If you're on any other subplot, plot Ncells without labels
            elif d!=len(direcs)-1:
                plt.semilogy(xvals,plotcells,color=colours[c],lw=4)
                # Add dots to the curve but not at every point if there's too  
                # many principal components 
                if len(model.R2Array)<=10:
                    plt.semilogy(xvals,plotcells,'o',color=colours[c],
                                 markersize=11,markeredgecolor=edgecolours[c],
                                 markeredgewidth=1.5)
                if len(model.R2Array) > 10:
                    plt.semilogy(xvals[0::len(model.R2Array)/7],
                                 plotcells[0::len(model.R2Array)/7],'o',
                                 color=colours[c],markersize=11,
                                 markeredgecolor=edgecolours[c],
                                 markeredgewidth=1.5)
                # Add model
                if makemodel:
                    plt.semilogy(xvals,Ncells_model(pnew[0],xvals,ncent),
                                 ls = '--',lw=3,color='r')
            
            # If on the first model, add title
            if m==0:
                if titles != []:
                    partway = 0.95*(ymax-ymin)
                    plt.text(-1+0.05*(len(model.R2Array)),10**partway,
                             titles[d],fontsize=15,va='top',
                             backgroundcolor='w')
     
            # Move to the next colour
            c+=1
        # Add major and minor ticks
        plt.tick_params(which='both', width=2)
        if d==0:
            plt.ylabel(r'$N_{\mathrm{cells}}$',fontsize=25)
            facts = factors(np.ceil(ymax-ymin))
            numlabels = np.max(facts[facts<=10])
            ticklocs = np.arange(ymin,ymax,(ymax-ymin)/numlabels)
            ticklabels = np.array([str(np.round(i))[:-2] for i in ticklocs])
            ticklabels = np.array(['{'+i+'}' for i in ticklabels])
            ticklabels = np.array(['$10^{0}$'.format(i) for i in ticklabels])
            plt.yticks(10**ticklocs,ticklabels.astype('str'),fontsize=25)
        # Have to erase tick labels after plotting in log scale case
        if d!=0:
            emptys = ['']*(ymax-ymin)
            plt.yticks(10**np.arange(ymin,ymax+1),emptys)
        # Reduce the number of xticks if there are many principal components
        if len(model.R2Array) > 10:
            steps = np.linspace(0,len(model.R2Array),5,dtype=int)[:-1]
            plt.xticks(steps,fontsize=20)
            stepsize = (steps[1]-steps[0])/2.
            xminorlocator = MultipleLocator(stepsize)
            ax.xaxis.set_minor_locator(xminorlocator)
        elif len(model.R2Array) < 10:
            plt.xticks(np.arange(0,len(model.R2Array),1),fontsize=20)
            xminorlocator = MultipleLocator(0.5)
            ax.xaxis.set_minor_locator(xminorlocator)

        # If you're on the last subplot, add the R2 legend
        if d==len(direcs)-1:
            if labels[0] != '':
                legend = plt.legend(loc='best',fontsize=15,
                                    title='$N_{\mathrm{cells}}$ calculation')
                legend.get_title().set_fontsize('16')
                legend.get_frame().set_linewidth(0.0)

        # Tweak tick parameters
        ax.yaxis.set_tick_params(width=2,which='major',size=7)
        ax.yaxis.set_tick_params(width=2,which='minor',size=4)
        ax.xaxis.set_tick_params(width=2,which='major',size=7)
        ax.xaxis.set_tick_params(width=2,which='minor',size=4)
    # Remove space between subplots
    plt.subplots_adjust(wspace=0.05)
    # Save the plot
    if savename:
        plt.savefig('{0}/{1}'.format(figdir,savename))
    if makemodel:
        return fitparams


def sample_compare_ncells(direcs,models,labels,colours=None,savename=None,
                          figsize=(15,6),subsamples=5,seeds=[],denom=consth,
                          rotation=30,ha='right',bottom_margin=0.25,**kwargs):
    """
    Create a plot to compare the number of cells at R^2-R^2_noise intersection                                                                                    direcs:        List of strings that name directories where model files                        are stored.                                                             
    models:        List of strings that name model files to access                         
    labels:        Labels for each model                                                   
    colours:       List of colours for each model                                          
    savename:      String file name to save the plot                                       
    figsize:       Tuple that deterimines the size of a figure                             
    subsamples:    Toggle to the number of subsamples in the jackknife analysis            
                   to show errorbars on the number of eigenvectors                         
    seeds:         A list with a seed for each directory that indicates which              
                   random jackknife result to use.
    denom:         Keyword for calculate_Ncells that specifies cell size
    rotation:      Angle of rotation for plot x-labels                                     
    ha:            Alignment position for plot x-labels                                    
    bottom_margin: Position of bottom margin for plot                                      
    **kwargs:      Keywords for chemical cell size function                      
          
    Returns Ncells at the intersection points for each model. 

    """
    # Intialize figure
    plt.figure(1,figsize=figsize)
    # Create subplot
    ax = plt.subplot(111)
    ax.set_yscale("log", nonposx='clip')
    # Choose colours if needed
    if not isinstance(colours,(list,np.ndarray)):
        colours = plt.get_cmap('inferno')(np.linspace(0,0.8,len(models)*len(direcs)))
    # Create arrays to store Ncells for each model
    points = np.ones((len(direcs)*len(models)))
    min_errorbars = np.zeros((len(direcs)*len(models)))
    max_errorbars = np.zeros((len(direcs)*len(models)))
    # Set model index to zero
    k = 0

    # DIRECTORIES
    for d in range(len(direcs)):
        
        # MODELS
        for m in range(len(models)):
            # Read model from file
            model = acs.pklread('{0}/{1}'.format(direcs[d],models[m]))
            # Get Ncells
            Ncells = calculate_Ncells(direcs[d],model,models[m],denom=denom,**kwargs)
            
            #Indexes whether appropriate jackknife files are found
            found = True
            # If using jackknife technique, read from file to determine
            # R^2-R^2_noise intersection
            if subsamples:
                func = models[m].split('_')
                func = func[-1].split('.')[0]
                # If seed not specified, use results from all random seeds
                if seeds != []:
                    matchfiles = glob.glob('{0}/subsamples{1}*{2}*seed{3}*numeigvec.npy'.format(direcs[d],subsamples,func,seeds[d]))
                # If seed specified, only find results that match seed
                if seeds == []:
                    matchfiles = glob.glob('{0}/subsamples{1}*{2}*numeigvec.npy'.format(direcs[d],subsamples,func))
                # If no files, derive intersection
                if matchfiles == []:
                    vec = np.interp(model.R2noise,model.R2Array,np.arange(len(model.R2Array)),left=0,right=-1)
                    points[k] = Ncells(vec)
                    min_errorbars[k] = 0
                    max_errorbars[k] = 0
                    k+=1
                elif matchfiles != []:
                    # If multiple seeds found, take the average of results
                    avgs = np.zeros(len(matchfiles))
                    sigs = np.zeros(len(matchfiles))
                    for f in range(len(matchfiles)):
                        avgs[f],sigs[f] = np.fromfile(matchfiles[f])
                    avg = np.mean(avgs)
                    sig = np.mean(sigs)
                    if avg !=-1:
                        points[k] = Ncells(avg)
                        min_errorbars[k] = points[k]-Ncells(avg-sig)
                        max_errorbars[k] = Ncells(avg+sig)-points[k]
                    elif avg==-1:
                        points[k] = -1
                    k+=1
    # Set equally separated positions for model labels
    xvals = np.arange(0,len(points))*2
    for i in range(len(points)):
        # Plot all valid points with shaded errorbars
        if points[i] != -1:
            plt.errorbar(xvals[i],points[i],
                         yerr=np.array([[min_errorbars[i],max_errorbars[i]]]).T,fmt='o',
                         elinewidth=3.5,ecolor=colours[i],color=colours[i],markersize=11,
                         capthick=6,markeredgewidth=2,markeredgecolor='k')
            plt.fill_between(np.arange(-2,np.max(xvals)+3),points[i]-min_errorbars[i],
                             points[i]+max_errorbars[i],color=colours[i],alpha=0.1)
            # Set transparency as a function of colour (lighter colours less transparent) 
            alph = 0.7*((i+1.)/len(points))+0.2
            plt.axhline(points[i]-min_errorbars[i],color=colours[i],alpha=alph,lw=1)
            plt.axhline(points[i]+max_errorbars[i],color=colours[i],alpha=alph,lw=1)
        
        # Plot invalid points as lower bounds
        if points[i] == -1:
            plt.plot(xvals[i],Ncells(len(model.R2Array)-1),'o',color=colours[i],
                     markersize=8)
            plt.arrow(xvals[i],Ncells(len(model.R2Array)-1),0,1,head_length=0.4,
                      head_width=0.05,color=colours[i])

    # Add model labels
    plt.xticks(xvals,labels,rotation=rotation,ha=ha)
    
    # Tweak axes
    plt.ylabel('number of cells')
    plt.xlim(-2,np.max(xvals)+2)
    plt.ylim(10**np.floor(np.log10(np.min(points-min_errorbars))-1),
             10**np.ceil(np.log10(np.max(points+max_errorbars))+1))
    minexp = np.floor(np.log10(np.min(points-min_errorbars))-1)
    maxexp = np.ceil(np.log10(np.max(points+max_errorbars))+1)
    a = np.arange(minexp,maxexp)
    skip = a[0::len(a)/7]
    yticklabels = []
    for i in range(len(skip)):
        string = '{0}'.format(int(skip[i]))
        string = '$10^{'+string+'}$'
        yticklabels.append(string)
    plt.yticks(10**skip,yticklabels,fontsize=21)
    ax.yaxis.set_tick_params(width=2,which='major',size=7)
    ax.yaxis.set_tick_params(width=2,which='minor',size=4)
    ax.xaxis.set_tick_params(width=2,which='major',size=7)
    # Adjust subplot position
    plt.margins(0.2)
    plt.subplots_adjust(bottom=bottom_margin,top=0.93,left=0.18,right=0.96)
    if savename:
        plt.savefig('{0}/{1}'.format(figdir,savename))
    return points
