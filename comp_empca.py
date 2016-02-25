"""

Usage:
comp_empca [-hvx] [-s SEARCHSTR]

Options:
    -h, --help
    -v, --verbose
    -x, --hidefigs                      Option to hide figures
    -s SEARCHSTR, --search SEARCHSTR    String to search for files to compare [default: None]
"""

import os
import docopt
import access_spectrum as acs
from run_empca import R2,R2noise
import numpy as np
import matplotlib.pyplot as plt

def gen_colours(n,cmap = 'Spectral'):
    return plt.get_cmap(cmap)(np.linspace(0, 1.0, n))

def read_files(searchstr):
    searchstr = '*'.join(searchstr)
    os.system('ls {0} > filelist.txt'.format(searchstr))
    filelist = np.loadtxt('filelist.txt',dtype=str)
    models = []
    for f in filelist:
        models.append(acs.pklread(f))
    return models,filelist


if __name__=='__main__':

    # Read in command line arguments
    arguments = docopt.docopt(__doc__)

    verbose = arguments['--verbose']
    hide = arguments['--hidefigs']
    search = arguments['--search']
    search = search.split(',')

    models,filelist = read_files(search)

    pixinds = [i for i in range(len(filelist)) if 'elem' not in filelist[i]]
    eleminds = [i for i in range(len(filelist)) if 'elem' in filelist[i]]

    pcolours = gen_colours(len(pixinds)*2)

    plt.figure(1,figsize=(16,9))

    cind = 0
    sind = 0
    for p in pixinds:
        print 'file = ',filelist[p]
        label = ''
        mad = False
        if 'Correct' in filelist[p]:
            label += 'SNR corrected'
        if 'MADTrue' in filelist[p]:
            label += ' M.A.D.'
            mad = True
        nvecs = len(models[p][0].eigvec)
        vec_vals = range(0,nvecs+1)
        R2vals1 = R2(models[p][0],usemad=mad)
        R2vals2 = R2(models[p][1],usemad=mad)
        R2n = R2noise(models[p][-1],models[p][1],usemad=mad)
        plt.figure(1)
        plt.subplot2grid((2,4),(0,sind))
        plt.ylim(0,1)
        plt.xlim(0,nvecs)
        plt.ylabel('R2')
        plt.xlabel('Number of EMPCA vectors')
        plt.axhline(R2n,linestyle='--',color = 'k')
        plt.fill_between(vec_vals,R2n,1,color=pcolours[cind],alpha=0.1)
        plt.plot(vec_vals,R2vals1,marker='o',linewidth = 3,markersize=8,label=label+' unweighted',color = pcolours[cind])
        plt.legend(loc='best',fontsize=10,title='R2_noise = {0:2f}'.format(R2n))
        plt.subplot2grid((2,4),(1,sind))
        plt.ylim(0,1)
        plt.xlim(0,nvecs)
        plt.ylabel('R2')
        plt.xlabel('Number of EMPCA vectors')
        plt.axhline(R2n,linestyle='--',color = 'k')
        plt.fill_between(vec_vals,R2n,1,color=pcolours[cind+1],alpha=0.1)
        plt.plot(vec_vals,R2vals2,marker='o',linewidth = 3,markersize=8,label=label+' weighted',color = pcolours[cind+1])
        plt.legend(loc='best',fontsize=10,title='R2_noise = {0:2f}'.format(R2n))
        plt.suptitle('Pixel Space')
        sind+=1
        cind+=2

    cind = 0
    sind = 0

    pcolours = gen_colours(len(eleminds)*2)

    plt.figure(2,figsize=(16,9))

    for p in eleminds:
        print 'file = ',filelist[p]
        label = ''
        mad = False
        if 'Correct' in filelist[p]:
            label += 'SNR corrected'
        if 'MADTrue' in filelist[p]:
            label += ' M.A.D.'
            mad = True
        nvecs = len(models[p][0].eigvec)
        vec_vals = range(0,nvecs+1)
        R2vals1 = R2(models[p][0],usemad=mad)
        R2vals2 = R2(models[p][1],usemad=mad)
        R2n = R2noise(models[p][-1],models[p][1],usemad=mad)
        plt.figure(2)
        plt.subplot2grid((2,4),(0,sind))
        plt.ylim(0,1)
        plt.xlim(0,nvecs)
        plt.ylabel('R2')
        plt.xlabel('Number of EMPCA vectors')
        plt.axhline(R2n,linestyle='--',color = 'k')
        plt.fill_between(vec_vals,R2n,1,color=pcolours[cind],alpha=0.1)
        plt.plot(vec_vals,R2vals1,marker='o',linewidth = 3,markersize=8,label=label+' unweighted',color = pcolours[cind])
        plt.legend(loc='best',fontsize=10,title='R2_noise = {0:2f}'.format(R2n))
        plt.subplot2grid((2,4),(1,sind))
        plt.ylim(0,1)
        plt.xlim(0,nvecs)
        plt.ylabel('R2')
        plt.xlabel('Number of EMPCA vectors')
        plt.axhline(R2n,linestyle='--',color = 'k')
        plt.fill_between(vec_vals,R2n,1,color=pcolours[cind+1],alpha=0.1)
        plt.plot(vec_vals,R2vals2,marker='o',linewidth = 3,markersize=8,label=label+' weighted',color = pcolours[cind+1])
        plt.legend(loc='best',fontsize=10,title='R2_noise = {0:2f}'.format(R2n))
        plt.suptitle('Element Space')
        sind+=1
        cind+=2
    plt.show()

