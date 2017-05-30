# calc_elemvar_table.py: computes a table that shows how much the spectrum varies in each sub-window when varying elements
import os, os.path
import sys
import pickle
import numpy
import apogee.spec.window as apwindow
_MYWINDOWS= True
_NO_INDIV_WINDOWS= True
def calc_elemvar_table(savename,_mywindows=False,_no_indiv_windows=False):
    if _mywindows: import window
    # Restore the file with the synthetic spectra with all variations
    if os.path.exists(savename):
        with open(savename,'rb') as savefile:
            baseline= pickle.load(savefile)
            elem_synspec= pickle.load(savefile)
    else:
        raise IOError("File %s with synthetic spectra does not exists; compute this first with apogee/test/test_windows.py" % savename)
    # Compute the whole table
    elems= ['C','N','O','Na','Mg','Al','Si','S','K','Ca','Ti','V','Mn','Fe',
            'Ni']
    table= []
    # Loop through synthetic spectra varying one element
    for elem in elems:
        col= []
        # Now look at the windows for each element
        for elemc in elems:
            if not _no_indiv_windows:
                si, ei= apwindow.waveregions(elemc,asIndex=True,pad=0,dr='13')
            if _mywindows:
                elemWeights= window.read(elemc,dr='13')
            else:
                elemWeights= apwindow.read(elemc,dr='13')
            elemWeights/= numpy.nansum(elemWeights)
            # Start with total
            col.append(numpy.sqrt(numpy.nansum((elemWeights\
                                         *(elem_synspec[elem][1]-elem_synspec[elem][0])**2.))/numpy.nansum(elemWeights)))
            if not _no_indiv_windows:
                for s,e in zip(si,ei):
                    col.append(numpy.sqrt(numpy.nansum((elemWeights\
                                                 *(elem_synspec[elem][1]-elem_synspec[elem][0])**2.)[s:e])/numpy.nansum(elemWeights[s:e])))
        table.append(col)
    return table

def print_elemvar_table(table,outname):
    # Write the table
    elems= ['C','N','O','Na','Mg','Al','Si','S','K','Ca','Ti','V','Mn','Fe',
            'Ni']
    with open(outname,'w') as outfile:
        cnt= 0
        for elemc in elems:
            # First write total
            tline= '%s ' % elemc
            ref= table[elems.index(elemc)][cnt]
            for ii,elem in enumerate(elems):
                if elem == elemc:
                    tline+= '& %.3f' % (table[ii][cnt])
                else:
                    tline+= '& %i' % (int(round(100.*table[ii][cnt]/ref)))
            outfile.write(tline+'\\\\\n')
            cnt+= 1
            if _NO_INDIV_WINDOWS: continue
            # Now do each individual window
            for jj in range(apwindow.num(elemc,pad=0)):
                tline= '%s%i ' % (elemc,jj+1)
                ref= table[elems.index(elemc)][cnt]
                for ii,elem in enumerate(elems):
                    if elem == elemc:
                        tline+= '& %.3f' % (table[ii][cnt])
                    else:
                        tline+= '& %i' % (100.*table[ii][cnt]/ref)
                outfile.write(tline+'\\\\\n')
                cnt+= 1
    return None

if __name__ == '__main__':
    table= calc_elemvar_table(sys.argv[1],_mywindows=_MYWINDOWS,
                              _no_indiv_windows=_NO_INDIV_WINDOWS)
    print_elemvar_table(table,sys.argv[2])
