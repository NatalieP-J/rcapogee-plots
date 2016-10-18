###############################################################################
# window.py: Code to modify ASPCAP windows
###############################################################################
import numpy
import apogee.spec.window as apwindow
from apogee.tools import _ELEM_SYMBOL as elems
from apogee.tools import toAspcapGrid
import calc_elemvar_table
def parse_elemvar():
    table= calc_elemvar_table.calc_elemvar_table('elemVariations_DR12_M67.sav')
    # Parse the table for easier use
    out1= {}
    out2= {}
    cnt= 0
    for elemc in elems:
        ref= {}
        others= {}
        cnt+= 1
        for jj in range(apwindow.num(elemc,pad=0)):
            ref[jj]= table[elems.index(elemc)][cnt]
            tothers= []
            for ii,elem in enumerate(elems):
                if elem == elemc:
                    continue
                else:
                    tothers.append(100.*table[ii][cnt]/ref[jj])
            others[jj]= tothers
            cnt+= 1
        out1[elemc]= ref
        out2[elemc]= others
    return (out1,out2)
ref, oth= parse_elemvar()

def bad(elem):
    """
    NAME:
       bad
    PURPOSE:
       return the index (on the apStar grid) of bad windows
    INPUT:
       elem - element
    OUTPUT:
       boolean array
    HISTORY:
       2015-09-02 - Written - Bovy (UofT)
    """
    si,ei= apwindow.waveregions(elem,asIndex=True,pad=0,dr='13')
    out= numpy.zeros(8575,dtype='bool')
    if elem.lower() == 'k': return out
    for ii,(s,e) in enumerate(zip(si,ei)):
        # First do the special cases
        if elem.lower() == 'c':
            if ref[elem.lower()][ii] <= 0.01 \
                    or numpy.any(numpy.array(oth[elem.lower()][ii]) > 100.):
                out[s:e]= True
        elif elem.lower() == 'n':
            if ref[elem.lower()][ii] <= 0.01 \
                    or numpy.any(numpy.array(oth[elem.lower()][ii]) > 200.):
                out[s:e]= True
        elif elem.lower() == 'o':
            if ref[elem.lower()][ii] <= 0.005 \
                    or numpy.any(numpy.array(oth[elem.lower()][ii]) > 500.):
                out[s:e]= True
        elif elem.lower() == 'na':
            if numpy.any(numpy.array(oth[elem.lower()][ii]) > 34.):
                out[s:e]= True
        elif elem.lower() == 'ti':
            if numpy.any(numpy.array(oth[elem.lower()][ii]) > 30.):
                out[s:e]= True
        elif elem.lower() == 'v':
            if numpy.any(numpy.array(oth[elem.lower()][ii]) > 40.):
                out[s:e]= True
        elif elem.lower() == 'mn':
            if numpy.any(numpy.array(oth[elem.lower()][ii]) > 25.):
                out[s:e]= True
        elif elem.lower() == 'ni':
            if numpy.any(numpy.array(oth[elem.lower()][ii]) > 50.):
                out[s:e]= True
        elif ref[elem.lower()][ii] <= 0.01 \
                or numpy.any(numpy.array(oth[elem.lower()][ii]) > 34.):
            out[s:e]= True
    return out

def read(elem,apStarWavegrid=True,dr=None):
    """
    NAME:
       read
    PURPOSE:
       read the window weights for a given element, modified to only return 'good' windows
    INPUT:
       elem - element
       apStarWavegrid= (True) if True, output the window onto the apStar wavelength grid, otherwise just give the ASPCAP version (blue+green+red directly concatenated)
       dr= read the window corresponding to this data release       
    OUTPUT:
       Array with window weights
    HISTORY:
       2015-01-25 - Written - Bovy (IAS)
       2015-09-02 - Modified for only returning 'good' windows - Bovy (UofT)
    """
    out= apwindow.read(elem,apStarWavegrid=True,dr=dr)
    out[bad(elem)]= 0.
    if not apStarWavegrid:
        return toAspcapGrid(out)
    else:
        return out

def tophat(elem,dr=None,apStarWavegrid=True):
    """
    NAME:
       tophat
    PURPOSE:
       return an array with True in the window of a given element and False otherwise, only for 'good' windows
    INPUT:
       elem - element     
       dr= read the window corresponding to this data release       
       apStarWavegrid= (True) if True, output the window onto the apStar wavelength grid, otherwise just give the ASPCAP version (blue+green+red directly concatenated)
    OUTPUT:
       array on apStar grid
    HISTORY:
       2015-01-26 - Written - Bovy (IAS@KITP)
       2015-09-02 - Modified for only returning 'good' windows - Bovy (UofT)
    """
    out= apwindow.tophat(elem,apStarWavegrid=True,dr=dr)
    out[bad(elem)]= 0.
    if not apStarWavegrid:
        return toAspcapGrid(out)
    else:
        return out
