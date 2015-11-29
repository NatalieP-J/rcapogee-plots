###############################################################################
# read_clusterdata.py: module to read APOGEE data on globular clusters
###############################################################################
import sys
import numpy
import apogee.tools.read as apread
from apogee.tools import bitmask
try:
    from apogee.spec import continuum
except RuntimeError:
    print 'Failed to load continuum'
import astropy.io.ascii
_COMBINED_INDEX=1
_GCS= ['M15','M92','M53','N5466','M13','M2','M3','M5','M107','M71']
_ERASESTR= "                                                                               "
def read_meszarosgcdata(filename='../clusterdata/aj509073t2_mrt.txt'):
    """
    NAME:
       read_meszarosgcdata
    PURPOSE:
       Read the data on globular clusters from Meszaros et al. (2015)
    INPUT:
       filename= Name of the file that has the ApJ machine-readable table
    OUTPUT:
       data structure with the data
    HISTORY:
       2015-02-11 - Started - Bovy (IAS@KITP)
       2015-08-13 - Re-written for new data format - Bovy (UofT)
    """
    data= astropy.io.ascii.read(filename)
    data.rename_column('Clust','CLUSTER')
    data.rename_column('Teff','TEFF')
    data.rename_column('log(g)','LOGG')
    data.rename_column('[Fe/H]','FEH')
    data.rename_column('2MASS','ID')
    # Now match to allStar to get the location_ids and H magnitudes
    alldata= apread.allStar(raw=True)
    locids= numpy.zeros(len(data),dtype='int')-1
    hmags= numpy.zeros(len(data),dtype='float')-1
    # and match to allVisit for the fibers that each star was observed in
    allvdata= apread.allVisit(raw=True)
    fibers= numpy.zeros((len(data),numpy.nanmax(alldata['NVISITS'])),
                        dtype='int')-1
    for ii in range(len(data)):
        if 'Pleiades' in data['CLUSTER'][ii]: continue
        indx= alldata['APOGEE_ID'] == data['ID'][ii]
        if numpy.sum(indx) == 0:
            raise ValueError('allStar match for %s not found ...' % (data['ID'][ii]))
        if len(list(set(alldata['LOCATION_ID'][indx]))) > 1:
            raise ValueError('Multiple matches found for for %s ...' % (data['ID'][ii]))
        locids[ii]= alldata['LOCATION_ID'][indx][0]
        hmags[ii]= alldata['H'][indx][0]
        for jj in range(alldata['NVISITS'][indx][0]):
            fibers[ii,jj]= allvdata[alldata['VISIT_PK'][indx][0,jj]]['FIBERID']
    data['LOCATION_ID']= locids
    data['H']= hmags
    data['FIBERID']= fibers
    return data

def read_caldata(filename='../clusterdata/aj485195t4_mrt.txt'):
    """
    NAME:
       read_caldata
    PURPOSE:
       Read the data on calibration clusters from Meszaros et al. (2013)
    INPUT:
       filename= Name of the file that has the ApJ machine-readable table
    OUTPUT:
       data structure with the data
    HISTORY:
       2015-02-11 - Written - Bovy (IAS@KITP)
    """
    data= astropy.io.ascii.read(filename)
    data.rename_column('Cluster','CLUSTER')
    data.remove_column('Teff')
    data.rename_column('TeffC','TEFF')
    data.remove_column('logg')
    data.rename_column('loggC','LOGG')
    data.remove_column('[M/H]')
    data.rename_column('[M/H]C','FEH')
    data.rename_column('2MASS','ID')
    # Now match to allStar to get the location_ids
    alldata= apread.allStar(raw=True)
    locids= numpy.zeros(len(data),dtype='int')-1
    hmags= numpy.zeros(len(data),dtype='float')-1
    # and match to allVisit for the fibers that each star was observed in
    allvdata= apread.allVisit(raw=True)
    fibers= numpy.zeros((len(data),numpy.nanmax(alldata['NVISITS'])),
                        dtype='int')-1
    for ii in range(len(data)):
        if 'Pleiades' in data['CLUSTER'][ii]: continue
        indx= alldata['APOGEE_ID'] == data['ID'][ii]
        if numpy.sum(indx) == 0:
            raise ValueError('allStar match for %s not found ...' % (data['ID'][ii]))
        if len(list(set(alldata['LOCATION_ID'][indx]))) > 1:
            raise ValueError('Multiple matches found for for %s ...' % (data['ID'][ii]))
        locids[ii]= alldata['LOCATION_ID'][indx][0]
        hmags[ii]= alldata['H'][indx][0]
        for jj in range(alldata['NVISITS'][indx][0]):
            fibers[ii,jj]= allvdata[alldata['VISIT_PK'][indx][0,jj]]['FIBERID']
    data['LOCATION_ID']= locids
    data['H']= hmags
    data['FIBERID']= fibers
    return data

def read_spectra(cluster,teffmin=4000.,teffmax=5000.,cont_type='cannon',
                 cont_deg=4):
    """
    NAME:
       read_spectra
    PURPOSE:
       Read the APOGEE spectra and their errors for stars in a given cluster
    INPUT:
       cluster - Name of the cluster (name in one of the data files)
       teffmin= (4000.) minimum temperature
       teffmax= (5000.) maximum temperature
       cont_type = ('cannon') type of continuum normalization to perform
       cont_deg= (4) degree polynomial to fit for continuum normalization
    OUTPUT:
       (data, spec, specerr) - (full data structure, spectra [nspec,nlam], spectral uncertainties [nspec,nlam]) nlam=7214 on ASPCAP grid
    HISTORY:
       2015-08-13 - Written based on some older code - Bovy (UofT)
    """
    if cluster.upper() in _GCS:
        data= read_meszarosgcdata()
    else:
        data= read_caldata()
    # Cut to just this cluster and temperature range
    if 'rc' in cluster.lower():
        # Only for NGC 6819
        rc= True
        cluster= cluster[:-2]
    else:
        rc= False
    data= data[data['CLUSTER'] == cluster.upper()]
    data= data[(data['TEFF'] < teffmax)\
                   *(data['TEFF'] > teffmin)]
    if cluster.lower() == 'n6819':
        g4CN= good4CN(cluster,data)
        g4CN[10]= False # another one, by hand!
        if rc:
            data= data[True-g4CN] # Just those!
        else:
            data= data[g4CN] # Just those!
    # Load all the spectra
    nspec= len(data)
    spec= numpy.zeros((nspec,7214))
    specerr= numpy.zeros((nspec,7214))
    # Setup bad pixel mask
    badcombpixmask= bitmask.badpixmask()\
        +2**bitmask.apogee_pixmask_int("SIG_SKYLINE")
    for ii in range(nspec):
        sys.stdout.write('\r'+"Loading spectrum %i / %i ...\r" % (ii+1,nspec))
        sys.stdout.flush()
        spec[ii]= apread.apStar(data['LOCATION_ID'][ii],
                                data['ID'][ii],
                                ext=1,header=False,
                                aspcapWavegrid=True)[_COMBINED_INDEX]
        specerr[ii]= apread.apStar(data['LOCATION_ID'][ii],
                                   data['ID'][ii],
                                   ext=2,header=False,
                                   aspcapWavegrid=True)[_COMBINED_INDEX]
        # Inflate uncertainties for bad pixels
        mask= apread.apStar(data['LOCATION_ID'][ii],
                            data['ID'][ii],
                            ext=3,header=False,
                            aspcapWavegrid=True)[_COMBINED_INDEX]
        specerr[ii,(mask & (badcombpixmask)) != 0]+=\
            100.*numpy.mean(spec[ii,True-numpy.isnan(spec[ii])])
        # Also inflate pixels with high SNR to 0.5%
        highsnr= spec[ii]/specerr[ii] > 200.
        specerr[ii,highsnr]= 0.005*numpy.fabs(spec[ii,highsnr])
        # Continuum-normalize
        cont= continuum.fit(spec[ii],specerr[ii],type=cont_type,deg=cont_deg)
        spec[ii]/= cont
        specerr[ii]/= cont
        specerr[ii,highsnr]= 0.005 # like standard APOGEE reduction
    sys.stdout.write('\r'+_ERASESTR+'\r')
    sys.stdout.flush()
    return (data,spec,specerr)

def good4CN(cluster,data):
    """
    NAME:
       good4CN
    PURPOSE:
       return the indices of stars that can be used to determine the spread in C/N
    INPUT:
       cluster - the cluster name
       data - the data for this cluster
    OUTPUT:
       index
    HISTORY:
       2015-09-04 - Written - Bovy (UofT)
    """
    if cluster.lower() == 'm67':
        indx= (data['LOGG'] > _m67rccut(data['TEFF']))\
            *(numpy.fabs(data['TEFF']-4600.)>3.)
    elif cluster.lower() == 'n6819':
        apokasc= apread.apokasc()
        ma= numpy.zeros(len(data),dtype='int')-1
        for ii in range(len(data)):
            try:
                ma[ii]= (numpy.arange(len(apokasc))[apokasc['APOGEE_ID'] == data['ID'][ii]])[0]
            except IndexError: pass
        indx= numpy.ones(len(data),dtype='bool')
        for ii in range(len(data)):
            if ma[ii] >= 0 \
                    and apokasc[ma[ii]]['SEISMO EVOL'] == 'CLUMP          ':
                indx[ii]= False
        # Also the clump stars' friends, they are all suspicious
        indx[numpy.fabs(data['TEFF']-4765.) < 60.]= False
    else:
        indx= numpy.ones(len(data),dtype='bool')
    return indx

def _m67rccut(t):
    return (3.4-1.7)/(4950.-4365)*(t-4950)+3.35
