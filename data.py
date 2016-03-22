import apogee.tools.read as apread
from apogee.tools import bitmask
from read_clusterdata import read_caldata
import window as wn

# Dictionary to translate APOGEE's pixel mask (DR12).
# Keys correspond to set bits in the mask.

aspcappix = 7214

APOGEE_PIXMASK={0:"BADPIX", # Pixel marked as BAD in bad pixel mask
                1:"CRPIX", # Pixel marked as cosmic ray in ap3d
                2:"SATPIX", # Pixel marked as saturated in ap3d
                3:"UNFIXABLE", # Pixel marked as unfixable in ap3d
                4:"BADDARK", # Pixel marked as bad as determined from dark frame
                5:"BADFLAT", # Pixel marked as bad as determined from flat frame
                6:"BADERR", # Pixel set to have very high error (not used)
                7:"NOSKY", # No sky available for this pixel from sky fibers
                8:"LITTROW_GHOST", # Pixel falls in Littrow ghost, may be affected
                9:"PERSIST_HIGH", # Pixel falls in high persistence region, may be affected
                10:"PERSIST_MED", # Pixel falls in medium persistence region, may be affected
                11:"PERSIST_LOW", # Pixel falls in low persistence region, may be affected
                12:"SIG_SKYLINE", # Pixel falls near sky line that has significant flux compared with object
                13:"SIG_TELLURIC", # Pixel falls near telluric line that has significant absorption
                14:"NOT_ENOUGH_PSF", # Less than 50 percent PSF in good pixels
                15:"POORSNR", # Signal to noise below limit
                16:"FAILFIT" # Fitting for stellar parameters failed on pixel
                } 

# Chosen set of bits on which to mask
badcombpixmask= bitmask.badpixmask()+2**bitmask.apogee_pixmask_int("SIG_SKYLINE")

elems = ['Al','Ca','C','Fe','K','Mg','Mn','Na','Ni','N','O','Si','S','Ti','V']

# Functions to access particular sample types
readfn = {'clusters' : read_caldata,        # Sample of open and globular clusters
          'OCs': read_caldata,                # Sample of open clusters
          'GCs': read_caldata,                # Sample of globular clusters
          'red_clump' : apread.rcsample        # Sample of red clump stars
          }

independentVariables = {'clusters':['TEFF'],
                        'OCs':['TEFF'],
                        'GCs':['TEFF'],
                        'red_clump':['TEFF','LOGG','FE_H']
                    }
elemwindows = {}
for elem in elems:
    w = wn.read(elem,dr=12,apStarWavegrid=False)
    elemwindows[elem] = w

detec_blue =[0,2920]
detec_blue_wv = [1.514,1.581] #microns
detec_green = [2920,5320]
detec_green_wv = [1.585,1.644] #microns
detec_red = [5320,7213]
detec_red_wv = [1.647,1.696] #microns
detectors = [0,2920,5320,7214]
ASPCAPdetectors = [0,2920,2920,5320,5320,7214]
apStarDetectors = [322,3242,3648,6048,6412,8306]


#def pix2wavelength():
    # log scale log wv[i+1] - log wv[i] = 6e-6
    
