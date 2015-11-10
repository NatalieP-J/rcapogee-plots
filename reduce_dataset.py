import numpy as np

def slice_data(data,sliceinfo):
    """
    Selects indices within a data set that belong to a certain range in a given parameter.
    
    data:       APOGEE data set (ex: data = apogee.tools.read.rcsample())
    sliceinfo   list containing instructions for selecting from data
                i.e. sliceinfo = [parameter to select for, lower limit on parameter, 
                                 upper limit on parameter]
    """
    label, lower, upper = sliceinfo
    indx = (data[label]>lower) & (data[label]<upper)
    return indx

def pixmask_find(maskdata,pix,bitval = None):
    """
    Checks if any spectra have a bit set at pix.
    
    maskdata:      list of 'spectra' of values indicating the fitness of each pixel
                   (ex: maskdata = get_spectra_ap(data,ext = 3,indx = 1))
    pix:           pixel at which to check fitness
    
    Returns a list of indices indicating spectra which are good at pix, and a list
    of indices indicating spectra which are bad at pix, both in the form of 
    numpy.where output
    """
    if bitval == None:
        indx = np.where(np.array(maskdata[:,pix]) == 0)
        bindx = np.where(np.array(maskdata[:,pix]) != 0)
    elif bitval != None:
        indx = np.where(np.array(maskdata[:,pix]) != bitval)
        bindx = np.where(np.array(maskdata[:,pix]) == bitval)  
    return indx,bindx