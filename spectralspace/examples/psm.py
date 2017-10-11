
import numpy as np

## Polynomial Model from Yuan-Sen Ting (Rix+ 2017) ##

datadir = '/geir_data/scr/price-jones/Code/synspec/data/'

psminfo = np.load('{0}/kurucz_quadratic_psm.npz'.format(datadir))
coeff_array = psminfo['coeff_array']

# a set of training labels
training_labels = psminfo['training_labels']
wavelength = psminfo['wavelength']

# auxiliary arrays to reconstruct the spectrum (because we need to choose a reference point to "Taylor-expand"
inds = psminfo['indices']
reference_flux = psminfo['reference_flux']
reference_point = psminfo['reference_point']
Teff,logg,vturb,ch,nh,oh,nah,mgh,alh,sih,sh,kh,cah,tih,vh,mnh,nih,feh,c12c13 = reference_point

#LABEL ORDER Teff [1000K], logg, vturb [km/s] (micro), ch, nh, oh, nah, mgh, alh, sih, sh, kh, cah, 
#tih, vh, mnh, nih, feh, log10(c12c13)

#==================================================
# make generate APOGEE spectrum
def generate_spectrum(labels=None,Teff=Teff,logg=logg,vturb=vturb,ch=ch,nh=nh,oh=oh,nah=nah,mgh=mgh,
                      alh=alh,sih=sih,sh=sh,kh=kh,cah=cah,tih=tih,vh=vh,mnh=mnh,nih=nih,feh=feh,
                      c12c13=c12c13,order=2):
    if not isinstance(labels,(list,np.ndarray)):
        labels = np.array([Teff,logg,vturb,ch,nh,oh,nah,mgh,alh,sih,sh,kh,cah,tih,vh,mnh,nih,
                           feh,c12c13])
    
    # make quadratic labels
    linear_terms = np.array(labels) - reference_point
    if order == 1.5:
        linind = 19
        t = linear_terms[0]
        g = linear_terms[1]
        f = linear_terms[17]
        fit_terms = np.array([t**2,t*g,t*f,g**2,g*f,f**2])
        lvec = np.hstack((linear_terms,fit_terms))
        coeffs = np.array([coeff_array[:,0+linind],coeff_array[:,1+linind],
                           coeff_array[:,17+linind],coeff_array[:,19+linind],
                           coeff_array[:,35+linind],
                           coeff_array[:,187+linind]])
        coeffs = np.concatenate((coeff_array.T[:len(linear_terms)],coeffs)).T
        spec_generate = np.dot(coeffs,lvec) + reference_flux
    if order == 2:
        quadratic_terms = np.einsum('i,j->ij',linear_terms,linear_terms)[inds[:,0],inds[:,1]]
        lvec = np.hstack((linear_terms, quadratic_terms))
        # generate spectrum                                                    
        spec_generate = np.dot(coeff_array,lvec) + reference_flux
    elif order == 1:
        lvec = np.hstack((linear_terms))
        # generate spectrum
        lin_coeff = coeff_array.T[:len(linear_terms)].T
        spec_generate = np.dot(lin_coeff,lvec) + reference_flux
    return spec_generate
