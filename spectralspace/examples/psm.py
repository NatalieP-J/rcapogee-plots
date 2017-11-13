
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
def generate_spectrum(labels=None,Teff=Teff,logg=logg,vturb=vturb,ch=ch,nh=nh,
                      oh=oh,nah=nah,mgh=mgh,alh=alh,sih=sih,sh=sh,kh=kh,
                      cah=cah,tih=tih,vh=vh,mnh=mnh,nih=nih,feh=feh,
                      c12c13=c12c13,order=2):
    if not isinstance(labels,(list,np.ndarray)):
        labels = np.array([Teff,logg,vturb,ch,nh,oh,nah,mgh,alh,sih,sh,kh,cah,
                           tih,vh,mnh,nih,feh,c12c13])
    
    # make quadratic labels
    linear_terms = np.array(labels) - reference_point
    if order == 1:
        lvec = np.hstack((linear_terms))
        # generate spectrum                                                 
        lin_coeff = coeff_array.T[:len(linear_terms)].T
        spec_generate = np.dot(lin_coeff,lvec) + reference_flux
    if order == 1.5:
        linind = 19
        t = linear_terms[0]
        g = linear_terms[1]
        f = linear_terms[17]
        fit_terms = np.array([t**2,t*g,t*f,g**2,g*f,f**2])
        lvec = np.hstack((linear_terms,fit_terms))
        coeffs = np.array([coeff_array[:,0+linind],coeff_array[:,1+linind],
                           coeff_array[:,17+linind],
                           coeff_array[:,19+linind],
                           coeff_array[:,35+linind],
                           coeff_array[:,187+linind]])
        coeffs = np.concatenate((coeff_array.T[:len(linear_terms)],
                                 coeffs)).T
        spec_generate = np.dot(coeffs,lvec) + reference_flux
    if order == 2:
        quadratic_terms = np.einsum('i,j->ij',linear_terms,
                                    linear_terms)[inds[:,0],inds[:,1]]
        lvec = np.hstack((linear_terms, quadratic_terms))
        # generate spectrum                                                
        spec_generate = np.dot(coeff_array,lvec) + reference_flux
    
    return spec_generate

linind = 19
lin_coeff = coeff_array.T[:linind].T
quad_coeff = np.array([coeff_array[:,0+linind],coeff_array[:,19+linind],
                       coeff_array[:,37+linind],coeff_array[:,54+linind],
                       coeff_array[:,70+linind],coeff_array[:,85+linind],
                       coeff_array[:,99+linind],coeff_array[:,112+linind],
                       coeff_array[:,124+linind],coeff_array[:,135+linind],
                       coeff_array[:,145+linind],coeff_array[:,154+linind],
                       coeff_array[:,162+linind],coeff_array[:,169+linind],
                       coeff_array[:,175+linind],coeff_array[:,180+linind],
                       coeff_array[:,184+linind],coeff_array[:,187+linind],
                       coeff_array[:,189+linind]]).T

cross_inds = {0:np.arange(1,19)+linind, #Teff
              1:np.append(np.arange(20,37),[1])+linind,#logg
              2:np.append(np.arange(38,54),[2,20])+linind, #vturb
              3:np.append(np.arange(55,70),[3,21,38])+linind, #ch
              4:np.append(np.arange(71,85),[4,22,39,55])+linind, #nh
              5:np.append(np.arange(86,99),[5,23,40,56,71])+linind, #oh
              6:np.append(np.arange(100,112),[6,24,41,57,72,86])+linind, #nah
              7:np.append(np.arange(113,124),[7,25,42,58,73,87,100])+linind, #mgh
              8:np.append(np.arange(125,135),[8,26,43,59,74,88,101,113])+linind, #alh
              9:np.append(np.arange(136,145),[9,27,44,60,75,89,102,114,125])+linind, #sih
              10:np.append(np.arange(146,154),[10,28,45,61,76,90,103,115,126,136])+linind, #sh
              11:np.append(np.arange(155,162),[11,29,46,62,77,91,104,116,127,137,146])+linind, #kh
              12:np.append(np.arange(163,169),[12,30,47,63,78,92,105,117,128,138,147,155])+linind, #cah
              13:np.append(np.arange(170,175),[13,31,48,64,79,93,106,118,129,139,148,156,163])+linind, # tih
              14:np.append(np.arange(176,180),[14,32,49,65,80,94,107,119,130,140,149,157,164,170])+linind, #vh
              15:np.append(np.arange(181,184),[15,33,50,66,81,95,108,118,131,141,150,158,165,171,176])+linind, #mnh
              16:np.append(np.arange(185,187),[16,34,51,67,82,96,109,119,132,142,151,159,166,172,177,181])+linind, #nih
              17:np.append(np.arange(188,189),[17,35,52,68,83,97,110,120,133,143,152,160,167,173,178,182,185])+linind, #feh
              18:np.array([18,36,53,69,84,98,111,121,134,144,153,161,168,174,179,183,186,188])+linind #c12c13
          }

def new_reference(labels=None,Teff=Teff,logg=logg,vturb=vturb,ch=ch,nh=nh,
                  oh=oh,nah=nah,mgh=mgh,alh=alh,sih=sih,sh=sh,kh=kh,
                  cah=cah,tih=tih,vh=vh,mnh=mnh,nih=nih,feh=feh,
                  c12c13=c12c13,order=2,newref=reference_point):
    if not isinstance(labels,(list,np.ndarray)):
        labels = np.array([Teff,logg,vturb,ch,nh,oh,nah,mgh,alh,sih,sh,kh,cah,
                           tih,vh,mnh,nih,feh,c12c13])
    refdiff = newref-reference_point
    newconst = generate_spectrum(labels=newref)
    linear_terms = np.array(labels) - newref
    quadratic_terms = np.einsum('i,j->ij',linear_terms,
                                    linear_terms)[inds[:,0],inds[:,1]]
    spec = newconst
    if order == 1:
        for l in range(linind):
            coeff = lin_coeff[:,l] + 2*refdiff[l]*quad_coeff[:,2]
            for c in range(linind-1):
                coeff += -coeff_array[:,cross_inds[l][c]]*refdiff[c]
            spec += linear_terms[l]*(coeff)

    if order == 1.5:
        for l in range(linind):
            coeff = lin_coeff[:,l] + 2*refdiff[l]*quad_coeff[:,2]
            for c in range(linind-1):
                coeff += -coeff_array[:,cross_inds[l][c]]*refdiff[c]
            spec += linear_terms[l]*(coeff)
        t = linear_terms[0]
        g = linear_terms[1]
        f = linear_terms[17]
        fit_terms = np.array([t**2,t*g,t*f,g**2,g*f,f**2])
        lvec = np.hstack((fit_terms))
        coeffs = np.array([coeff_array[:,0+linind],coeff_array[:,1+linind],
                           coeff_array[:,17+linind],
                           coeff_array[:,19+linind],
                           coeff_array[:,35+linind],
                           coeff_array[:,187+linind]]).T
        
        spec += np.dot(coeffs,lvec)
    
    if order == 2:
        for l in range(linind):
            coeff = lin_coeff[:,l] + 2*refdiff[l]*quad_coeff[:,2]
            for c in range(linind-1):
                coeff += -coeff_array[:,cross_inds[l][c]]*refdiff[c]
            spec += linear_terms[l]*coeff
            
        lvec = np.hstack((quadratic_terms))
        # generate spectrum       
        spec += np.dot(coeff_array.T[linind:].T,lvec)
            
    return spec
