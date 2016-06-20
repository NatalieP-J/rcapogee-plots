import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from mask_data import maskFilter
from empca_residuals import empca_residuals,smallEMPCA


font = {'family': 'serif',
        'weight': 'normal',
        'size'  :  20
}

matplotlib.rc('font',**font)
plt.ion()

class eigenvector_project(empca_residuals):
    """
    
    Contains functions to transform a set of residuals to a given set of 
    eigenvectors
    
    """
    def __init__(self,sampleType,maskFilter,ask=True,degree=2,
                 fname='madFalse_corr.pkl',useEMPCAres=True,numberSamples=1):
        """
        Fit a masked subsample.
        
        sampleType:      designator of the sample type - must be a key in 
                         readfn and independentVariables in data.py
        maskFilter:      function that decides on elements to be masked
        ask:             if True, function asks for user input to make 
                         filter_function.py, if False, uses existing 
                         filter_function.py
        degree:          degree of polynomial to fit
        fname:           file name from which to draw eigenvectors - directory
                         will be taken from filter_function.py
        useEMPCAres:     if True include stars used in EMPCA eigenvector 
                         generation
        numberSamples:   number of additional samples to use
        
        """
        empca_residuals.__init__(self,sampleType,maskFilter,ask=ask,
                                 degree=degree)
        self.findResiduals(gen=False)
        self.stars = self.residuals
        self.pixelEMPCA(gen=False,savename=fname)
        self.eigvec = self.empcaModelWeight.eigvec
        self.coeff = self.empcaModelWeight.coeff
        self.sections = {}
        self.sections[0] = self.residuals.shape[0]
        for i in range(numberSamples):
            sample = raw_input('Sample type? ')
            if sample =='rc' or sample=='red_clump':
                sampleType='red_clump'
            elif sample=='c' or sample=='clusters':
                sampleType='clusters'
            empca_residuals.__init__(self,sampleType,maskFilter,ask=True,
                                     degree=degree)
            self.findResiduals(gen=False)
            self.sections[i+1] = self.residuals.shape[0]
            self.stars=np.concatenate((self.stars,self.residuals))
        

    def projection(self):
        """
        Project all stars along eigenvectors.
        """
        self.coords = np.zeros((len(self.eigvec),self.stars.shape[0]))
        e=0
        for eig in self.eigvec:
            self.coords[e] = np.dot(self.stars,eig)
            e+=1

    def plot_projection(self,ax1=0,ax2=1,**kwargs):
        """
        Plot projected coordinates.

        ax1:   x-axis to use
        ax2:   y-axis to use
        """
        self.projection()
        plt.figure(figsize=(10,8))
        plt.plot(self.coords[ax1],self.coords[ax2],'.',**kwargs)
        
