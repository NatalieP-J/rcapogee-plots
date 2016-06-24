import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation,KMeans
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
                 fname='madFalse_corr.pkl',useEMPCAres=True,numberSamples=1
                 direclist = []):
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
        self.errs = np.median(self.spectra_errs,axis=1)
        self.pixelEMPCA(gen=False,savename=fname)
        self.eigvec = self.empcaModelWeight.eigvec
        self.coeff = self.empcaModelWeight.coeff
        self.sections = {}
        self.sections[0] = (0,self.residuals.shape[0])
        for i in range(numberSamples):
            if direclist == []:
                sample = raw_input('Sample type? ')
                if sample =='rc' or sample=='red_clump':
                    sampleType='red_clump'
                elif sample=='c' or sample=='clusters':
                    sampleType='clusters'
                empca_residuals.__init__(self,sampleType,maskFilter,ask=True,
                                         degree=degree)
                self.findResiduals(gen=False)
            elif direclist != []:
                self.residuals = np.load(direclist[i]+'/residuals.npy')
                self.spectra_errs = np.load(direclist[i]+'/spectra_errs.npy')
            start = self.sections[i][1]+1
            self.sections[i+1] = ((start,
                                   start+self.residuals.shape[0]))
            self.stars=np.concatenate((self.stars,self.residuals))
            self.errs=np.concatenate((self.errs,
                                      np.median(self.spectra_errs,axis=1)))
        

    def projection(self):
        """
        Project all stars along eigenvectors.
        """
        self.coords = np.zeros((len(self.eigvec),self.stars.shape[0]))
        e=0
        for eig in self.eigvec:
            self.coords[e] = np.dot(self.stars,eig)
            e+=1

    def plot_projection(self,ax1=0,ax2=1,bins=200):
        """
        Plot projected coordinates.

        ax1:   x-axis to use
        ax2:   y-axis to use
        bins:  number of bins in the histogram
        """
        H,xedges,yedges = np.histogram2d(self.coords[ax1],self.coords[ax2],
                                         bins=bins)
    	# Reorient appropriately
    	H = np.rot90(H)
    	H = np.flipud(H)
        Hmasked = np.ma.masked_where(H==0,H)
        plt.figure(figsize=(10,8))
        plt.pcolormesh(xedges,yedges,Hmasked,
                       cmap = plt.get_cmap('plasma'))
        plt.xlabel('Projection on eigenvector {0}'.format(ax1+1))
        plt.ylabel('Projection on eigenvector {0}'.format(ax2+1))
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Number of stars')    
        plt.xlabel('Projection on eigenvector {0}'.format(ax1+1))
        plt.ylabel('Projection on eigenvector {0}'.format(ax2+1))

    def find_clusters(self,algorithm=KMeans,**kwargs):
        """
        Locate clusters in data after it has been projected along eigenvectors.
        
        algorithm:   scikit-learn algorithm to use (KMeans, AffinityPropagation                      already imported)
        **kwargs:    keyword arguments for chosen algorithm

        """
        self.projection()
        cluster_find = algorithm(**kwargs)
        cluster_find.fit(self.coords.T)
        self.centers = cluster_find.cluster_centers_
        self.labels = cluster_find.predict(self.coords.T)

    def sort_labels(self):
        """

        Reindex clusters by cluster population (cluster 0 has highest pop). 
        Also sorts cluster centers accordingly

        """
        clusterpop,binEdges=np.histogram(self.labels,bins=self.centers.shape[0])
        indbypop = clusterpop.argsort()[::-1]
        self.centers = self.centers[indbypop]
        for star in range(len(self.labels)):
            self.labels[star] = np.where(self.labels[star]==indbypop)[0][0]

    def known_clusters(self,sectioninds):
        """
        
        Find centers of clusters known to exist by their indices in the
        self.section dictionary.

        sectionids:   list of keys in self.section to get indices from whole 
                      star list to use as known clusters

        """
        self.known_centers = np.zeros((len(sectioninds),len(self.eigvec)))
        i=0
        for ind in sectioninds:
            start,end = self.sections[ind]
            self.known_centers[i] = np.mean(self.coords.T[start:end])
            i+=1
            
    def known_check(self,sectioninds):
        """
        
        Find distances between known cluster centers and the centers to which
        the stars were assigned by the cluster-finding algorithm.

        sectionids:   list of keys in self.section to get indices from whole 
                      star list to use as known clusters.

        """
        self.known_clusters(sectioninds)
        i=0
        for ind in sectioninds:
            start,end = self.sections[ind]
            cluster_labels = self.labels[start:end]
            points = self.coords.T[start:end]
            known_center = self.known_centers[i]
            distances = np.zeros(len(cluster_labels))
            for l in range(len(cluster_labels)):
                point = points[l]
                center = self.centers[cluster_labels[l]]
                assigneddist = np.sqrt(np.sum((point-center)**2))
                realdist = np.sqrt(np.sum((point-known_center)**2))
                distances[l] = assigneddist/realdist
            self.plotHistogram(distances,norm=False,
                               bins=len(np.unique(distances)))
            plt.xlim(0,1)
            if np.all(cluster_labels==cluster_labels[0]):
                otherstars = len(np.where(self.labels==cluster_labels[0])[0])
                otherstars -= len(cluster_labels)
                print '{0} other stars in this cluster'.format(otherstars)
            i+=1
                
                
        

    def plot_clusters(self,ax1=0,ax2=1,bins=200,knownclusters=None):
        H,xedges,yedges = np.histogram2d(self.coords[ax1],self.coords[ax2],
                                         bins=bins)
	# Reorient appropriately
	H = np.rot90(H)
	H = np.flipud(H)
        Hmasked = np.ma.masked_where(H==0,H)
        plt.figure(figsize=(10,8))
        plt.pcolormesh(xedges,yedges,Hmasked,
                       cmap = plt.get_cmap('plasma'))
        plt.xlabel('Projection on eigenvector {0}'.format(ax1+1))
        plt.ylabel('Projection on eigenvector {0}'.format(ax2+1))
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Number of stars')
        plt.plot(self.centers[:,ax1],self.centers[:,ax2],'o',
                 markerfacecolor='w',markeredgecolor='k',markeredgewidth=1,
                 markersize=5)
        if knownclusters:
            plt.plot(self.known_centers[:,ax1],self.known_centers[:,ax2],'o',
                     markerfacecolor='r',markeredgecolor='k',markeredgewidth=1,
                     markersize=5)
