from itertools import combinations
import numpy as np

def codedcolumns(x,order):
    indepvals = {}
    for o in range(1,order+1):
        for n in range(len(x)):
            indepvals['i{0}_o{1}'.format(n,o)] = x[n]**o
    return indepvals

def matrixterms(x,order,cross=True):
    orderdict = {}
    indepinds = {}
    indepvals = codedcolumns(x,order)
    for o in range(1,order+1):
        for n in range(len(x)):
            orderdict['i{0}_o{1}'.format(n,o)] = o
            indepinds['i{0}_o{1}'.format(n,o)] = n
    columncode = []
    matrixcolumns = []
    for pcombnum in range(1,len(x)+1):
        combos = [tuple(i) for i in combinations(indepvals.keys(),pcombnum)]
        for c in combos:
            totalindep = np.ones(len(x[0]))
            totalorder = 0
            inds = []
            for key in c:
                totalorder += orderdict[key]
                inds.append(indepinds[key])
                totalindep *= indepvals[key]
            matrixcolumns.append(totalindep)
            columncode.append(c)
            # If no cross terms are to be included
            if cross == False:
                if totalorder > order:
                    columncode.pop()
                    matrixcolumns.pop()
                elif len(inds) > 1:
                    columncode.pop()
                    matrixcolumns.pop()
            # If all cross terms are to be included
            if cross == True:
                if totalorder > order:
                    columncode.pop()
                    matrixcolumns.pop()
                elif len(inds) > 1:
                    if all(i == inds[0] for i in inds):
                        columncode.pop()
                        matrixcolumns.pop()
            # If only some cross terms are to be included
            elif isinstance(cross,tuple):
                if totalorder > order:
                    columncode.pop()
                    matrixcolumns.pop()
                elif len(inds) > 1:
                    if all(i == inds[0] for i in inds):
                        columncode.pop()
                        matrixcolumns.pop()
                    elif not all(i==inds[0] for i in inds):
                        for partial in cross:
                            print inds
                            if not set(inds) <= set(partial):
                                columncode.pop()
                                matrixcolumns.pop()

    return np.array(matrixcolumns),columncode

def makematrix(x,order,cross = True):
    """
    Creates a matrix of the independent variable(s) for use in linear regression.
    
    x:       array of independent variable 
             (may be tuple containing arrays of multiple variables)
    order:   order of polynomial to fit
    

    Returns a matrix constructed from the independent variable(s).
    """
    if isinstance(x,tuple):
        matrixcols,colcode = matrixterms(x,order,cross=cross)
        X = np.empty((len(x[0]),len(matrixcols)+1))
        X[:,0] = x[0]**0
        for mcol in range(len(matrixcols)):
            X[:,mcol+1] = matrixcols[mcol]
    elif isinstance(x,(list,np.ndarray)):
        X = np.empty((len(x),order+1))
        for o in range(order+1):
            X[:,o] = x**o
    X = np.matrix(X)
    return X,colcode

def regfit(X,y,C = 0,order = 1):
    """
    Fits a (nD-)polynomial of specified order with independent values given in x,
    given dependent values in y.
    
    x:       array of independent variable 
             (may be tuple containing arrays of multiple variables)   
    y:       array of dependent variable (must have same shape as array in x)
    order:   order of polynomial to fit (kwarg, default = 1)
    
    Returns the polynomial coefficents ascending from 0th order. In the case of
    multiple independent variables, returns coefficients at each polynomial order
    in order of the variables listed in x.
    """
    if isinstance(C,(float,int)):
        return np.array(np.linalg.inv(X.T*X)*X.T*np.matrix(y).T)
    elif isinstance(C,(list,np.ndarray)):
        icov = np.linalg.inv(C)
        return np.array(np.linalg.inv(X.T*icov*X)*(X.T*icov*np.matrix(y).T))

def poly(p,colcode,x,order = 1):
    """
    For a given set of polynomial coefficients ascending from
    0th order, and a independent variables, returns polynomial.
    
    p:       coefficients of the polynomial in ascending order
    x:       array of independent variable
            (may be tuple containing arrays of multiple variables)
    order:   order of polynomial (kwarg, default = 1)
    
    Returns an array of polynomial values.
    """
    indepvals = codedcolumns(x,order)
    if isinstance(x,tuple):
        nindeps = len(x)
        order = (len(p)-1)/nindeps
        y = np.zeros(x[0].shape)
        y += p[0]*x[0]**0
        for i in range(1,len(p)):
            multi = np.ones(len(x[0]))
            for key in colcode[i-1]:
                multi *= indepvals[key]
            y += p[i]*multi

    elif isinstance(x,(list,np.ndarray)):
        order = len(p)-1
        y = np.zeros(x.shape)
        o = 0
        while o <= order:
            y += p[o]*x**o
            o += 1
    return y

def idealerrs(p,x,err,order=1):
    X = makematrix(x,order)
    C = np.diag(err**2)
    return np.array(np.linalg.inv(X.T*C*X))

def bootstrap(p,x,y,err,order=1,ntrials = 10):
    params = np.zeros((ntrials,len(p)))
    for n in range(ntrials):
        if isinstance(x,(list,np.ndarray)):
            sample = np.random.choice(x,size = x.shape,replace = True,p = None) #set p = 1/err**2?
        elif isinstance(x,tuple):
            sample = ()
            for indep in x:
                sample += (np.random.choice(indep,size = indep.shape,replace=True,p=None),)
        params[n] = regfit(sample,y,err = err,order = order)
    return (1./ntrials)*np.sum((params-p)**2,axis = 0)
    
def jackknife(p,x,y,err,order=1):
    params = np.zeros((len(x),len(p)))
    for n in range(len(x)):
        if isinstance(x,(list,np.ndarray)):
            params[n] = regfit(np.delete(x,n),np.delete(y,n),err = np.delete(err,n),order=order)
        elif isinstance(x,tuple):
            sample = ()
            for indep in x:
                sample += (np.delete(indep,n),)
            params[n] = regfit(sample,np.delete(y,n),err = np.delete(err,n),order=order)
    finalparam = (1./len(x))*np.sum(params,axis = 0)
    return finalparam,((len(x)-1)/len(x))*np.sum((params-finalparam)**2,axis = 0)


def normweights(weights):
    """
    Normalize an array of weights by dividing them by their sum.
    
    weights:     an array of weights
    
    Returns an array of the normalized weights.
    """
    return weights/np.sum(weights)

def genresidual(weights,residuals):
    """
    Generate a single residual value from an array of fit residuals weighted
    according to an array of weights.
    
    weights:     an array of weights
    residuals:   an array of residuals from a fit for a single star 
    
    Returns a single residual value.
    """
    return np.sum(weights*residuals)

def magenresidual(weights,residuals):
    """
    Generate an array of residual values from a 2D array of fit residuals 
    weighted according to an array of weights.
    
    weights:     an array of weights
    residuals:   a 2D array of residuals from a fit, where the 0-axis spans 
                 pixel and the 1-axis spans stars
                 
    Returns an array of residual values
    """
    return np.array(np.matrix(weights)*np.matrix(residuals))[0]
