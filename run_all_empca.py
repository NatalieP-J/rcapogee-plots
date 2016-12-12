from comparison_plots import compR2
from empca_residuals import empca_residuals

sample = raw_input('red clump, red giant or cluster?')
nvecs = raw_input('how many eigvecs (Enter for 60):')

if nvecs=='':
    nvecs=60
else:
    nvecs=int(nvecs)

samplekeys = {'rc':'red_clump',
              'red clump':'red_clump',
              'RC':'red_clump',
              'Red Clump':'red_clump',
              'rg':'red_giant',
              'red giant':'red_giant',
              'RG':'red_giant',
              'Red Giant':'red_giant',
              'cluster':'clusters',
              'C':'clusters',
              'clusters':'clusters'}

if sample not in samplekeys.keys():
    sample = raw_input('key not found, please choose red clump, red giant or cluster')

name = samplekeys[sample]

sample = empca_residuals('apogee',name,maskFilter,ask=True)

sample.directoryClean()

sample.findResiduals()

print 'DONE RESIDUALS'

sample.pixelEMPCA(nvecs=nvecs,mad=False,savename='corrNone_madFalse.pkl')

print 'DONE EMPCA 8'

sample.pixelEMPCA(nvecs=nvecs,mad=True,savename='corrNone_madTrue.pkl')

print 'DONE EMPCA 10'

sample.pixelEMPCA(nvecs=nvecs,mad=False,correction='pickles/n6819_13_30px.pkl',
                  savename='corr30px_madFalse.pkl')

print 'DONE EMPCA 9'

sample.pixelEMPCA(nvecs=nvecs,mad=True,correction='pickles/n6819_13_30px.pkl',
                  savename='corr30px_madTrue.pkl')

print 'DONE EMPCA 11'
