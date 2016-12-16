from comparison_plots import comp_R2
from empca_residuals import empca_residuals,maskFilter

sample = raw_input('red clump, red giant or cluster? ')
nvecs = raw_input('how many eigvecs (Enter for 60): ')
minsnr = raw_input('minimum signal to noise? (Enter for 50): ')

if nvecs=='':
    nvecs=60
else:
    nvecs=int(nvecs)
if minsnr=='':
    minsnr=50
else:
    minsnr=int(minsnr)

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
              'c':'clusters',
              'clusters':'clusters'}

correction = {'13':'pickles/n6819_13_30px.pkl', 
              '12':'pickles/n6819_12_30px.pkl'}

if sample not in samplekeys.keys():
    sample = raw_input('key not found, please choose red clump, red giant or cluster ')

name = samplekeys[sample]

sample = empca_residuals('apogee',name,maskFilter,ask=True)

sample.directoryClean()

sample.findResiduals()

corr = correction[sample.DR]

print 'DONE RESIDUALS'

sample.pixelEMPCA(nvecs=nvecs,mad=False,
                  savename='eig{0}_minSNR{1}_corrNone_madFalse.pkl'.format(nvecs,minsnr))

print 'DONE EMPCA 8'

sample.pixelEMPCA(nvecs=nvecs,mad=True,
                  savename='eig{0}_minSNR{1}_corrNone_madTrue.pkl'.format(nvecs,minsnr))

print 'DONE EMPCA 10'

sample.pixelEMPCA(nvecs=nvecs,mad=False,correction=corr,
                  savename='eig{0}_minSNR{1}_corr30px_madFalse.pkl'.format(nvecs,minsnr))

print 'DONE EMPCA 9'

sample.pixelEMPCA(nvecs=nvecs,mad=True,correction=corr,
                  savename='eig{0}_minSNR{1}_corr30px_madTrue.pkl'.format(nvecs,minsnr))

print 'DONE EMPCA 11'
