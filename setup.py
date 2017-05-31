from setuptools import setup
import warnings

setup(name='spectralspace',
      version='1.',
      description='Spectral space dimensionality in APOGEE',
      author='Natalie Price-Jones',
      author_email='price-jones@astro.utoronto.ca',
      url='https://github.com/NatalieP-J/spectralspace',
      package_dir = {'spectralspace/': ''},
      packages=['spectralspace','spectralspace/examples','spectralspace/data','spectralspace/analysis','spectralspace/sample'],
      package_data={'spectralspace/data':['clusterdata/elemVariations_DR12_M67.sav',
                                          'clusterdata/aj485195t4_mrt.txt',
                                          'clusterdata/aj509073t2_mrt.txt']},
      dependency_links = ['https://github.com/jobovy/apogee/tarball/master#egg=apogee',
                          'https://github.com/NatalieP-J/empca/tarball/master#egg=empca',
                          'https://github.com/jobovy/galpy/tarball/master#egg=galpy',
                          'https://github.com/jobovy/isodist/tarball/master#egg=isodist'],
      install_requires=['numpy','scipy','matplotlib','jupyter','tqdm','astropy',
                        'statsmodels','scikit-learn','apogee','galpy','isodist']
      )

warnings.warn('''APOGEE installation requires environment variables to be set: SDSS_LOCAL_SAS_MIRROR=<path to file storage>, RESULTS_VERS=v603, APOGEE_APOKASC_REDUX=v7.3''')
