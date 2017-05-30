from setuptools import setup

setup(name='spectralspace',
      version='1.',
      description='Spectral space dimensionality in APOGEE',
      author='Natalie Price-Jones',
      author_email='price-jones@astro.utoronto.ca',
      url='https://github.com/NatalieP-J/spectralspace',
      package_dir = {'spectralspace/': ''},
      packages=['spectralspace','spectralspace/examples','spectralspace/data']
      package_data={'spectralspace/data':['clusterdata/elemVariations_DR12_M67.sav',
                                          'clusterdata/aj485195t4_mrt.txt',
                                          'clusterdata/aj509073t2_mrt.txt']},
      dependency_links = ['https://github.com/jobovy/apogee/tarball/master#egg=apogee',
                          'https://github.com/NatalieP-J/empca/tarball/master#egg=empca'],
      install_requires=['numpy','scipy','matplotlib','jupyter','tqdm','astropy',
                        'statsmodels','sklearn']
      )
