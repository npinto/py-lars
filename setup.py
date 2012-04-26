import setuptools

setuptools.setup(
    name='lmj.lars',
    version='0.1',
    namespace_packages=['lmj'],
    packages=setuptools.find_packages(),
    author='Leif Johnson',
    author_email='leif@leifjohnson.net',
    description='An implementation of Least Angle Regression',
    long_description=open('README.md').read(),
    license='MIT',
    keywords=('regression '
              'sparse '
              'regularized '
              ),
    url='http://github.com/lmjohns3/py-lars',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        ],
    )
