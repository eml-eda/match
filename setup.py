import setuptools

setuptools.setup(
    name='match',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=['setuptools'],
    setup_requires=['setuptools_scm'],
    include_package_data=True,
)