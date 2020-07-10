from setuptools import setup, find_packages

setup(
    name='search',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click', 'numpy', 'matplotlib'
    ],
    entry_points='''
        [console_scripts]
        search=search:cli
    ''',
)