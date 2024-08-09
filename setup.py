from typing import List
from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e .'
def get_requirements(filepath:str)->List[str]:
    '''
    this function will return the list of all the dependecies in requirements.txt file
    '''
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [reqirement.replace("\n","") for reqirement in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(

    name='Ml-datapipeline-project',
    version='0.0.1',
    author='Shiza Azam',
    author_email='shizaazam6@gamil.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)