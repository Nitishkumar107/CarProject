from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(path: str) -> List[str]:
    ''' This function returns a list of requirements'''
    requirements = []

    with open (path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n",'') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements





setup(
    name='ml_project',
    version='0.1',
    author='nitish',
    author_email='nitish@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)