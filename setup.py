from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return requirements
    '''
    requirements = []
    with open(file_path) as file_object:
        requirements = file_object.readline()
        requirements = [req.replace('\n','') for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)



setup(
    name= "Student Performace Indicator ML Project",
    version = "0.0.1",
    author="Ujjwal Solanki",
    author_email="ujjwal.programmer@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)