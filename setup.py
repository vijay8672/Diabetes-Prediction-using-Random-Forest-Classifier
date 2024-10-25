from setuptools import find_packages, setup
from src.utils_code.utils import get_requirements



setup(
name= 'Diabetes prediction using Machine Learning Techniques', 
version='0.0.1',
author='Vijay Kumar Kodam',
author_email='vijay.kodam98@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)