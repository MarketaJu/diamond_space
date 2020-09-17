from setuptools import setup
import os

current_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_folder, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='diamond_space',
    version='0.0.1',
    description='Diamond space transform for python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MarketaJu/diamond_space',
    author='Marketa Jurankova',
    author_email='jurankovam@fit.vutbr.cz',
    license='BSD3',
    keywords='diamond space, pclines, hough transform, vanishing point detection, cascaded hough transform',
    packages=["diamond_space"],
    install_requires=['numpy','scikit-image'],
    python_requires='>=3.6',
    project_urls={
        "Bug reports": 'https://github.com/MarketaJu/diamond_space/issues',
    },
)
