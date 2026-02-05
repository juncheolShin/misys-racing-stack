from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mppi'

setup(
    name=package_name,
    version='0.0.1',
    packages=['mppi', 'config'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='misys',
    maintainer_email='jcshin2003@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mppi_node = mppi.mppi_node:main'
        ],
    },
)
