import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'two_wheeled_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '**', '*.launch.py'), recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Two wheeled robot package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_follower_complete = two_wheeled_robot.scripts.line_follower_complete:main',
            'random_drive = two_wheeled_robot.scripts.random_drive:main',
            'train = two_wheeled_robot.rl.train:main',
        ],
    },
)