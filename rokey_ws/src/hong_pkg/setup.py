from setuptools import find_packages, setup

package_name = 'hong_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='mhi1248@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'depth_camera_mouse = hong_pkg.depth_camera_mouse:main',
	'depth_camera_yolo = hong_pkg.depth_camera_yolo:main',
	'depth = hong_pkg.depth:main',
	'nav_through_poses = hong_pkg.nav_through_poses:main',
	'follow_waypoints = hong_pkg.follow_waypoints:main',
	'depth_goal = hong_pkg.depth_goal:main',
	'test = hong_pkg.node_test:main',
	'main = hong_pkg.main:main'
	'hong_yolo_compact = hong_pkg.hong_yolo_compact:main',
        'hong_yolo_compact2 = hong_pkg.hong_yolo_compact2:main',
        'hong_depth = hong_pkg.hong_depth:main',
        '3_3_c_depth_to_nav_goal = hong_pkg.3_3_c_depth_to_nav_goal:main'
        ],
    },
)
