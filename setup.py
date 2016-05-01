from setuptools import setup, find_packages

setup(
    name='sparsex',
    version='0.0.0.dev2',
    description='Object recognition library using sparse coding for feature extraction.',
    long_description='This is an RnD project part of the curriculum for the Master of Autonomous Systems program in University of Applied Sciences Bonn-Rhein-Sieg. The aim is to build an object recognition tool using sparse coding for feature extraction. \
    Project is currently maintained by Nitish Reddy Koripalli (21st February 2016).',
    author='Nitish Reddy Koripalli',
    author_email='nitish.k.reddy@gmail.com',
    url='https://bitbucket.org/nitred/sparsex',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 2.7',
    ],
    license='MIT',
    install_requires=['pip==8.1.1', 'Pillow==3.2.0', 'numpy==1.10.4', 'scipy==0.17.0', 'scikit-learn==0.17.1',
     'scikit-image==0.12.3', 'h5py==2.6.0', 'pyzmq==15.2.0', 'protobuf==2.6.1'],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts':[
            'sparsex_communication_test=sparsex.tests.pipeline_test:test_pipeline_empty',
            'sparsex_get_features_test=sparsex.tests.pipeline_test:test_pipeline_get_features_from_image_array',
            'sparsex_get_predictions_test=sparsex.tests.pipeline_test:test_pipeline_get_predictions_from_image_array'
        ]
    }
)