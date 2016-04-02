from setuptools import setup, find_packages

setup(
    name='sparsex',
    version='0.0.0',
    description='A sample Python project',
    author='Nitish Reddy Koripalli',
    author_email='nitish.k.reddy@gmail.com',
    url='https://bitbucket.org/nitred/sparsex',
    classifiers=[
        'Development Status :: 1 - Development'
        'Programming Language :: Python :: 2.7',
    ],
    packages=find_packages(),
    install_requires=['pip', 'protobuf', 'pyzmq'],
    include_package_data=True,
    entry_points={
        'console_scripts':[
            'sparsex_communication_test=sparsex.tests.pipeline_test:test_pipeline_empty',
            'sparsex_get_features_test=sparsex.tests.pipeline_test:test_pipeline_get_features_from_image_array',
            'sparsex_get_predictions_test=sparsex.tests.pipeline_test:test_pipeline_get_predictions_from_image_array'
        ]
    }
)