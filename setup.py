from setuptools import setup


hyperdopamine_description = (
    '"Learning to Represent Action Values as a Hypergraph on the Action Vertices" built on Dopamine')

setup(
    name='hyperdopamine',
    version='0.1.0',
    description=hyperdopamine_description,
    url='https://github.com/atavakol/action-hypergraph-networks',
    python_requires='>= 3.5',
    install_requires=['gin-config', 'absl-py', 'opencv-python', 'atari-py', 
                      'gym', 'matplotlib', 'pandas', 'numpy'],
    license='MIT',
    keywords=['hyperdopamine', 
              'dopamine', 
              'reinforcement learning', 
              'deep learning', 
              'machine learning', 
              'python']
)
