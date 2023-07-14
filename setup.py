from setuptools import setup, find_packages
# 

setup(
  name = 'LongNet',
  packages = find_packages(exclude=[]),
  version = '0.3.8',
  license='MIT',
  description = 'LongNet - Pytorch',
  author = 'Kye Gomez',
  author_email = 'kye@apac.ai',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/kyegomez/LongNet',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers',
    "Prompt Engineering"
  ],
    install_requires=[
        'torch',
        'einops',
        'torchscale',
        'transformers',
        'accelerate',
        'bitsandbytes',
        'fairscale',
        'timm',
        'dataclasses',
    ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)