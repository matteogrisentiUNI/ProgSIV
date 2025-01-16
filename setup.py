from setuptools import setup, find_packages

setup(
    name="object_tracker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics",
        "opencv-python",
        "numpy",
        "psutil",
        "matplotlib",
        "networkx",
        "tqdm"
    ],
    author="Adami Filippo & Grisenti Matteo",
    description="Efficient Object Tracking Library using YOLO and Image Processing",
    url='https://github.com/matteogrisenti/ProgSIV',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=">=3.8",
)
