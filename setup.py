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
    author="Your Name",
    description="Efficient Object Tracking Library using YOLO and Image Processing",
    python_requires=">=3.8",
)
