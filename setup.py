from setuptools import setup, find_packages

setup(
    name="rl-locomotion-cbf",
    version="0.1.0",
    description="RL Locomotion with Control Barrier Function safety layer",
    author="Ansh Bhansali",
    author_email="anshbhansali5@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "osqp>=0.6.0",
        "gym>=0.21.0",
        "pybullet>=3.2.0",
        "stable-baselines3>=1.6.0",
        "torch>=1.10.0",
    ],
    python_requires=">=3.8",
)
