from setuptools import setup, find_packages

setup(
    name="incident-rca-env",
    version="1.0.0",
    description="OpenEnv environment for incident response & root cause analysis RL training",
    author="Your Name",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.29.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "baseline": ["openai>=1.0.0"],
        "test": ["pytest>=8.0", "httpx>=0.27.0"],
    },
    entry_points={
        "console_scripts": [
            "incident-rca-server=environment.server:app",
        ],
    },
)
