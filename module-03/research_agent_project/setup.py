"""
Setup configuration for the Research Agent package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="research-agent",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-ready research agent built with LangGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/research-agent",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-community>=0.0.13",
        "langgraph>=0.0.20",
        "duckduckgo-search>=4.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "postgres": ["psycopg2-binary>=2.9.0"],
        "async": ["aiosqlite>=0.19.0"],
    },
    entry_points={
        "console_scripts": [
            "research-agent=research_agent.__main__:main",
        ],
    },
)
