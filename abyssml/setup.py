import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# from https://justinnhli.com/posts/2020/05/sharing-git-repositories-between-pip-and-setuppy.html
DEPENDENCIES = {
    'transformers': 'https://github.airbus.corp/Airbus/transformers.git',
    'tsfm-public': 'https://github.airbus.corp/Airbus/tsfm.git',
    'accelerate': 'https://github.airbus.corp/Airbus/accelerate.git',
}

def get_dependency(package, location):
    if location == 'install':
        return f'{package} @ git+{DEPENDENCIES[package]}'
    elif location == 'link':
        return f'{DEPENDENCIES[package]}#egg={package}'
    else:
        raise ValueError(f'Unknown location: {location}')

# setup(
#     # ...
#     install_requires=[
#         # ...
#         # experiment packages
#         get_dependency('transformers', location='install'),
#         get_dependency('tsfm', location='install'),
#         # ...
#     ],
#     dependency_links=[
#         get_dependency('transformers', location='link'),
#         get_dependency('tsfm', location='link'),
#     ],
# )


setuptools.setup(
    name="abyssml",
    version="1.0.0",
    author="Ze Zhang",
    author_email="ze.zhang@sheffield.ac.uk",
    description="Electric drill data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zee0549/Drilling_deployable",
    project_urls={
        "Bug Tracker": "https://github.com/Zee0549/Drilling_deployable/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'numpy',
        'pandas',
        'matplotlib',
        'pytorch-cuda',
        'tensorboardX',
        get_dependency('transformers', location='install'),
        get_dependency('tsfm-public', location='install'),
        get_dependency('accelerate', location='install'),
    ],
    dependency_links=[
        get_dependency('transformers', location='link'),
        get_dependency('tsfm-public', location='link'),
        get_dependency('accelerate', location='link'),
    ]
)
