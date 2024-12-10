import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# from https://justinnhli.com/posts/2020/05/sharing-git-repositories-between-pip-and-setuppy.html

# uncomment if on Airbus network
# DEPENDENCIES = {
#     'transformers': 'https://github.airbus.corp/Airbus/transformers.git',
#     'tsfm-public': 'https://github.airbus.corp/Airbus/tsfm.git',
#     'accelerate': 'https://github.airbus.corp/Airbus/accelerate.git',
# }

# uncomment if on public internet
DEPENDENCIES = {
    'transformers': 'https://github.com/whutabarat-sheffield/transformers-wh.git',
    'tsfm-public': 'https://github.com/ibm-granite/granite-tsfm.git',
    'accelerate': 'https://github.com/huggingface/accelerate.git',
}

def get_dependency(package, location):
    if location == 'install':
        return f'{package} @ git+{DEPENDENCIES[package]}'
    elif location == 'link':
        return f'{DEPENDENCIES[package]}#egg={package}'
    else:
        raise ValueError(f'Unknown location: {location}')

setuptools.setup(
    name="abyss",
    version="2.0.0",
    author="David Miller",
    author_email="d.b.miller@sheffield.ac.uk",
    description="Electric drill data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.airbus.corp/Airbus/uos-drilling",
    project_urls={
        "Bug Tracker": "https://github.airbus.corp/Airbus/uos-drilling/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'openpyxl',
        'h5py',
        'nptdms',
        'scaleogram',
        'torch',
        'torchvision',
        'torchaudio',
        'numpy',
        'pandas',
        'matplotlib',
        # 'pytorch-cuda',
        'tensorboardX',
        'ipykernel',
        'jupyter',
        get_dependency('transformers', location='install'),
        get_dependency('tsfm-public', location='install'),
        get_dependency('accelerate', location='install'),
    ],
    dependency_links=[
        get_dependency('transformers', location='link'),
        get_dependency('tsfm-public', location='link'),
        get_dependency('accelerate', location='link'),
    ],
    entry_points={
        'console_scripts': [
            'depth-est = abyss:run.uos_depth_estimation',
        ],
    },
    include_package_data=True,
)