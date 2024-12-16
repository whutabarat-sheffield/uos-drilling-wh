import os
import setuptools

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# These are local tarballs of the dependencies
LOCAL_DEPENDENCIES = {
    'transformers': 'transformers-4.41.0.dev0.tar.gz',
    'tsfm-public': 'tsfm_public-0.2.17.tar.gz',
    'accelerate': 'accelerate-1.2.1.tar.gz',
}

def get_dependency(package, location):
    if location == 'git':
        return f'{package} @ git+{DEPENDENCIES[package]}'
    elif location == 'link':
        return f'{DEPENDENCIES[package]}#egg={package}'
    elif location == 'local':
        local_dep = os.path.join(CURRENT_DIR, '..', 'deps', f'{LOCAL_DEPENDENCIES[package]}')
        return f'{package} @ file:///{local_dep}'
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
        # 'torch@https://download.pytorch.org/whl/cu118/torch-2.3.1%2Bcu118-cp310-cp310-win_amd64.whl',
        'torch==2.3.1+cu118', # This ensures that the CUDA 11.8 version of torch is installed
        'torchvision',
        'torchaudio',
        'numpy',
        'pandas',
        'matplotlib',
        'tensorboardX',
        'ipykernel',
        'jupyter',
        'paho-mqtt', # MQTT client
        'pyyaml', # YAML parser
        get_dependency('transformers', location='local'),
        get_dependency('tsfm-public', location='local'),
        get_dependency('accelerate', location='local'),
    ],
    entry_points={
        'console_scripts': [
            'depth-est = abyss:run.uos_depth_estimation',
        ],
    },
    include_package_data=True,
)