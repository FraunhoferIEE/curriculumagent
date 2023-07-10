"""Setup-script for curriculumagent package."""
import os
import sys
import importlib
from setuptools import setup, find_packages
import versioneer

PACKAGE_NAME = 'curriculumagent'  # name used by PIP
IMPORT_NAME = 'curriculumagent'  # name used for imports
NAMESPACE = ""

with open('requirements.txt', 'r') as f:
    requirements_install = f.readlines()

# Add all the packages in requirements.txt to an extra named '[dev]'
with open('requirements-dev.txt', 'r') as f:
    requirements_dev = f.readlines()

# All the requirements
requirements = {
    'install': [
        r for r in requirements_install if not r.startswith('-r ')
    ],
    'test': [  # testing
        'pytest',
        'coverage',
    ],
    'docs': [  # documentation generation and tools
        'Sphinx',
        'sphinx-rtd-theme',  # nicer theme for the html docs
        'sphinx-autodoc-typehints',  # use typehints to provide types
    ],
    'dev': [
        r for r in requirements_dev if not r.startswith('-r ')
    ],
}

URLS = {'GITHUB': 'https://github.com/FraunhoferIEE/curriculumagent/',
        'Readthedocs': 'https://curriculumagent.readthedocs.io/en/latest/'
}
SCRIPTS = []  # no scripts necessary
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Programming Language :: Python :: 3.10',
]

# ======================================================================================
# No configurable options below this line

# Get current version etc. from the `__about__.py` file --------------------------------
# Very evil way to retrieve the variables contained in about.
# Those will be added to the current scope
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
about = importlib.import_module(IMPORT_NAME + '.__about__')

# def find_version(*file_paths):
#    """Function used to find the version of the package."""
#    version_file = read(*file_paths)
#    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
#                              version_file, re.MULTILINE)
#    if version_match:
#        return version_match.group(1)
#    raise RuntimeError("Unable to find version string for the package.")


if __name__ == '__main__':
    setup(
        # General ---
        name=PACKAGE_NAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),

        packages=find_packages(exclude=['docs', 'tests', 'tests.*']),
        namespace_packages=([NAMESPACE] if NAMESPACE else []),
        scripts=SCRIPTS,

        install_requires=requirements['install'],
        extras_require={k: r for k, r in requirements.items() if k != 'install'},

        # Testing ---
        test_suite='pytest.collector',
        tests_require=requirements['test'],

        package_data={
            # If any package contains *.rst files, include them:
            '': ['*.rst', '*.npy'],
        },

        # Metadata --- (needed if we ever have a private Package Index)
        author=about.__author__,
        description=about.__description__,
        classifiers=CLASSIFIERS,
        long_description = open('README.md').read(),
        long_description_content_type = 'text/markdown',
        #long_description='CurriculumAgent is a cleanup and improved version of the NeurIPS 2020 Competition Agent by '
        #                 'Binbinchen.The agent is build to extract action sets of the Grid2Op Environment and then '
        #                 'use rule-based agent to train a Reinforcement Learning agent.',
        # license='???',
        # url='???'  # should link somewhere, perhaps to the docs
        project_urls=URLS,

        # could also include long_description, download_url, classifiers, etc.
    )
