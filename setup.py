from setuptools import setup

setup(
    name='XtrViz',
    version='1.0',
    author='Aaron Gerston (X-trodes LTD)',
    author_email='aarong@xtrodes.com',
    packages=['XtrViz'],
    include_package_data=True,
    license='GNU GPLv3',
    long_description=open('README.md').read(),
    url="https://github.com/aarongerston/XtrViz/",
    install_requires=open('requirements.txt').read()
)
