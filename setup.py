from setuptools import setup

with open('requirements.txt') as fp:
    install_requires = fp.read()

    setup(
        name='XtrViz',
        version='1.0',
        author='Aaron Gerston (X-trodes LTD)',
        author_email='aarong@xtrodes.com',
        packages=['XtrViz'],
        include_package_data=True,
        license='GNU GPLv3',
        long_description=open('README.md').read(),
        # url="",
        install_requires=install_requires
    )
