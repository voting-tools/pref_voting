import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pref_voting",                   
    version="0.4.13",     
    author="Eric Pacuit",
    author_email='epacuit@umd.edu',      
    description="pref_voting is a Python packaging that contains tools to reason about election profiles and margin graphs, and implementations of a variety of preferential voting methods.",
    long_description=long_description,     
    long_description_content_type="text/markdown",
    packages=["pref_voting", "pref_voting.profiles", "pref_voting.voting_methods"],
    license='MIT',
    url='https://github.com/voting-tools/pref_voting',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                     
    python_requires='>=3.6',    
    py_modules=["pref_voting"],              
    install_requires=[]                     
)
