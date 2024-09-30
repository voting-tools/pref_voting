from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

import argparse

init_file = './pref_voting/__init__.py'
pyproject_file = './pyproject.toml'
conf_file = './docs/source/conf.py'

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)


# Instantiate the parser
parser = argparse.ArgumentParser(description='Update version of pref_voting')

parser.add_argument('--version', type=str)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()

check = args.check

if check:
    versions = list()
    print("Checking version\n")
    print("__init__.py: ")
    with open(init_file) as old_file:
        for line in old_file:
            print(line)
            versions.append(line.split("=")[1].strip().replace("'", ""))
    print("conf.py: ")
    with open(conf_file) as old_file:
        for line in old_file:
            if "release" in line:
                print(line)
                versions.append(line.split("=")[1].strip().replace("'", ""))

    print("pyproject.toml: ")
    with open(pyproject_file) as old_file:
        for line in old_file:
            if "version" in line:
                print(line)
                versions.append(line.split("=")[1].strip().replace('"',''))
    if len(list(set(versions))) > 1: 
        print("ERROR: Multiple versions found!!")
    print("Versions: ", list(set(versions)))
else:
    new_version = args.version


    print(new_version)
    with open(init_file) as old_file:
        for line in old_file:
            old_version = line.split("=")[1].strip().replace("'", "")

    print("The current version is: ", old_version)
    print("The new version is: ", new_version)

    if old_version == new_version:
        print("Versions are the same, exiting")
        exit()
    replace(pyproject_file, f'version = "{old_version}"', f'version = "{new_version}"')
    print("Updated pyproject.toml")
    replace(conf_file, f"release = '{old_version}'", f"release = '{new_version}'")
    print("Updated conf.py")
    replace(init_file, f"__version__ = '{old_version}'", f"__version__ = '{new_version}'")
    print("Updated __init__.py")
