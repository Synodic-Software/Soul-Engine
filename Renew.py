import subprocess
import sys
import os
import importlib

from pathlib import Path



def main():

    #pipenv installation
    try:
        importlib.import_module("pipenv")
    except ImportError:
        import pip
        pip.main(['install', "pipenv"])
    finally:
        globals()["pipenv"] = importlib.import_module("pipenv")

    #TODO: Conditional environment update with no long lock process
    #Update the virtual python enviroment
    print("Updating the virtual python enviroment")

    os.chdir(Path('.') / "Tools" / "Python")

    subprocess.call([sys.executable, "-m", "pipenv", "update"])


    print("Updating the C++ enviroment")

    #Set the conan remote
    subprocess.call([sys.executable, "-m", "pipenv", "run", "conan", "remote", "add", "--force", "bincrafters", "https://api.bintray.com/conan/bincrafters/public-conan"])

    #Create build directory if it does not exist
    if not os.path.exists(Path('.') / ".." / ".." / "Build"):
        os.makedirs(Path('.') / ".." / ".." / "Build")

    #install conan dependancies for Debug and Release
    subprocess.call([sys.executable, "-m", "pipenv", "run", "conan", "install", str(Path('.') / ".." / "Conan"), "-if", str(Path('.') / ".." / ".." / "Build"), "-g", "cmake_multi", "-s", "build_type=Release"])
    subprocess.call([sys.executable, "-m", "pipenv", "run", "conan", "install", str(Path('.') / ".." / "Conan"), "-if", str(Path('.') / ".." / ".." / "Build"), "-g", "cmake_multi", "-s", "build_type=Debug"])

main()