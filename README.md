# PyPACHIP
PyPACHIP - Python Proton Accelerator on a Chip Simulation

# Using standalone
At the moment it is only provided for Windows OS. No python installation is required to run the executable. You can download _pypachip.exe_ from the [release page](https://github.com/scimax/PyPACHIP/releases/tag/v1.0). 

# Developer installation guide
When running the scripts directly, the following is required:
- Python 3.5
- Packages:
  - numpy
  - scipy
  - matplotlib
  - wxpython phoenix

The first three packages can be installed via
```bash
pip install -r requirements.txt
```

The last one is not available at the python package index yet since it's still under development. Information about the project can be found [here](https://wiki.wxpython.org/ProjectPhoenix) as well as the [snapshot-builds](https://wxpython.org/Phoenix/snapshot-builds/).

To install the snapshot simply use

```bash
pip install -U --pre \
        -f https://wxpython.org/Phoenix/snapshot-builds/ \
        wxPython_Phoenix
```

After downloading the files from the *src* directory, run
```bash
python main.py
```

# Build instructions
In order to build the executable using *pyinstaller*, the following errors have to be solved:
- http://stackoverflow.com/questions/35478526/pyinstaller-numpy-intel-mkl-fatal-error-cannot-load-mkl-intel-thread-dll
- https://github.com/pyinstaller/pyinstaller/issues/1566

As the second link mentions, Visual Studio 2015 was used to implement the dll files required for *numpy*.

 






