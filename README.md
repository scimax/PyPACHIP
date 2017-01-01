# PyPACHIP
PyPACHIP - Python Proton Accelerator on a Chip Simulation


# Using standalone
In this case, no python version or something similar is required. You can simply download the executable and run it.

# Using as python scripts
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






