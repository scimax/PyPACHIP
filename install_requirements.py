# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:34:10 2016

@author: Max
"""

import pip


if __name__ == "__main__":
    # pip install -U --pre -f https://wxpython.org/Phoenix/snapshot-builds/ \wxPython_Phoenix
    pip.main(["install", "-U", "--pre", "-f",
              r"https://wxpython.org/Phoenix/snapshot-builds/",
              r"wxPython_Phoenix"])
              
    # before :  (3.0.3.dev2467+0a17c5d)
    pip.main(["install", "-r", "requirements.txt"])
    