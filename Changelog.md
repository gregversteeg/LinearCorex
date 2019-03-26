## March 25 - 2019

Changes in vis_corex.py

1. codebase Python 2 to Python 3 migration `

2. DeprecationWarning: 'U' mode is deprecated
  
    
    * Before: `with open(filename, 'rU') as csvfile:`  
    * After: `with open(filename, 'r') as csvfile:`


Changes in readme.md

3. added `graphviz` installation instructions for Ubuntu users

4. added `requirements.txt` which lets users to install all the dependencies with a single command `conda install --yes --file requirements.txt`
