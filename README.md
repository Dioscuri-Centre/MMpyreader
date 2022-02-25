# Purpose 

This simple code allows to quickly load and process images acquired from micro-manager
Under the hood it uses pycromanager [https://github.com/micro-manager/pycro-manager] 

WRITE HOW TO INSTALL NECESSARY STUFF

BW: on windows 10 and using miniconda, I had to do this 


1.Download the lastest version of micro-manager 2.0
2.Install pycro-manager using pip install pycromanager
3. Run Micro-Manager, select tools-options, and check the box that says Run server on port 4827 (you only need to do this once)
4. install:   pip install docstring-inheritance
5. then the test code "win64_test.py":

  from pycromanager import Bridge

  with Bridge() as bridge:
	  print(bridge.get_core())
  
gives the following output:

(base) PS C:\Users\bwaclaw\Documents\GitHub\MMpyreader> python .\win64_test.py
<pycromanager.zmq.mmcorej_CMMCore object at 0x000001B2FFAD9888>
