# Purpose 

This simple code allows to quickly load and process images acquired from micro-manager
Under the hood it uses pycromanager [https://github.com/micro-manager/pycro-manager] 

# How to install 
Clone the repository: 

```bash 
git clone https://github.com/Dioscuri-Centre/MMpyreader.git
```

Then install 
```
cd MMpyreader && pip install .
```

You also need to install pycro-manager (my change has been merged)

```bash 
pip install -r requirements.txt
```

Check that you can import the library with python:
``` python
import mmpyreader
```

# How to run 

There is an introductive notebook on how to use the library: `tutorial.ipynb`

## Mandatory 
`pycro-manager` library acts as a bridge between Micro-Manager and python. Therefore it requires a running micro-manager version in the background and the server mode activated  (cf: https://pycro-manager.readthedocs.io/en/latest/setup.html ).

In summary: 
1 .Download the lastest version of micro-manager 2.0 (https://download.micro-manager.org/nightly/2.0/Windows/)
2. Run Micro-Manager, select tools-options, and check the box that says Run server on port 4827 (you only need to do this once)

If you receive an error about ZMQ like:  

```python
TimeoutError: Socket timed out after 500 milliseconds. Is Micro-Manager running and is the ZMQ server on 4827 option enabled?
```

Which is self-explanatory, you need to start micro-manager (usually someone on the server has it running but if it's not the case, do it.)

Another issue is mentioned in the aforementioned link: if there's a version mismatch, simply follow the instructions.
