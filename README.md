# Purpose 

This simple code allows to quickly load and process images acquired from micro-manager
Under the hood it uses pycromanager [https://github.com/micro-manager/pycro-manager] 

# How to install 

If you have never installed the package, install with pip: 
```bash 
 pip install  git+https://github.com/Dioscuri-Centre/MMpyreader
```

If you have already installed, to be on the safe side you can force to reinstall all related packages: 
```bash 
 pip install --force-reinstall  git+https://github.com/Dioscuri-Centre/MMpyreader
```

Check that you can import the library with python:
``` python
import mmpyreader
```


## Mandatory 
The code uses `pycro-manager` library which acts as a bridge between Micro-Manager and python. Therefore it requires a running micro-manager version in the background and the server mode activated  (cf: https://pycro-manager.readthedocs.io/en/latest/setup.html).

In summary: 

1. Download the lastest version of micro-manager 2.0 (https://download.micro-manager.org/nightly/2.0/Windows/)
2. You need to start micro-manager and activate the "server". For that purpose go into Tools -> Options and tick the Run server on port XXXX

**IMPORTANT: Note the port number `XXXX` for later uses**

If you receive an error about ZMQ like:  
```
TimeoutError: Socket timed out after 500 milliseconds. Is Micro-Manager running and is the ZMQ server on YYYY option enabled?
```
Which is self-explanatory, you either: 
- need to start micro-manager:
- Are not using the correct port number, check that `YYYY` is not different than `XXXX`. It should be automatically set but if not, you can manually change it by doing so:
    ``` python 
    from mmpyreader import MMpyreader
    MMpyreader(port=XXXX)
    ```
 
Another issue is mentioned in the aforementioned link: if there's a version mismatch, simply follow the instructions.


# How to use 

There is an introductive notebook on how to use the library: `tutorial.ipynb`
