Debugging
=========

Debugging Buda (C++) while running PyBuda
*****************************************

- Build `python_api` in debug mode: `CONFIG=debug make python_api`
- In Code, the following debug configuration works:

.. code-block:: json

   {
        "name": "python gdb debug",
        "type": "cppdbg",
        "request": "launch",            
        "program": "${workspaceFolder}/build/python_env/bin/python",
        "args": ["foo.py"], 
        "stopAtEntry": false,
        "cwd": "${workspaceFolder}",
        "environment": [],
        "externalConsole": false,
        "MIMode": "gdb",
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": true
            },
            {
                "text": "catch throw"
            }
        ],
        "miDebuggerPath": "/usr/bin/gdb"
   }

