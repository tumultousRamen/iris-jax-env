
Add project directory to PATH:

For mac
```shell
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

and run example
```angular2html
 mjpython ./examples/trifinger/test_env.py
```


For Linux (not tested yet)
```shell
export PYTHONPATH=/path/to/this/repo:$PYTHONPATH
```

and run example
```angular2html
python ./examples/trifinger/test_env.py
```

### Allegro hand env
For Allegro hand manipulation environment. There are around 20 objects, you can choose from.
See folder ```./envs/xmls/.```. To use one object environment, specify model path as ```env_allegro_****.xml``` 
where ```***``` is the object name. See the example.

### TriFinger env
For TriFinger manipulation environment. There are around 20 objects, you can choose from.
See folder ```./envs/xmls/.```. To use one object environment, specify model path as ```env_trifinger_****.xml``` 
where ```***``` is the object name. See the example.
