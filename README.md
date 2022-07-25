## Install environment
```
conda anv create -f env.yaml
```

## Update environment
```
conda env update -f env.yaml
```

## activate environment
```bash
conda  activate qm
```


## run tests
```bash
invoke test
```

## run tests using testmon, only running tests affected by changes
```bash
invoke testm
```
