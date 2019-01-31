# label_identification

## How to use

### Prepare Data

- mytrain/data/
	- [origin](https://pan.baidu.com/s/1dTEIfnnJ5_Gsru2nszcntw?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/path=%2Fgithub%2Fprivate%2Flabel_identification%2Fdata&parentPath=%2Fgithub/origin.zip) (password: tlt5)
	
### Make XML
```
python do_xml.py
```

### Train Model
```
sh train.sh
```

### predict
'''
python mytrain/shape_predictor.py
'''
 