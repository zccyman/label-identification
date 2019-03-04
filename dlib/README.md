# label_identification

## How to use

### Prepare Data

- mytrain/data/
    - [origin](https://pan.baidu.com/s/1dTEIfnnJ5_Gsru2nszcntw?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/path=%2Fgithub%2Fprivate%2Flabel_identification%2Fdata&parentPath=%2Fgithub/origin.zip) (password: )

### Build 
```
mkdir -p examples/build
cd examples/build
cmake -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 ..
make -j
```

### Make XML
```
cd mytrain
python do_xml.py
```

### Train Model
```
cd mytrain
sh train.sh
```

### predict
```
python mytrain/shape_predictor.py
```
 