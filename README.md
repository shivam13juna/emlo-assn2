## Building Image

make build

    which triggers this command`docker build -t session2 .`

## For Training

1. Set the timm model name in `configs/model/cifar.yaml` default is resnet18
2. following docker command will trigger training

```
docker run --mount type=bind,source=`pwd`,target=/src/ session2 python3 src/train.py experiment=cifar
```

For trying out different metrics, can specify in callbacks, `configs/callbacks/model_checkpoint.yaml`

## For Prediction (Optional)

1. Copy the location of best checkpoint from training into predict.yaml
2. Paste the location of image(which one wants to run prediction on) in the predict.yaml and output will be the prediction class

```
docker run --mount type=bind,source=`pwd`,target=/src/ session2 python3 src/predict_v1.py
```

## For Eval

1. Copy the location of best checkpoint from training into eval.yaml

```
docker run --shm-size 25G --mount type=bind,source=`pwd`,target=/src/ session2 python3 src/eval.py
```

--shm-size for increasing shared memory size for containers, was running OOM earlier.

## For COG inference

1. in `src/predict.py` appropriate timm model needs to be specified (with which cifar model is trained), and path in which checkpoints are saved have to be specified. Corresponding state-dict are loaded into models, which are used in inferencing.
2. Output is the prediction class

```
cog predict -i image=@tmp/dog.jpg
```
