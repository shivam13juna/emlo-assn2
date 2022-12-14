# Session 6

#### Without 2 nodes

Tensorboard link to logs with multi gpu training without 2 nodes https://tensorboard.dev/experiment/Y965wCV6SX6yswvqE8cJnw/

s3://test-bucket-emlo-1/s6/without_2_nodes_epoch_007.ckpt

max_batch_size = 20000

#### With 2 nodes

Tensorboard link to logs with gpu training with 2 nodes https://tensorboard.dev/experiment/YXzLURTzSGy7EjmpfGj8nA/

s3://test-bucket-emlo-1/s6/with_2_nodes_epoch_005.ckpt

max_batch_size = 20000

was able to pass same max batch-size

# Session 4

### Docker Image url

shivam13juna/tsai_emlo4

### For running docker-image

docker run -p 8080:8080 shivam13juna/tsai_emlo4

### For building Docker-image

cd dockerize/

docker image build -t torch_script .

## Link to Github REPO

https://github.com/shivam13juna/emlo-assn2.git

# Session 2

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
