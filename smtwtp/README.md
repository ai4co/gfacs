## GFACS for SMTWTP

### Dataset Generation

Generate the test and validation datasets.
```raw
$ python utils.py
```


### Training

The checkpoints will be saved in [`../pretrained/smtwtp`](../pretrained/smtwtp).

Train GFACS model for SMTWTP with `$N` nodes
```raw
$ python train.py $N
```


### Testing

Test GFACS for SMTWTP with `$N` nodes
```raw
$ python test.py $N -p "path_to_checkpoint"
```
