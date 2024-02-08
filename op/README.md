## GFACS for OP

### Dataset Generation

Generate the test and validation datasets.
```raw
$ python utils.py
```


### Training

The checkpoints will be saved in [`../pretrained/op`](../pretrained/op).

Train GFACS model for OP with `$N` nodes
```raw
$ python train.py $N
```


### Testing

Test GFACS for OP with `$N` nodes
```raw
$ python test.py $N -p "path_to_checkpoint"
```
