## GFACS for PCTSP

### Dataset Generation

Generate the test and validation datasets.
```raw
$ python utils.py
```


### Training

The checkpoints will be saved in [`../pretrained/pctsp`](../pretrained/pctsp).

Train GFACS model for PCTSP with `$N` nodes
```raw
$ python train.py $N
```


### Testing

Test GFACS for PCTSP with `$N` nodes
```raw
$ python test.py $N -p "path_to_checkpoint"
```
