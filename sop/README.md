## GFACS for SOP

### Dataset Generation

Generate the test and validation datasets.
```raw
$ python utils.py
```


### Training

The checkpoints will be saved in [`../pretrained/sop`](../pretrained/sop).

Train GFACS model for SOP with `$N` nodes
```raw
$ python train.py $N
```


### Testing

Test GFACS for SOP with `$N` nodes
```raw
$ python test.py $N -p "path_to_checkpoint"
```
