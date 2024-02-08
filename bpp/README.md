## GFACS for BPP

### Dataset Generation

Generate the test and validation datasets.
```raw
$ python utils.py
```


### Training

The checkpoints will be saved in [`../pretrained/bpp`](../pretrained/bpp).

Train GFACS model for BPP with `$N` nodes
```raw
$ python train.py $N
```


### Testing

Test GFACS for BPP with `$N` nodes
```raw
$ python test.py $N -p "path_to_checkpoint"
```
