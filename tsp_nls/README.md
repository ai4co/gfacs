## GFACS with NLS for TSP

Note that the dataset is already provided in the [`../data/tsp_nls`](../data/tsp_nls) directory.

### Training

The checkpoints will be saved in [`../pretrained/tsp_nls`](../pretrained/tsp_nls).

Train GFACS model for TSP with `$N` nodes
```raw
$ python train.py $N
```


### Testing

Test GFACS for TSP with `$N` nodes
```raw
$ python test.py $N -p "path_to_checkpoint"
```

Test GFACS for TSPlib instances assigned to model trained with `$N` nodes
```raw
$ python test.py -p "
```

Note that we test the model with TSPlib instances assigned to the model trained with the similar number of nodes. For models trained with 200, 500, 1000 nodes, we test the model with TSPlib instances with 100-299, 300-699, 700-1499 nodes, respectively.
