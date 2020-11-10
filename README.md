# Py-Vis
A visualisation tool to automate the process of grabbing tensorboard events
data and visualising them.  This allows for faster result analysis in my work.

## Notes
Assumes a single experiment is passed initially to start development.

This also requires the use of `torch.utils.tensorboard.SummaryWriter` to create experiments and running the following command before visualising.

```
tensorboard dev upload --logdir {experiment_path}
```

**TODO:** Implement feature for when multiple experiments are passed.

## Benefits
1. Faster result analysis
2. Less code writting
3. Separate experiments from analysis
4. Allows for more research tim
