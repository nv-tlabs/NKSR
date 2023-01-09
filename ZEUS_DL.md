# Zeus Deep Learning Library

The Zeus library is a set of files that constitute a minimal yet versatile deep learning training/testing system. It is based on `pytorch-lightning` and `pycg`.

Up until now, Zeus has the following files: `zeus.py`, `train.py`, `test.py`, `overfit.py`, `models/base_model.py`, and `configs/default/zeus.yaml`.

## Usage

### Train from scratch

```shell
python train.py <CONFIG> --logger_type <wandb/tb> --wname <NAME>
```

- By default, `logger_type` is `wandb`. If you don't provide `wname`, a random wandb name will be generated.

Other useful flags:
- `--fast_dev_run`: Dry run. Just run one batch. This will not create checkpoints.
- `--profiler`: Value can be `simple` or `advanced`. Will print a log at trainer fit finished.
- `--limit_<train/val/test>_batches`: limit the number of respective batches.
- `--max_epochs`
- `--debug`: If there is **NaN issue**. Please use this flag and manually checkout the stack trace. You cannot get a pdb interactive environment because the previous frames&data are not saved.

### Resume training

For `wandb` (Note that checkpoints need to be on this machine!):
```shell
python train.py wdb:<WDB-URL> --resume
```

For `tensorboard`:
```shell
python train.py <TB-HPARAM>.yaml --resume
```

### Test a model

For `wandb`:
```shell
python test.py none --ckpt <WDB-URL>:<CKPT-ID> -v
```

- `-v` is automatically expanded to `--visualize`.

### Overfit a model

```shell
python overfit.py <CONFIG> \
  --interval 10 \
  --log_dir ../overfit/<LOG-FOLDER> \
  --log_plot -v
```

### Retriving checkpoints from remote

On remote server, run:
```shell
python upload_ckpt.py <WDB-URL>:<CKPT-ID>
```

### Note

- `<WDB-URL>` can be in 2 types:
  - `<USER>/<WANDB-PROJECT>/<WANDB-RUN-ID>`
  - `<USER>/<WANDB-PROJECT>/<WANDB-NAME>`: This is not unique, will prompt if more than 1 candidates exist.
- `<CKPT-ID>`:
  - `last`: last epoch
  - `best/epoch`: best epoch
  - `3`: the 3rd epoch
  - `all`: all epochs

### WANDB Sweep

To run hyperparameter sweep, first write a config file like `sweeps/*.yaml`. Then run on `ws`:
```shell
wandb sweep sweeps/<CONFIG>.yaml
```

Then, on NGC, run:
```shell
CUDA_VISIBLE_DEVICES=... wandb agent <SWEEP-ID> --count 1
```
where `count` specifies how many jobs should this agent run.
Agents will ask for job from the wandb server to run.
It will call our training script with `--exec` set.

Alternatively, the following summarizes the `ngman -> jupyter -> jn -> git pull -> run agent` step chain.
```shell
ngman remote sh --type 32g.7 --sweep <SWEEP-ID>
```
where `SWEEP-ID` is something like `huangjh/nkfw-shapenet/j9bbtcm1`.

## Configuration

Here are some configurations that you can tweak. Please create a file named `zeus_config.yaml` in the same folder as `train.py` to override the ones in `configs/default/zeus.yaml`.

## Profiling

Depending on your needs, many different profiling strategies can be used.

### Pytorch Profiler

The built-in profiler of pytorch is actually very useful! You can use it in the following two ways:

1. Using `pycg.exp`. Annotate your function with `pycg.exp.pt_profile` or code block with `pycg.exp.pt_profile_named(xxx)`. Then run the code with `PT_PROFILE=1/2` set, where 1 will only profile CPU function calls while 2 will also record cuda calls (usually unnecessary).
2. Using `pytorch_lightning` in `tensorboard`. First annotate your code block with `record_function()`, then run the script with a logging dir and flag `--profiler pytorch`. Open tensorboard in the logging dir and the profile tool will appear.

### Memory

First, use the decorator `@exp.mem_profile` from `pycg.exp` to annotate your function (the class decorator does not work very well yet).
Then we follow the 2-step approach:
1. Set environment variable `MEM_PROFILE=1,<TH>`. This will print a brief description.
2. Set environment variable `MEM_PROFILE=2`. This will print a long description.

### Time

If your train/val code is slow and you doubt there is a bottleneck function, then find it out using the following steps:

1. Use the following flags: `--max_epochs 1 --limit_train/val/test_batches 20 --profiler simple` to run the model for a few iterations. At exit the program will print the time used for each pytorch-lightning step. Check which part is the bottleneck, especially mind the data loading part.

2. `py-spy` is able to gives you real-time display of your function, its computation utility (Active%). You can start it by running `sudo env "PATH=$PATH" py-spy top --pid <Your program>`.

3. `kernprof` gives you per-line execution time, and is very useful if you do not have many iterations in your code. In order to use it, add `@pycg.exp.profile` to your function and run `kernprof -l train.py ...`. When program exits, it will tell you something.

4. Pytorch profiler is the ultimate tool of choice, which output a chrome trace for all CPU/GPU functions and has solved many of your problems effectively. Just uncomment the line in `train.py`. The trace json file can be loaded using Chrome's `chrome://tracing`. (Note: Pytorch-lightning also provides pytorch profiler, but the documentation is very vague.)

5. `exp.Timer` class is also useful if you want to be flexible.

### Using Nvidia's Profiler

Nvidia is developing many tools for better performance diagnosis. The old one is `nvvp` visual profiler, and new ones are `Nsight Compute` and `Nsight System`.
The tools are also bundled with CUDA. To debug a program, simply run:
```
sudo -E env PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH /usr/local/cuda-11.6/bin/nsight-sys
```

Note that this has to run as root. Older versions earlier than 2020.5 does not profile memory.
In my experience, it's hard to debug using this tool. For memory issue the best way is still memlab.
Memlab is accurate -- it just ignores some cuda context overhead (~1.6GiB on your machine and ~800MiB on 1080).

## How to use parallel training

Just add
```
--gpus 2
```
to enable `DDP`, or add `--accelerator dp` to use `DP`.

Remember, when using DDP, your dataloader's batch size should be `self.hparams.batch_size // self.trainer.world_size`, to match the original training.

Usually, pytorch tutorial will use something like `mp.spawn()`, which resembles `--accelerator ddp_spawn` and this according to PL, has many limitations. Internally, `ddp` in PL use `subprocess.Popen` instead of `multiprocess`, so you will observe the entire script being run again because `__name__` is not set to `__mp_main__` as in `multiprocess`.

**Another hint**: for `multiprocess`, it has two modes `fork` and `spawn`, both of which will not run `__main__` again, but the latter one will re-import all packages (because it creates a fresh python interpreter instance). Pytorch cuda tensor requires `spawn` to work.

