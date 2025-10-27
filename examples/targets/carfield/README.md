# Carfield Example

To install MATCHA and dependencies you need to follow the same procedure as [MATCH](/usr/scratch/larain11/enrusso/conda_envs/match).

To run MATCHA for Carfield and ResNet8 you can run:

```sh
python run.py -i model_fp16/cifar10_resnet8_fp16/input.txt -m model_fp16/cifar10_resnet8_fp16/model_fp16_nchw.onnx -o output
```

This should generate the C code in the `output` folder.

Before compiling you need to:

```sh
export CAR_ROOT=path/to/carfield
```

You can get the carfield repo with `git clone https://github.com/pulp-platform/carfield`. 

**Tested carfield commit is `7edf3b51aaff66cbb235c6cacada3f8a2fb1bc5e`.** But we considered 8 pulp cluster cores with RedMule HWPE only, please apply `carfield.patch`. The HW config we used is a modified `carfield_l2dual_pulp_periph.sv` (with Spatz activated)

Make sure to compile carfield software.

Now, you can compile MATCHA output:

```sh
cd output; make build-host
```

This should generate an `host.elf` that you can load and run.

