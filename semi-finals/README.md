## Introduction

We try two different system.

1. End-to-end system. Split X into 16 groups, and train models with Full Connected Network seperately.
2. Three steps. Denoise $Y_{pilot}$, estimate $\hat{H}$ with pilot. Transfer $\hat{H}$ to $H$ using network, and divided by $Y_{data}$ to get $\hat{X}$. Another network to transfer $\hat{X}$ to $X$.

## Run

Get dataset and put it into folder `data`, and create a new folder `train_cache`.

```shell
data
├── H.bin
├── H_val.bin
├── Pilot_16
├── Pilot_64
├── Y_1.csv
├── Y_2.csv
└── train_cache
```

##### Firstly, we will generate extra dataset for training.

```shell
bash run_shell/generate_data.sh
```

##### Then, train the model,

End-to-End

```shell
bash run_shell/ffn_concat.sh
```

or three steps

```shell
bash run_shell/YHX_linear.sh
```

##### Finally, modify the `generate.sh`, and 

```shell
bash run_shell/generate.sh
```

