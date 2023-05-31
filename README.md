# IDistill

Official repository for the Unveiling the Two-Faced Truth: Disentangling Morphed Identities for Face Morphing Detectiont paper at [EUSIPCO 2023](http://eusipco2023.org/).

The paper can be viewed at: Soon

## Abstract


## How to run

Example command: 
```bash
python3 code/train.py --train_csv_path="morgan_lma_train.csv" --test_csv_path="morgan_test.csv" --max_epoch=250 --batch_size=16 --latent_size=32 --lr=0.00001 --weight_loss=100
```

## Acknowledgement
The code was extended from the initial code of [SMDD-Synthetic-Face-Morphing-Attack-Detection-Development](https://github.com/naserdamer/SMDD-Synthetic-Face-Morphing-Attack-Detection-Development-dataset) and [OrthoMAD](https://github.com/netopedro/orthomad).

## Citation
If you use our code or data in your research, please cite with:


