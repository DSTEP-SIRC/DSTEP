# DSTEP
Dynamic Spatio-TEmpotal Pruning 

## Reproducing the results:

### Dependencies
```
pip install -r requierments.txt
```


### Training
run main_dynstc.py as the following (set hyperparameters if needed):
```
python main_dynstc.py somethingv2 RGB --arch res18_dynstc_net --num_segments 8 --lr 0.01 --lr_step 20 40 --epochs 50 --wd 5e-4 --npb --ada_reso_skip --init_tau 0.67 --gate_history --shift --gbn --grelu --gsmx --gate_hidden_dim 1024 --batch-size 32 --workers 8 --gpus 0 --spatial_masking --den_target 0.5 --exp_header reproduce_train 
```


### Inference
Load the models from [this link](https://drive.google.com/drive/folders/1Qgj2mjQ2TjTm2MbdO1LYj9o6fiZlnGdm).
There are 4 checkpoints:
  - 3 ckpt from the same setup with different dense target (high, medium, low).
  - Another ckpt, from table 3.

To use them, run the same command as training and add "test from **Path_to_ckpt**"
