# Pancreas Lesion Segmentation

### Commands used:
`nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity` to setup preproceess directory

`nnUNetv2_train Dataset001_M31Pancreas 3d_fullres 1 -device 'cpu' -num_gpus 0`

`nnUNetv2_plan_and_preprocess -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)` - Used `nnUNetPlannerResEncM` as required by the task

Taken from docs: "Now, just specify the correct plans when running nnUNetv2_train, nnUNetv2_predict etc. The interface is consistent across all nnU-Net commands: -p nnUNetResEncUNet(M/L/XL)Plans"