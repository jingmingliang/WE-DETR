
**(TGRS 2026) WE-DETR: Wavelet-Enhanced Multi-Scale Feature Compensation Detection Transformer for Remote Sensing Small Objects**

Implementation of WE-DETR for remote sensing small-object detection, published in *IEEE Transactions on Geoscience and Remote Sensing*.

[Paper](https://doi.org/10.1109/TGRS.2026.3730195)

Run all commands from the repository root.

## Data Preparation

Prepare the training and validation datasets with **COCO-format annotations**. For VisDrone, VOC, or custom datasets, convert the original annotations to COCO format first.

Update the dataset paths and category settings in `configs/dataset/coco_detection.yml`. For example, for a dataset with ten foreground categories:

```yaml
num_classes: 10
remap_mscoco_category: False

train_dataloader:
  dataset:
    img_folder: ./dataset/visdrone/train/images/
    ann_file: ./dataset/visdrone/annotations/train.json

val_dataloader:
  dataset:
    img_folder: ./dataset/visdrone/val/images/
    ann_file: ./dataset/visdrone/annotations/val.json
```

With `remap_mscoco_category: False`, category IDs must be contiguous, starting from `0` and ending at `num_classes - 1`. Use COCO bounding boxes in `[x, y, width, height]` format and include an `area` value for each annotation.

For MS COCO, keep `num_classes: 80` and `remap_mscoco_category: True`, and set the corresponding image and annotation paths.

## Training and Evaluation

### Training

Single-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rtdetr/wedetr.yml \
  --seed 42
```

Multi-GPU training on a single machine:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 tools/train.py \
  -c configs/rtdetr/wedetr.yml \
  --seed 42
```

Add `--amp` to enable mixed-precision training. Adjust batch sizes in `configs/rtdetr/include/dataloader.yml` and optimization settings in `configs/rtdetr/include/optimizer.yml`. Batch size is specified per process.

Checkpoints and logs are saved to `./output/wedetr/` by default.

### Resume Training

```bash
python tools/train.py \
  -c configs/rtdetr/wedetr.yml \
  -r output/wedetr/checkpoint.pth
```

### Evaluation

```bash
python tools/train.py \
  -c configs/rtdetr/wedetr.yml \
  -r output/wedetr/checkpoint.pth \
  --test-only
```

Evaluation uses the validation dataset specified in the configuration and reports COCO bounding-box AP and AR metrics.

## Whole-Image Inference

```bash
python tools/infer.py \
  -c configs/rtdetr/wedetr.yml \
  -r output/wedetr/checkpoint.pth \
  -f path/to/image.jpg \
  -d cuda:0
```

Use a checkpoint matching the model configuration and number of classes. The input is resized to 640 × 640, and predicted boxes are mapped back to the original image. The visualization is saved as `results_0.jpg` in the current directory. Use `-d cpu` for CPU inference.

## Sliced Inference

For large remote sensing images, enable overlapping sliced inference:

```bash
python tools/infer.py \
  -c configs/rtdetr/wedetr.yml \
  -r output/wedetr/checkpoint.pth \
  -f path/to/large_image.jpg \
  -d cuda:0 \
  -s True \
  -nc 25
```

`-nc` controls the approximate slicing grid; the actual number of crops depends on image dimensions and overlap. The current overlap ratio is `0.2`. Predictions are mapped back to the original image and overlapping same-class boxes are merged. The visualization is saved as `results_0.jpg`.

Omit `-s` for whole-image inference. The current visualization threshold is `0.6`, set in `tools/infer.py`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{song2026wedetr,
  author  = {Song, Yu and others},
  title   = {{WE-DETR}: Wavelet-Enhanced Multiscale Feature Compensation Detection Transformer for Remote Sensing Small Objects},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {64},
  year    = {2026},
  doi     = {10.1109/TGRS.2026.3730195}
}
```

## Acknowledgments

This project builds on [RT-DETR](https://github.com/lyuwenyu/RT-DETR). We thank the authors for their open-source implementation.
