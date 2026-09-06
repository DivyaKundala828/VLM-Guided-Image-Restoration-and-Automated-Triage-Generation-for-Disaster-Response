# Disaster Restoration Experiment

## Objective

Evaluate whether the trained U-Net Dense Attention restoration model can restore synthetically degraded disaster images.

## Dataset

DisasterVQA was loaded directly from Hugging Face.

Dataset statistics used in this experiment:

- Total QA pairs: 4,405
- Total unique images: 1,395
- Selected unique images: 199
- QA rows associated with selected images: 623

## Disaster Categories

The selected 199 images contain the following disaster categories:

| Disaster Type | Images |
|---|---:|
| Earthquake | 61 |
| Flood | 41 |
| Hurricane | 38 |
| Fire | 19 |
| Accidents | 10 |
| Wildfire | 8 |
| Landslide | 7 |
| Storm | 7 |
| Other disasters | 6 |
| Other | 2 |

## Synthetic Haze Generation

Synthetic haze was generated on the selected original disaster images.

The original image was retained as the ground-truth reference.

Experiment parameters:

- Severity: Moderate
- Transmission range: 0.55 to 0.748
- Atmospheric light range: 0.85 to 1.0

The parameters are stored in:

`results/haze_parameters.csv`

## Restoration Model

The restoration model is a U-Net Dense Attention architecture with residual learning.

The model was previously trained on the RESIDE dataset.

The trained best model was applied to the synthetically hazy disaster images.

Because some disaster images were high resolution, tiled inference was used:

- Tile size: 320 × 320
- Overlap: 64 pixels

This avoided GPU memory overflow while preserving the original image resolution.

## Restoration Results

The experiment evaluated 199 disaster images.

### Overall Results

| Metric | Hazy | Restored | Improvement |
|---|---:|---:|---:|
| PSNR | 15.0642 dB | 24.0999 dB | +9.0357 dB |
| SSIM | 0.8111 | 0.9277 | +0.1166 |

## Results by Disaster Type

The restoration model improved PSNR and SSIM for every evaluated disaster category.

Detailed results are stored in:

`results/restoration_metrics_by_disaster.csv`

Per-image results are stored in:

`results/restoration_metrics.csv`

## Conclusion

The results indicate that the RESIDE-trained U-Net Dense Attention model can improve the visual quality of synthetically hazy disaster images.

The next stage of the project will evaluate whether this restoration improvement also improves downstream disaster understanding by a Vision-Language Model.

## Next Experiment

The VLM evaluation will compare three image conditions:

1. Original disaster image → VLM
2. Synthetic hazy disaster image → VLM
3. Restored disaster image → VLM

VLM responses will be compared with DisasterVQA ground-truth answers.