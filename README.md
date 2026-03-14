# Computer Vision Tracking Pipeline

This repository contains a modular and comprehensive computer vision pipeline of ALTEK UAV team for object detection and target tracking aiming to perform missions in TEKNOFEST Combat UAV Contest.
The main runtime flow lives under `pipeline/`, while tracker-specific research implementations live under `tracking_implementations/`.

## Project Structure

- `pipeline/`
    - Main applications and runtime logic
    - `main.py`: entry point for running the pipeline
    - `main/pipeline.py`: state machine (`SEARCHING` / `TRACKING`) and tracking flow
    - `modules/`: detector and tracker adapters
    - `models/`: runtime model files used by the pipeline
    - `params/`: tracker and runtime parameter files
    - `videos/`: input videos
- `tracking_implementations/`
    - `AVTrack/`
    - `ORTrack/`
    - `MixFormerV2/`
- `yolo_engine/`
    - ONNX-related model conversion/inference utilities for YOLO
- `files/`, `papers/`, `videos/`
    - Supplementary assets and documents

## Environment Setup

Using a specific conda environment is recommended.

```bash
conda create -n avtrack_env python=3.8 -y
conda activate avtrack_env
```

Install PyTorch (CUDA 11.7 example as recommended version):

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install AVTrack dependencies:

```bash
pip install -r tracking_implementations/AVTrack/requirements.txt
```

If you plan to run ORTrack or MixFormerV2, also install dependencies from their own `requirements` files and README instructions.

## Required Assets

- Download AVTrack model weights from the shared drive used by the project team or using its documentaton.
- Place required tracker models in `pipeline/models/` (or the model path expected by your selected tracker config).
- Place test videos under `pipeline/videos/`.

> The repository may not include large model/video files by default.

## Running the Pipeline

Before running, verify model and video paths in:

- `pipeline/main.py`
- `pipeline/params/tracker/av_track_params.py` (considering current tracker is AVTrack)

Then run:

```bash
cd pipeline
python main.py
```

## Tracker-Specific Notes

For tracker implementations under `tracking_implementations/`, initialize local path configs when required.

### AVTrack

```bash
cd tracking_implementations/AVTrack
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

### ORTrack

```bash
cd tracking_implementations/ORTrack
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

### MixFormerV2

```bash
cd tracking_implementations/MixFormerV2
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

## Export and Engine Utilities (AVTrack, others will be coming soon)

- ONNX exporter: `tracking_implementations/AVTrack/avtrack_onnx_exporter.py`
- TensorRT engine builder: `tracking_implementations/AVTrack/engine_builder.py`

Example engine build:

```bash
cd tracking_implementations/AVTrack
python engine_builder.py --onnx output/onnx/avtrack.onnx --engine output/avtrack_fp16.engine --fp16 --workspace-gb 1.0
```

## Troubleshooting

- If path-related errors appear (`local.py`, `prj_dir`, dataset root), regenerate local config files using the commands above.
- If imports such as `torch`, `onnxruntime`, or `tensorrt` are unresolved, verify the active environment and package installation.
- If model files are missing, confirm paths and filenames in both pipeline config and tracker config files.

## License

This repository contains multiple subprojects that may use different licenses.
Please check license files under each tracker directory (for example, `tracking_implementations/ORTrack/LICENSE`).
