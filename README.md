## Installation
1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download weight files:
  Run the command below to download weights to the `weights` directory (Linux):

  ```bash
  # Download specific model weights
  sh download.sh [model_name]
  # Available models: resnet18, resnet34, resnet50, mobilenetv2, mobileone_s0

  # Example:
  sh download.sh resnet50
  ```

3. Inference:
```
python inference.py --model resnet50 --weight weights/resnet50.pt --view --source 0 --calibrate --fullscreen-screen
```
