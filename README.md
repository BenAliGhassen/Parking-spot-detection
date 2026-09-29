# Parking Spot Detection

Computer vision project that analyzes a parking lot video and detects, frame by frame, which spots are **free** and which are **occupied**.


## How It Works

1. **Spot definition**: each parking spot is defined once as a bounding box (`<mask image — adjust>`).
2. **Frame processing**: every frame of the video is read and each spot region is cropped.
3. **Classification**: each cropped spot is classified as empty or occupied using `<trained model>`.
4. **Visualization**: spots are drawn on the frame with their status and a live count of available spots.

## Tech Stack

- Python
- OpenCV
- NumPy
- `<scikit-learn /PyTorch>`

## Getting Started

### Prerequisites

- Python `>= 3.9`

### Installation

```bash
git clone <repo-url>
cd parking-spot-detection
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

```bash
python main.py --video data/parking.mp4 --mask data/mask.png
```

| Argument  | Description                          |
|-----------|--------------------------------------|
| `--video` | Path to the input parking video      |
| `--mask`  | Path to the spot mask / coordinates  |
| `--output`| (optional) Path to save the result   |

## Project Structure

```
parking-spot-detection/
├── data/          # Input video and mask
├── models/        # Trained model (if any)
├── main.py        # Entry point
├── utils.py       # Helper functions
└── requirements.txt
```


## Limitations

- Sensitive to lighting changes and shadows
- Fixed camera angle required
- Spots must be defined beforehand

## Future Improvements

- [ ] Automatic spot detection
- [ ] Real-time camera stream support
- [ ] Web dashboard showing availability

## Author

**Ghassen** — `<GitHub / LinkedIn link>`
