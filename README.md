# ADMS — AI Driver Monitoring System

Real-time detection of five unsafe driver behaviours from a standard webcam, running fully offline on a consumer laptop GPU.

> Final-year major project — BEng (Hons) Computer Systems Design Engineering, Middlesex University London (2026).
> Full development write-up: **[aidms2026.hashnode.dev](https://aidms2026.hashnode.dev)**

---

## The problem

Driver distraction and fatigue are leading contributors to road collisions, but most commercial driver-monitoring systems are tied to specific vehicles and stream footage off-device. ADMS was built to test whether a useful monitoring system could run on commodity hardware, from an ordinary webcam, with **no footage ever leaving the machine**.

## What it does

The system watches a live camera feed and detects five behaviours:

| Class | Meaning |
|---|---|
| `PhoneUse` | Driver handling or looking at a phone |
| `Drowsy` | Eye closure / head-drop indicators of fatigue |
| `Distracted` | Attention directed away from the road |
| `Drinking` | Driver drinking while at the wheel |
| `Seatbelt` | Seatbelt detection |

Raw detections pass through two custom post-processing stages before anything is reported to the driver.

## Results

Final model — YOLOv26n trained on the AI_DMSv4 dataset.

| Metric | Value |
|---|---|
| mAP@0.5 | **0.900** |
| Precision | 0.906 |
| Recall | 0.813 |
| Live inference speed | 24–30 FPS |
| Hardware | NVIDIA RTX 4070 Laptop GPU |

**Per-class Average Precision @ IoU 0.5**

| Class | AP |
|---|---|
| PhoneUse | 0.969 |
| Drowsy | 0.945 |
| Drinking | 0.930 |
| Distracted | 0.834 |
| Seatbelt | 0.820 |

---

## How it works

```
Webcam frame
     │
     ▼
┌──────────────────────────┐
│  YOLOv26n detector       │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Per-class threshold      │  0.15 (Distracted) → 0.40 (Seatbelt)
│  filtering                │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  DetectionSmoother        │  3-of-5 frame sliding window
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  RiskScoreTracker         │  0–10 score with time decay
└──────────┬───────────────┘
           ▼
   On-screen alerts + audio + CSV event log
```

**Per-class confidence thresholds.** A single flat threshold across all five classes performed poorly, because the F1–confidence analysis showed each class peaks at a different confidence value — Drinking in particular needed a lower threshold to reach usable recall. Thresholds were therefore set empirically per class from the F1–confidence curves, ranging from 0.15 for Distracted to 0.40 for Seatbelt.

**DetectionSmoother.** Raw frame-by-frame YOLO output is noisy: a head turn or lighting change can drop a detection for a single frame while the behaviour is still happening, and transient false positives can appear and vanish just as quickly. A class is therefore only treated as active once it has been detected in at least 3 of the last 5 frames, with each class tracked in its own independent window. At 24–30 FPS a five-frame window is roughly 200 ms, and the worst-case added delay is about 80 ms — well below human reaction time, so the system stays responsive while ignoring single-frame noise.

**RiskScoreTracker.** Confirmed detections feed a single 0–10 risk score that rises with dangerous behaviour and decays over time when driving is safe. Alerts fire on the score rather than on individual detections, so a brief glance away is treated differently from sustained phone use.

## What I'd want a reader to know about the results

**v4 was not an improvement across the board, and it's worth being precise about that.** Moving from the AI_DMSv3 dataset to AI_DMSv4 raised overall mAP@0.5 from 0.883 to 0.900, precision from 0.831 to 0.906 and recall from 0.743 to 0.813. But two classes went backwards: Drowsy fell from 0.974 to 0.945 and Seatbelt from 0.858 to 0.820. Seatbelt also still had 27% of its instances predicted as background.

The likely cause is class imbalance — PhoneUse had roughly 2.5× the instances of Seatbelt, and Seatbelt was the weakest class in both training runs. The aggregate metric improved while the two classes I'd most want to be reliable did not, which is exactly the case for not judging a model on a single headline number.

**A note on architecture.** AI_DMSv3 and AI_DMSv4 are *dataset* versions, not models. Both were trained on the same YOLOv26n architecture. A brief comparison run against YOLOv26s showed only a small mAP gain while running noticeably slower, so the nano variant was kept for real-time performance. The contribution here is the dataset engineering and the analysis of how composition changed per-class behaviour, plus the full real-time pipeline — not a novel architecture.

## Limitations

- Training used a dataset assembled largely from controlled or semi-controlled settings rather than real driving, so real-world performance is likely lower than the reported metrics.
- Live testing used a webcam indoors plus AI-generated driving footage (Pixverse, Seedance), since filming genuinely dangerous driving was neither practical nor ethical. AI-generated footage is cleaner and more consistent than real driving video, which probably flatters the results.
- Not validated in a moving vehicle over extended real-world driving.
- `Seatbelt` remains the weakest class (AP 0.820, 27% predicted as background), driven by class imbalance and occlusion.
- The dataset is not demographically balanced, so per-group performance is unverified.
- Real-time performance depends on a CUDA GPU. An RTX 4070 Laptop GPU sustains 24–30 FPS; a Raspberry Pi 4 alone could not, without quantisation or a hardware accelerator.

---

## Running it

```bash
git clone https://github.com/FawzyDev/adms-driver-monitoring.git
cd adms-driver-monitoring
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# live webcam
python src/run_live.py --weights weights/best.pt

# on a video file
python src/run_live.py --weights weights/best.pt --source path/to/video.mp4
```

Requires Python 3.10+. Real-time performance needs a CUDA-capable GPU; it will run on CPU at a reduced frame rate.

## Repository layout

```
src/            inference app, DetectionSmoother, RiskScoreTracker, UI, logging
training/       training and resume-training scripts
weights/        trained model weights (best.pt)
docs/           evaluation results, demo media, diagrams
requirements.txt
```

## Dataset

Not included in this repository, due to size and the licensing of the source datasets.

**AI_DMSv4** was assembled in Roboflow from several publicly available driver-behaviour datasets on Roboflow Universe, merged and then supplemented with additional images sourced and labelled by hand to fill gaps in the weaker classes. All labels were manually reviewed. Train/validation/test splits were produced with Roboflow's built-in split function at 80/10/10, which prevents the same image appearing across splits.

- Source images: **37,760**
- After 3× augmentation: **~99,000**
- Augmentation: mosaic (p=1.0) and mixup (p=0.1) during training, plus Roboflow-side augmentation

**Why two dataset versions.** AI_DMSv3 was the baseline. AI_DMSv4 removed the Eating and Smoking classes, which were causing too many misclassifications, and added images to the weaker remaining classes. Keeping both allowed a direct before-and-after measurement of how dataset composition affected per-class performance.

**Training configuration.** 180 epochs with patience 75, batch size 16 (higher caused CUDA OOM), initial learning rate 0.015 with cosine decay. The v4 run early-stopped at 47 epochs.

## Built with

Python · Ultralytics YOLO (YOLOv26n) · OpenCV · NumPy · PyTorch · Roboflow · CUDA

## Author

**Mohamed Dawoud** — BEng (Hons) Computer Systems Design Engineering, Middlesex University London
mohamedfawzyuk@gmail.com · [LinkedIn](https://www.linkedin.com/in/mohamed-f-dawoud/) · [Blog series](https://aidms2026.hashnode.dev)

## Licence

MIT — see [LICENSE](LICENSE).
