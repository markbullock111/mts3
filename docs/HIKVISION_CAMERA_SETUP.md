# Hikvision IP Camera Setup for Offline Attendance System (Windows, RTSP)

## Purpose

This guide describes the correct way to integrate a Hikvision IP camera with a **Windows-based offline attendance system** where the **local Windows PC** runs both:

- Real-time inference (person detection, tracking, face recognition)
- Backend/API + attendance database

This system must run **fully offline** after installation and model download. For camera ingestion, use **direct RTSP streaming** from the Hikvision camera (or a local webcam fallback).

## Scope and Assumptions

- OS: Windows 10/11
- Camera: Hikvision IP camera (single entrance)
- Runtime: OpenCV-based capture pipeline (Python)
- Use case: person tracking + face recognition at an entrance
- Network: camera and Windows PC on the same LAN (no internet required)

---

## 1) Why NOT to Use Hikvision HTTP API for Video

### Do not use Hikvision HTTP snapshot endpoints for live attendance inference

Hikvision cameras expose HTTP endpoints (for example image snapshots), but those are **not appropriate for real-time video analytics**.

### Reasons

#### 1. Snapshot APIs are request/response, not streaming
- HTTP snapshot endpoints typically return **single JPEG frames**.
- Real-time attendance requires **continuous video frames** with stable timing.
- Polling snapshots introduces uneven intervals and frame jitter.

#### 2. Higher latency and unstable timing
- Repeated HTTP requests add overhead per frame (TCP/HTTP request processing, auth handling, response parsing).
- This increases latency and causes irregular frame cadence.
- Tracking quality degrades when frame timing is inconsistent.

#### 3. Poor tracking quality (ByteTrack / IoU tracking)
- Person tracking relies on temporal continuity.
- Snapshot polling can skip motion transitions (especially at entrances), causing:
  - track ID switches
  - missed line crossing events
  - duplicate check-ins

#### 4. Lower reliability under load
- HTTP polling creates repeated request bursts and may stress the camera web service.
- RTSP is designed for continuous media transport and is more reliable for video pipelines.

#### 5. Harder to control transport and buffering
- RTSP allows better control over streaming transport (TCP/UDP) and player/decoder buffering.
- HTTP snapshot polling provides much less control for low-latency tuning.

### Correct approach

Use **direct RTSP streaming** from Hikvision into OpenCV / FFmpeg / VLC.

---

## 2) Hikvision RTSP URL Format (Required)

Use the Hikvision RTSP URL format below.

### Main stream (recommended for higher quality face recognition when bandwidth/CPU allows)
```text
rtsp://username:password@IP:554/Streaming/Channels/101
```

### Sub stream (recommended for lower latency / lower bandwidth)
```text
rtsp://username:password@IP:554/Streaming/Channels/102
```

### Examples
```text
rtsp://admin:StrongPass123@192.168.1.64:554/Streaming/Channels/101
rtsp://admin:StrongPass123@192.168.1.64:554/Streaming/Channels/102
```

### Notes
- Replace:
  - `username` with the camera user (for example `admin`)
  - `password` with the camera password
  - `IP` with the camera LAN IP (for example `192.168.1.64`)
- Default RTSP port is typically `554`
- If your camera was configured with a different RTSP port, replace `554`

---

## 3) Hikvision Channel Mapping Table (101 / 102 / 201 / 202)

Hikvision uses a `channel + stream` numbering convention in `/Streaming/Channels/<id>`.

### Channel / Stream Mapping

| RTSP Channel ID | Physical Camera Channel | Stream Type | Typical Use |
|---|---:|---|---|
| `101` | 1 | Main stream | Higher quality analytics / recording |
| `102` | 1 | Sub stream | Lower latency real-time inference |
| `201` | 2 | Main stream | Multi-channel devices / NVR channel 2 |
| `202` | 2 | Sub stream | Multi-channel devices / NVR channel 2 |

### Interpretation rule
- First digit(s): physical channel number (`1`, `2`, ...)
- Last two digits:
  - `01` = main stream
  - `02` = sub stream

### Single-camera deployments
For a single entrance camera, most deployments use:
- `101` (main)
- `102` (sub)

---

## 4) How to Enable RTSP in Hikvision Web Interface

Hikvision UI labels vary by firmware version, but the workflow is usually similar.

### Typical steps (Hikvision web UI)

1. Open the camera web UI in a browser on the LAN:
   - `http://<camera-ip>` or `https://<camera-ip>`
2. Log in with an administrator account.
3. Go to the RTSP configuration page.

### Common Hikvision UI paths (firmware-dependent)

- `Configuration -> Network -> Advanced Settings -> Integration Protocol`
- `Configuration -> Network -> Advanced Settings -> RTSP`
- `Configuration -> Network -> Platform Access / Integration`

### What to enable/check

- **Enable RTSP Authentication** (or ensure RTSP is enabled and user has permission)
- Confirm **RTSP port** (default `554`)
- Ensure the user account used in the RTSP URL is allowed to access live view

### If RTSP is disabled by policy/firmware
- Enable the relevant integration/streaming protocol in the camera settings
- Save settings and reboot camera if required

### Verify after enabling
- Test in VLC first (see section 6)
- Then test in your Python/OpenCV pipeline

---

## 5) Recommended Hikvision Video Settings for AI (Face + Tracking)

For entrance analytics, optimize for **stable decoding + low latency + face clarity**.

## Recommended settings (per stream)

### Video Encoding
- **Codec: H.264** (recommended)
- Avoid H.265 for real-time AI unless you have a strong reason and verified low-latency decode

### Resolution
Use one of these:
- **1280x720 (recommended starting point)**
- **640x480** (CPU-friendly, lower bandwidth, faster inference)

### Frame Rate
- **20-25 FPS** (recommended)
- 15 FPS can work, but tracking becomes less stable for fast movement/bicycles

### Bitrate
- **Moderate bitrate** (avoid extremely low bitrate that causes compression artifacts)
- Typical starting range:
  - `1.5-3 Mbps` for `1280x720 H.264`
  - `0.5-1.5 Mbps` for `640x480 H.264`

### GOP / I-frame Interval (if available)
- Set reasonably low for responsiveness (for example `1-2 seconds` worth of frames)
- Example:
  - 20 FPS -> I-frame interval `20-40`

### Smart Codec
- **Smart codec: OFF** (important)

Why:
- Smart codec / H.264+ / H.265+ can introduce variable compression behavior and latency
- May degrade face detail in motion and reduce decode consistency

### Recommended stream strategy
- **Main stream (`101`)**: higher quality reference / better face detail if GPU available
- **Sub stream (`102`)**: lower latency inference or CPU mode

---

## 6) Test RTSP Stream Using VLC (Windows)

Before integrating with OpenCV, validate the stream in VLC.

### Steps

1. Install VLC Media Player on Windows.
2. Open VLC.
3. Go to:
   - `Media -> Open Network Stream...`
4. Paste the RTSP URL, for example:
```text
rtsp://admin:StrongPass123@192.168.1.64:554/Streaming/Channels/102
```
5. Click `Play`.

### What to confirm
- Video opens successfully
- No frequent freezes / reconnects
- Acceptable latency for entrance usage
- Face visibility is sufficient under real lighting

### If VLC fails
- Verify username/password
- Verify RTSP is enabled in camera UI
- Verify camera IP and port
- Verify Windows Firewall / LAN ACLs
- Try `101` and `102`
- Try TCP transport in VLC preferences if UDP is unstable

---

## 7) OpenCV RTSP Example (Windows) Using `CAP_FFMPEG`

Use OpenCV with FFmpeg backend for better RTSP compatibility than generic defaults.

### Python Example (RTSP, Windows)
```python
import cv2
import time

RTSP_URL = "rtsp://admin:StrongPass123@192.168.1.64:554/Streaming/Channels/102"

# Optional: force RTSP transport via FFmpeg (before VideoCapture)
# Set to tcp for reliability, udp for potentially lower latency on clean LANs
# import os
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000|stimeout;5000000"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream")

# Try reducing internal buffer (works depending on backend/build)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev = time.time()
frame_count = 0

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("Read failed, retrying...")
        time.sleep(0.1)
        continue

    frame_count += 1
    now = time.time()
    if now - prev >= 1.0:
        print(f"FPS(read): {frame_count}")
        frame_count = 0
        prev = now

    cv2.imshow("Hikvision RTSP", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
```

### Notes for Windows
- `cv2.CAP_FFMPEG` is preferred for RTSP streams
- `cv2.CAP_DSHOW` is for local webcams (not RTSP)
- If FFmpeg backend is missing in your OpenCV build, install a standard `opencv-python` wheel and verify support

---

## 8) Low-Latency RTSP Tips (Important for Entrance Tracking)

Low latency matters because your system uses:
- person detection
- tracker (ByteTrack / IoU tracker fallback)
- line crossing logic
- face recognition timing

### 1. Prefer substream `102` for real-time inference
Use:
```text
rtsp://username:password@IP:554/Streaming/Channels/102
```

Why:
- Lower resolution/bitrate -> faster decode
- Lower bandwidth -> fewer stalls
- Lower end-to-end latency on CPU systems

### 2. Choose RTSP transport: TCP vs UDP

#### TCP (recommended default)
- More reliable on Windows LANs
- Fewer visual artifacts/loss under network jitter
- Slightly higher latency than UDP in some cases

#### UDP (optional, low-latency tuning)
- Can be lower latency on a clean dedicated LAN
- More sensitive to packet loss/jitter

### 3. Reduce buffers
- In OpenCV/FFmpeg, reduce capture buffers where possible
- Keep `CAP_PROP_BUFFERSIZE` low (`1`) when supported
- Avoid large player-side buffering

### 4. Avoid H.265 for real-time AI ingestion
- H.265 often increases decode latency and CPU usage
- H.264 is more predictable and broadly compatible with OpenCV/FFmpeg pipelines

### 5. Keep resolution practical
- If latency is high on CPU, step down from `101` to `102`
- Consider `640x480` substream for CPU-only deployments

### 6. Avoid camera-side “smart” compression features
- Disable Smart Codec / H.264+ / H.265+
- These can hurt frame consistency and face detail during motion

---

## 9) Optional Ultra-Low Latency Method: FFmpeg Pipe into Python

For stricter low-latency requirements, you can use an FFmpeg subprocess and read raw frames via `stdout`.

This approach gives more control over transport, buffering, and decode options than `cv2.VideoCapture` in some environments.

### Python Example (FFmpeg -> rawvideo pipe)
```python
import subprocess
import numpy as np
import cv2

RTSP_URL = "rtsp://admin:StrongPass123@192.168.1.64:554/Streaming/Channels/102"
WIDTH = 640
HEIGHT = 480

cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",        # or udp
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-strict", "experimental",
    "-i", RTSP_URL,
    "-an",
    "-sn",
    "-dn",
    "-vf", f"scale={WIDTH}:{HEIGHT}",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo",
    "-f", "rawvideo",
    "-"
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

frame_bytes = WIDTH * HEIGHT * 3

try:
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            print("Short read / stream ended")
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        cv2.imshow("FFmpeg Pipe RTSP", frame)

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break
finally:
    proc.terminate()
    cv2.destroyAllWindows()
```

### When to use this method
- OpenCV RTSP capture is unstable on a specific Windows machine
- You need lower latency than `VideoCapture` provides
- You want explicit FFmpeg low-latency options (`-fflags nobuffer`, transport selection)

### Trade-offs
- More implementation complexity
- Manual frame size/format management
- Must handle restarts/reconnects explicitly

---

## 10) Performance Tuning Tips for Face Recognition (Attendance)

This system performs face recognition plus tracking, so camera settings and inference settings must be tuned together.

## Camera-side recommendations

### Use framing that prioritizes faces at the entrance
- Mount camera to capture faces near-frontal at check-in zone
- Avoid extreme top-down angles
- Ensure faces are large enough (not tiny in frame)

### Match stream resolution to hardware
- GPU system: can often use `101` (main stream) if latency is acceptable
- CPU-only system: prefer `102` substream to maintain stable FPS

## Pipeline-side recommendations

### 1. Use tracking to reduce redundant recognition
- Detect and track every frame (or near real-time)
- Run face embedding on best frames only (already part of your pipeline design)

### 2. Use best-frame selection for each track
- Prefer frames with:
  - high face detection score
  - high sharpness (variance of Laplacian)
  - low motion blur

### 3. Keep face crops clean
- Avoid over-compressed streams (bitrate too low)
- Disable Smart Codec
- Prefer H.264 for stable decode + quality

### 4. Tune thresholds empirically onsite
- Face threshold and ReID threshold must be calibrated on actual entrance footage
- Test with masks/helmets/bicycles

### 5. Minimize end-to-end latency
High latency can cause:
- delayed line crossing finalization
- stale track states
- poorer operator confidence during live review

### 6. Separate stream profiles if needed
If camera supports dual stream profiles:
- `101`: higher quality for debug/recording or enrollment checks
- `102`: inference stream for low latency

---

## 11) Troubleshooting

## A. RTSP stream does not open

### Symptoms
- VLC cannot connect
- OpenCV `cap.isOpened()` is false
- `401 Unauthorized` / `454 Session Not Found` / timeout

### Checks
- Confirm camera IP address (ping from Windows PC)
- Confirm RTSP port (`554`) and credentials
- Confirm RTSP is enabled in Hikvision UI
- Confirm user account has live view permissions
- Test both `101` and `102`
- Try VLC first before Python

### Example URLs to verify
```text
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/102
```

## B. High latency / delayed video

### Symptoms
- Video appears several seconds behind real time
- Line crossing events fire late

### Fixes
- Switch from `101` to `102`
- Reduce resolution and bitrate
- Disable Smart Codec (H.264+/H.265+)
- Use H.264 instead of H.265
- Use TCP first (stability) or try UDP on clean LAN
- Reduce OpenCV/FFmpeg buffers

## C. Stuttering / dropped frames

### Possible causes
- CPU overload on Windows PC
- Too high resolution or bitrate
- Network congestion
- H.265 decode overhead

### Fixes
- Use substream `102`
- Lower stream resolution (`640x480`)
- Lower FPS slightly (e.g., 20 FPS)
- Use GPU acceleration for inference if available
- Close other heavy applications on the Windows PC

## D. Face recognition accuracy is poor

### Common causes
- Camera angle too steep
- Backlighting / poor lighting
- Motion blur from bicycle entry
- Faces too small in frame
- Bitrate too low / aggressive compression

### Fixes
- Reposition camera for better face angle
- Improve lighting at entrance
- Increase face size in ROI (move camera closer / adjust zoom)
- Increase bitrate moderately
- Disable Smart Codec
- Enroll more representative face samples (masks, helmets, lighting variations)

## E. OpenCV RTSP works in VLC but not in Python

### Fixes
- Force `cv2.CAP_FFMPEG`
- Ensure OpenCV wheel includes FFmpeg backend
- Try FFmpeg pipe method (section 9)
- Test transport modes (`tcp`, `udp`)

## F. Intermittent reconnects

### Fixes
- Use static IP for camera
- Use wired Ethernet for camera and PC
- Check power stability (PoE switch health)
- Implement reconnect logic in the inference process

---

## 12) Security Notes (Offline LAN)

Even in a fully offline deployment, basic camera security matters.

### Change default credentials immediately
- Do not keep factory default `admin` password
- Use a strong password

### Limit access to local LAN only
- Keep camera on a private VLAN/LAN segment if possible
- Do not expose RTSP or web UI to the internet
- Restrict access to the Windows server and maintenance PCs only

### Use dedicated service account if supported
- Create a separate user for the attendance system with only required permissions (live view)
- Avoid using the main admin account in the inference URL when possible

### Document configuration locally
Store (securely, offline) the following for maintenance:
- Camera IP
- RTSP port
- Stream profile settings
- Credentials storage procedure (do not hardcode secrets in shared docs)

---

## 13) Webcam Fallback Instructions (Windows, `cv2.CAP_DSHOW`)

If the Hikvision camera is unavailable or during initial testing, use a local webcam.

### Why `CAP_DSHOW` on Windows
- `cv2.CAP_DSHOW` (DirectShow) is often more stable on Windows for webcams
- Avoids some backend auto-selection issues

### Python Example (Webcam Fallback)
```python
import cv2

# 0 = default webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Failed to open webcam")

# Optional webcam settings (depends on device support)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    cv2.imshow("Webcam Fallback", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
```

### In your attendance system CLI
Use webcam mode when needed (example):
```text
python -m inference.run --camera webcam --backend http://127.0.0.1:8000 --roi-config config/roi.yaml --show
```

---

## 14) Best Practices for Entrance Placement and Lighting

Correct camera placement matters more than model changes in many real deployments.

## Camera placement

### Height and angle
- Mount at approximately face-friendly height/angle (not extreme ceiling top-down)
- Aim for a slight downward angle, not vertical top-down
- Keep the entrance crossing zone within a stable field of view

### Distance to subject
- Ensure faces are sufficiently large when the person crosses the entry line
- Test with both walking and bicycle entry speeds

### Field of view (FOV)
- Avoid overly wide FOV if faces become too small
- Narrow the view to the actual entrance path/ROI

## Lighting

### Front/side lighting is best
- Illuminate faces from front or soft side lighting
- Avoid strong backlight from outdoors behind the subject

### Avoid rapid exposure changes
- Sudden bright/dark transitions can hurt face detection and tracking
- If possible, stabilize lighting in the entrance zone

### Night / low light
- If using IR/night mode, verify face quality for recognition (not just visibility)
- Excessive noise/compression degrades embeddings

## Scene setup for tracking reliability

- Define ROI polygon tightly around the entrance passage
- Place the virtual entry line where people pass one-by-one when possible
- Minimize background clutter and moving objects near the line
- Keep the camera physically stable (no vibration)

## Operational validation checklist (recommended)

Before going live, validate on-site with real staff traffic:
- Walking entries
- Bicycle entries
- Helmet/mask cases
- Morning lighting conditions (sunrise/backlight)
- Peak traffic crossing/occlusion cases
- Duplicate suppression behavior (single check-in per day)

---

## Quick Reference (Recommended Starting Point)

- RTSP URL: `rtsp://user:pass@IP:554/Streaming/Channels/102`
- Codec: `H.264`
- Resolution: `1280x720` (or `640x480` on CPU-only)
- FPS: `20-25`
- Smart Codec: `OFF`
- Test tool: `VLC`
- Python backend capture: `cv2.VideoCapture(..., cv2.CAP_FFMPEG)`
- Webcam fallback (Windows): `cv2.CAP_DSHOW`

