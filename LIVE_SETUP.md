# Live setup: camera → Wekinator → Max

The camera app turns your body into 21 numbers. Wekinator turns those 21 numbers into 25 control values. Max turns the 25 values into sound.

```
camera ──► pose app ──21 numbers──► Wekinator ──25 numbers──► Max (sound)
           (this repo)  port 6448    (wekinator/)  port 12000   (max/)
```

Start the three programs in this order: the app, then Wekinator, then Max.

## Words used here

- **OSC**: a simple way for music programs to send numbers to each other over the network. For example, the app sends `/pose/data 0.12 -0.40 ...` to port 6448 about 30 times a second.
- **Port**: a numbered "mailbox" on the computer. A program listens on one port, and the other program sends to that number. `127.0.0.1` and `localhost` both mean "this Mac".
- **Wekinator**: a free app that learns from examples. You show it a pose and the sound settings you want for that pose. After training, it outputs those settings whenever it sees a similar pose. It is installed at `/Applications/Wekinator.app`.

## 1. Start the pose app (built-in camera)

```bash
cd /Users/weidongyang/KinetechArts/Projects/rerun.io/dance_embedding
uv run examples/osc_streaming_example.py --camera-id 0 --skip-matching --raw-pose
```

- `--camera-id 0` is the MacBook camera. Camera 1 is the Canon EOS webcam. To list them, run `uv run examples/osc_streaming_example.py --list-cameras`.
- `--skip-matching` runs only the pose tracking and OSC. It does not search the dance video database.
- A camera window opens. Press `q` in that window to quit, or `v` to start and stop recording.

You should see these lines:

```
✅ OSC client initialized for pose_data at 127.0.0.1:6448
✅ Camera 0 initialized successfully
```

## 2. Start Wekinator

1. Open **Wekinator** from `/Applications`.
2. Choose **File → Open project…** and pick
   `wekinator/21_plus_4/WekinatorProject.wekproj`.
3. Click **Start Listening** if it is not already on. The OSC light turns green once numbers from the app arrive.
4. Click **Run**. Wekinator now sends its 25 values to Max.

The project is already set up as follows:

| Setting | Value |
|---|---|
| Listens on port | `6448` |
| Input message | `/pose/data`, 21 inputs |
| Sends to | `localhost`, port `12000`, message `/wek` |
| Outputs | 21 sliders, `outputs-1` … `outputs-21` (0.0 to 1.0), then 4 class choices, `class-1` … `class-4` |
| Training examples saved | 662 |

**Retraining:** set the outputs to the sound you want, strike the pose, and click **Start Recording**. Click it again to stop. Then click **Train**, then **Run**. Choose **File → Save** to keep the new examples. They are saved into `wekinator/21_plus_4/` and show up as changes in git.

## 3. Start Max

Open these two patches from `max/`:

- `wek-receive-05.maxpat`: listens on port `12000` (`udpreceive 12000`), takes the `/wek` message, and splits it into 21 sliders and 4 classes.
- `_paf-8out-Wek-03.maxpat`: the synth, built from 4 `paffy~` voices. It receives the values from `wek-receive-05`.

The other patches in `max/` (`paffy~`, `paffycontrol`, `fader`, and others) are parts that these two patches use. Keep them in the same folder.

## When something doesn't work

| What you see | Likely cause | Fix |
|---|---|---|
| The Wekinator OSC light stays grey | The app isn't running, or is sending to another port | Check that the app printed `127.0.0.1:6448`. |
| Wekinator says the input count doesn't match | It is getting a different message, not the 21 values | The raw landmarks (132 values) go to port **6449**, not 6448. Check `src/recall/config.json`. |
| Max gets nothing | Wekinator is not in **Run** mode, or another program already uses port 12000 | Click **Run**. Quit other copies of the patch. |
| The camera window is black | The wrong camera was picked | Run `--list-cameras` and try the other number. |

---

## Technical Detail

All the OSC ports the app sends to. They are set in `src/recall/config.json` under `osc_streaming.streams`.

| Stream | Port | Address | Values | Used by |
|---|---|---|---|---|
| `pose_data` | 6448 | `/pose/data` | 21 floats | Wekinator |
| `pose_data_duplicate` | 1234 | `/pose/data` | 21 floats | GraphXR forward |
| `raw_pose_landmarks` | 6449 | `/pose/raw_landmarks` | 132 floats (33 points × x, y, z, confidence) | `src/osc_to_websocket/simple_server.py` |

The 21 inputs, in order (names from `wekinator/21_plus_4/inputConfig.xml`):
`lh-x lh-y lh-z rh-x rh-y rh-z lf-x lf-y lf-z rf-x rf-y rf-z body-yaw body-pitch head-yaw head-pitch pos-x pos-y pos-z vel acc`.
Here lh, rh, lf and rf mean left hand, right hand, left foot and right foot. The meaning of each value is in `src/recall/OSC_STREAMING.md`.

Where the files came from: the Wekinator project was copied from `2025_Montalvo/21_plus_4/WekinatorProject`, and the Max patches from `2025_Montalvo/max_02` (the newest set, Sep 2025). The originals are still in `2025_Montalvo`, so remember that Wekinator saves to whichever copy you opened.
