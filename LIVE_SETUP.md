# Live setup: camera → Wekinator → Max

You move in front of the laptop camera and the sound changes. Three programs pass numbers along a chain:

```
 laptop camera
      │
      ▼
 pose app (this repo)    sees your body and turns it into 21 numbers
      │  port 6448, about 18 times a second
      ▼
 Wekinator               learned which poses go with which sound settings
      │  port 12000
      ▼
 Max                     plays the sound
```

**Every time:** start the app, then Wekinator, then Max. First time on a new Mac? Do [One-time install](#one-time-install) first.

---

## Words used here

- **OSC**: a simple way for music programs to send numbers to each other. Example: the app sends the message `/pose/data 0.12 -0.05 1.64 …` (21 numbers) many times a second.
- **Port**: a numbered "mailbox" on the computer. One program listens on a number and another sends to it. Only **one** program can listen on a port at a time. `127.0.0.1` and `localhost` both mean "this Mac".
- **Wekinator**: a free app that learns from examples. You strike a pose and set the sound sliders. After you show it a few poses and click Train, it moves the sliders on its own whenever it sees a similar pose.
- **Max**: a visual music programming app from Cycling '74. A Max file is called a *patch* (`.maxpat`).

---

## One-time install

| You need | Where to get it | Check that it works |
|---|---|---|
| This repo | `git clone https://github.com/kinetecharts/dance-embedding.git` | `ls LIVE_SETUP.md` |
| `uv` (runs the Python app and installs its parts) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `uv --version` |
| Wekinator 2.1 | [wekinator.org/downloads](http://www.wekinator.org/downloads/). Drag it into `/Applications`. Java is built in. | It opens |
| Max 8 or newer (the patches were made in Max 8.6.2) | [cycling74.com](https://cycling74.com/downloads) | It opens |

Then install the app's Python parts once:

```bash
cd dance-embedding        # the repo folder
uv sync                   # takes a few minutes the first time
```

**Camera permission:** the first time the app starts, macOS asks whether **Terminal** (or iTerm, or VS Code, whichever you run it from) may use the camera. Click **Allow**. If you clicked Don't Allow, turn it on in **System Settings → Privacy & Security → Camera**, then restart Terminal.

---

## Every time: start the three programs

### Step 1. Start the pose app

Run this from the repo folder. The app reads its settings from `src/recall/config.json` relative to that folder, so other folders won't work.

```bash
uv run examples/osc_streaming_example.py --camera-id 0 --skip-matching --raw-pose
```

**In plain words:** "use the built-in camera, send OSC only, and skip the dance-video search."

A window opens with your camera picture and a skeleton drawn on you. The terminal shows:

```
✅ OSC client initialized for pose_data at 127.0.0.1:6448
✅ Camera 0 initialized successfully
```

- To quit, click the camera window and press `q`. Press `v` to start and stop recording a video.
- Picture black or showing the wrong camera? Run `uv run examples/osc_streaming_example.py --list-cameras` and try the other number. On the studio MacBook, `0` is the built-in camera and `1` is the Canon EOS webcam.

### Step 2. Start Wekinator

1. Open **Wekinator**.
2. On the welcome screen, click **Open an existing project** (or choose **File → Open project…**) and pick
   `wekinator/21_plus_4/WekinatorProject.wekproj` in this repo.
3. If **Start Listening** is showing, click it. The **OSC In** light flashes green while numbers arrive from the app. If it stays dark, see [Troubleshooting](#troubleshooting).
4. Click **Run**. Wekinator now moves its 25 output sliders as you move and sends them to Max.

**In plain words:** Wekinator listens on mailbox 6448, guesses the sound settings for your current pose, and posts them to mailbox 12000.

### Step 3. Start Max

Open these two patches from the `max/` folder (double-click them in Finder):

- **`wek-receive-05.maxpat`** receives from Wekinator on port 12000 and splits the message into 21 sliders + 4 classes.
- **`_paf-8out-Wek-03.maxpat`** is the synth: 4 voices built from `paffy~`.

Turn on audio in Max (the speaker button, or **Options → Audio Status → Audio On**). Then move, and you should hear the sound change.

Keep all the `.maxpat` files together in `max/`. The two patches above load the others (`paffy~`, `paffycontrol`, `fader` and so on) by name.

---

## Check it's working

Each program shows a sign of life:

| Where | What you should see |
|---|---|
| App camera window | A skeleton drawn on your body |
| Wekinator | OSC In light flashing; output sliders moving when you move |
| Max `wek-receive-05` | Number boxes changing |

To check the numbers themselves, use the checker. **Quit Wekinator first**, because only one program can listen on 6448:

```bash
uv run scripts/osc_check.py          # what Wekinator would hear (port 6448)
```

Good output (the app is sending):

```
Listening on port 6448. Press Ctrl+C to stop.
/pose/data: 18 msgs/s, 21 values, first 3: [0.12, -0.05, 1.64]
```

Bad output: `nothing yet` every second means the app isn't running, or isn't sending to 6448.

The same tool can check Wekinator → Max. Quit Max, start Wekinator in **Run**, then run `uv run scripts/osc_check.py 12000`. You should see `/wek` with 25 values.

---

## Train Wekinator on new poses

The saved project already holds 662 examples. To add your own:

1. In Wekinator, click **Stop Running** if it is running.
2. Set the output sliders to the sound you want for a pose. Max must be running so you can hear it.
3. Hold the pose in front of the camera. Click **Start Recording**, move a little within the pose for 2–3 seconds, then click **Stop Recording**.
4. Repeat steps 2–3 for each pose you want. Use 3–5 different poses with clearly different sounds.
5. Click **Train**, then **Run**. Move between your poses and listen.
6. **File → Save** keeps the examples. They are saved into `wekinator/21_plus_4/` and show up in `git status`. Commit them if you want to keep them.

To start over without the old examples, use **Delete all examples** in Wekinator before step 2. Then Train again.

---

## Stop everything

1. Click the app's camera window and press `q`.
2. Quit Wekinator. Save first if you trained.
3. Quit Max.

---

## Troubleshooting

| What you see | Likely cause | Fix |
|---|---|---|
| Wekinator's OSC In light stays dark | The app isn't running, or it was started from the wrong folder | Restart the app from the repo folder. Check that it printed `127.0.0.1:6448`. |
| Wekinator: "Port 6448 already in use" | Another program, or a second Wekinator, is listening there | Quit the other one. `lsof -nP -iUDP:6448` shows who has it. |
| `osc_check.py`: "Port 6448 is already in use" | Wekinator is running | Quit Wekinator, or check port 1234 instead. The app sends the same 21 numbers there. |
| Wekinator complains about the number of inputs | It is getting a different message | It should get `/pose/data` with 21 values. The 132-value raw landmarks go to **6449**. See `src/recall/config.json`. |
| Max gets nothing | Wekinator isn't in **Run** mode, or two copies of the patch are open | Click **Run**. Close duplicate patches. `lsof -nP -iUDP:12000` should show one `Max`. |
| No sound, but numbers move in Max | Audio is off | Turn on audio in Max and check the output device in **Options → Audio Status**. |
| Black camera window, or `Failed to open camera` | No camera permission, or the wrong camera number | See [Camera permission](#one-time-install). Try `--camera-id 1`. |
| `Config file not found: src/recall/config.json` | The app was started outside the repo folder | `cd` into the repo folder first. |

---

## Technical Detail

**All the OSC the app sends** (set in `src/recall/config.json`, under `osc_streaming.streams`):

| Stream | Port | Address | Values | Read by |
|---|---|---|---|---|
| `pose_data` | 6448 | `/pose/data` | 21 floats | Wekinator |
| `pose_data_duplicate` | 1234 | `/pose/data` | 21 floats | GraphXR forward |
| `raw_pose_landmarks` | 6449 | `/pose/raw_landmarks` | 132 floats (33 points × x, y, z, confidence) | `src/osc_to_websocket/simple_server.py` |

**The 21 inputs, in order** (from `wekinator/21_plus_4/inputConfig.xml`):
`lh-x lh-y lh-z rh-x rh-y rh-z lf-x lf-y lf-z rf-x rf-y rf-z body-yaw body-pitch head-yaw head-pitch pos-x pos-y pos-z vel acc`.
Here lh, rh, lf and rf mean left hand, right hand, left foot and right foot. Hands and feet are measured relative to the body, so stepping closer to the camera doesn't change them. How each value is computed is in `src/recall/OSC_STREAMING.md`.

**The Wekinator project** (`wekinator/21_plus_4/`):

| File | What it holds |
|---|---|
| `WekinatorProject.wekproj` | Receive port, `6448` |
| `inputConfig.xml` | Input message `/pose/data`, 21 named inputs |
| `outputConfig.xml` | Sends `/wek` to `localhost:12000`. Outputs: `outputs-1`…`outputs-21` (number, 0.0–1.0, soft limits), then `class-1`…`class-4` (classes) |
| `current/currentData.arff` | The 662 training examples |
| `current/models/model0.xml` … `model24.xml` | One trained model per output |

**The Max side:** `max/wek-receive-05.maxpat` runs `udpreceive 12000` → `OSC-route /wek` → `unpack` (21 floats + 4 ints) → `s` sends such as `1-gain` and `2-topaffyline`, which `_paf-8out-Wek-03.maxpat` picks up.

**Where these came from:** copied from `~/KinetechArts/Projects/2025_Montalvo/21_plus_4/WekinatorProject` and `2025_Montalvo/max_02` (the newest patch set, Sep 2025). The originals still exist there. Wekinator saves into whichever copy you opened, so use the one in this repo.
