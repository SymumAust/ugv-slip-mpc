# Data

`processed/<drive>/` holds compact extracts of the CitrusFarm signals this project uses, written by `extract.py`:

| File | Columns | Source topic |
|---|---|---|
| `odom.csv.gz` | t, v_w, w_w | `/jackal_velocity_controller/odom` (50 Hz) |
| `imu.csv.gz` | t, gx, gy, gz, ax, ay, az (sensor frame) | `/microstrain/imu/data` or `/zed2i/zed_node/imu/data` |
| `gt.csv.gz` | t, x, y (GPS antenna, as recorded) | `/gps/fix/odometry` (gt.bag) or gt.csv |
| `meta.json` | source files, IMU source, row counts, load flags | |

Nothing is filtered or synchronised at this stage, so every later step can be reproduced from these files alone.

**Raw bags are not stored here.** They are many GB and are publicly re-downloadable; see `docs/feasibility_README.md` for the exact `aws s3 sync` commands.

**Source and licence.** CitrusFarm dataset, H. Teng, Y. Wang, X. Song, K. Karydis, "Multimodal Dataset for Localization, Mapping and Crop Monitoring in Citrus Tree Farms", ISVC 2023 (arXiv:2309.15332). The dataset is licensed CC BY-SA 4.0, so these derived files are also CC BY-SA 4.0 and must keep this attribution if shared.
