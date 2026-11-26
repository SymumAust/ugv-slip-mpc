#!/usr/bin/env bash
# Download only the CitrusFarm files this project needs (drives 01-05: odom + base bags + ground truth,
# plus the two calibration files for the GPS lever arm). Plain HTTPS with wget; resumable; MD5-checked.
# Usage:  bash download_data.sh [target_dir]      (default: ~/citrusfarm)
# Optional: SKIP_BASE=1 bash download_data.sh     (skip the large base_*.bag IMU/LiDAR files, e.g. if you use ZED bags)
# Optional: VERIFY_ONLY=1 bash download_data.sh   (no network: only check the files you already have)
# Files already present with the correct MD5 are never re-downloaded.
set -u
D="${1:-$HOME/citrusfarm}"
B="https://ucr-robotics.s3.us-west-2.amazonaws.com/citrus-farm-dataset"
FILES=(
  "01_13B_Jackal/base_2023-07-18-14-26-48_0.bag 2030090a78eee36ee0661cf46a8d5853"
  "01_13B_Jackal/base_2023-07-18-14-35-37_1.bag cc587512aeb390f54d40dd0f82df3727"
  "01_13B_Jackal/base_2023-07-18-14-44-35_2.bag 81e712ba03053c3e3e306ac83ee65936"
  "01_13B_Jackal/odom_2023-07-18-14-26-48.bag 016555f3497e3fedc6316de5c21bc823"
  "ground_truth/01_13B_Jackal/gt.bag f8a5725dd3ad701c9d8beddaf98ed8a0"
  "ground_truth/01_13B_Jackal/gt.csv 99d04794d4865df5e8d38a5f3921b52b"
  "02_13B_Jackal/base_2023-07-18-14-51-21_0.bag c1b0034497cf081be728eb8611ba746a"
  "02_13B_Jackal/odom_2023-07-18-14-51-21.bag e5b289bb5060c36d3baf66419f6d0041"
  "ground_truth/02_13B_Jackal/gt.bag fdba96931786d47bed81a76515446566"
  "ground_truth/02_13B_Jackal/gt.csv a6264633d338792129cd91532b4d6fd3"
  "03_13B_Jackal/base_2023-07-16-19-17-20_0.bag 37edc0605079497b689e50b51a023d8e"
  "03_13B_Jackal/base_2023-07-16-19-26-21_1.bag 040b901a5b98914a8da8995be8bc3674"
  "03_13B_Jackal/base_2023-07-16-19-35-22_2.bag 3ab4c3a1274ef88bad6ea29a13ec9145"
  "03_13B_Jackal/base_2023-07-16-19-44-22_3.bag e841ab1c7c02528b5913888a6ddebe0a"
  "03_13B_Jackal/odom_2023-07-16-19-17-20.bag d6188dbf07899e214a60642d41736271"
  "ground_truth/03_13B_Jackal/gt.bag fac413239f2605838914c87be5690b91"
  "ground_truth/03_13B_Jackal/gt.csv 58114a58d213810e09f70c98c51a18ec"
  "04_13D_Jackal/base_2023-07-18-14-25-34_0.bag 2c0dd381142f2a0c59c4388587688b2a"
  "04_13D_Jackal/base_2023-07-18-14-34-05_1.bag 0b9e7e14adb40d208184139f47bc148f"
  "04_13D_Jackal/odom_2023-07-18-14-25-34.bag 8d6344241e2307bc11f88c8873342c40"
  "ground_truth/04_13D_Jackal/gt.bag 873e7fcdbfda6b1957bd38eb8864dda8"
  "ground_truth/04_13D_Jackal/gt.csv c1e5603cf9547221769139a9afb1160b"
  "05_13D_Jackal/base_2023-07-16-17-05-13_0.bag c22c7d8433561e39ef0a6f2a7f14b05e"
  "05_13D_Jackal/base_2023-07-16-17-13-40_1.bag 07657aa15e6cc69d70a7bb771ff5de04"
  "05_13D_Jackal/odom_2023-07-16-17-05-13.bag 7801b5838233c8af44b9cd1d40b85a55"
  "ground_truth/05_13D_Jackal/gt.bag 68bf2bce57dd1e9dc32a408846475170"
  "ground_truth/05_13D_Jackal/gt.csv 53c3f5553988216faf46e6a8a5db2228"
  "Calibration/results/04-lidar-gps-result.txt fb59ab6a84ce69c13a67e2dd1b9722e6"
  "Calibration/results/05-baselink-lidar-result.txt 038af200f21ef18ee467f6cc9f5efeea"
)
fail=0
for entry in "${FILES[@]}"; do
  p="${entry% *}"; md5="${entry##* }"
  case "$p" in */base_*) [ "${SKIP_BASE:-0}" = "1" ] && { echo "skip  $p"; continue; };; esac
  mkdir -p "$D/$(dirname "$p")"
  if [ -s "$D/$p" ] && [ "$(md5sum "$D/$p" | cut -d" " -f1)" = "$md5" ]; then echo "ok    $p (already complete)"; continue; fi
  if [ "${VERIFY_ONLY:-0}" = "1" ]; then
    if [ -s "$D/$p" ]; then echo "BAD MD5: $p (incomplete or corrupt)"; else echo "missing $p"; fi
    fail=1; continue
  fi
  echo "get   $p"
  wget -q --show-progress -c -O "$D/$p" "$B/$p" || { echo "FAILED download: $p"; [ -s "$D/$p" ] || rm -f "$D/$p"; fail=1; continue; }
  got=$(md5sum "$D/$p" | cut -d" " -f1)
  if [ "$got" = "$md5" ]; then echo "ok    $p"; else echo "BAD MD5: $p (delete it and rerun)"; fail=1; fi
done
[ $fail = 0 ] && echo "All files downloaded and verified in $D" || echo "Some files failed; rerun the script (it resumes)."
