# Learned slip-residual MPC for a skid-steer agricultural UGV

Term project for MO89905 (Prof. Kyunghwan Choi, KAIST, Fall 2026): learning-based control of a Clearpath Jackal driving in citrus orchards, using the public CitrusFarm dataset.

**Current stage: go/no-go feasibility check.** Before any MPC work, we test whether a learned slip residual predicts the robot's 1–2 s motion clearly better than nominal kinematics and a calibrated physics (ICR) slip model. The rules are fixed in advance in [`PLAN.md`](PLAN.md).

## Layout

```
PLAN.md                  pre-registered decision rules (+ amendments log)
extract.py               raw bags -> data/processed/<drive>/ (small, committed)
feasibility.py           slip computation, models, rollout errors, verdict
selftest_synthetic.py    fake drives with a known slip model, to test the pipeline
config.json              drives, fields, GPS lever arm, paths
data/processed/          compact extracts of odometry, IMU, GPS (committed)
results/                 decision.md, tables, figures from the real run (committed)
docs/                    detailed run instructions, self-test example outputs
```

## Workflow

```bash
pip install -r requirements.txt
# 1. download raw bags outside the repo (see docs/feasibility_README.md), set data_root in config.json
python extract.py --config config.json            # writes data/processed/, then commit it
# 2. set lever_arm_xy in config.json from the calibration files
python feasibility.py --config config.json        # writes results/, then commit it
```

Once `data/processed/` is committed, anyone who clones the repo can rerun step 2 without the raw bags.

## Timeline

| Date | Milestone |
|---|---|
| Nov 18 | Feasibility verdict |
| Nov 22 | Proposal due |
| Dec 13 | Final report + runnable controller |
| Dec 15 | Individual Q&A |
