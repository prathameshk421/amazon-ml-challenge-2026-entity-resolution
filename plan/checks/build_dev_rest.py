"""Candidates + stage-3 features for the two DEV_REST slices, without the learning-curve training that
dev_rest_180k.py / dev_rest_full.py also run. Same functions, same seeded S1 slices, same output directories:
  DEV_REST_180K  (180,000 S1)  -> cross-encoder ce_rec training pairs
  DEV_REST_REST  (the rest)    -> fold 0: cross-encoder c0 training pairs; fold 1: matcher training rows

  PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/build_dev_rest.py
"""
import time

from ber import io_utils, match
import dev_rest_180k as d180
from dev_rest_full import SET_NAME_REST, pick_remaining_slice


def main():
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    lg = io_utils.get_logger("build_dev_rest")
    s1_rest, s1_180k = pick_remaining_slice()
    for name, s1 in (("DEV_REST_180K", s1_180k), (SET_NAME_REST, s1_rest)):
        d180.SET_NAME = name  # generate_candidates and build_features write to work/{candidates,match/features}/<SET_NAME>
        print(f">>> {time.strftime('%H:%M:%S')} {name}: candidates for {s1.size:,} S1", flush=True)
        d180.generate_candidates(s1, cfg, lg)
        print(f">>> {time.strftime('%H:%M:%S')} {name}: features", flush=True)
        d180.build_features(s1, cfg, mcfg, lg)
    d180.SET_NAME = "DEV_REST_180K"


if __name__ == "__main__":
    main()
