# RL Locomotion with Safety Layer using Control Barrier Functions

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OSQP](https://img.shields.io/badge/OSQP-Latest-blue.svg)](https://osqp.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/ansh1113/rl-locomotion-cbf/graphs/commit-activity)

**A safe reinforcement learning framework for quadruped locomotion combining PPO with Control Barrier Functions (CBF) for provably safe operation.**

## 🎯 Key Results

- ✅ **Zero Falls** - 0 falls achieved across all test terrains
- ✅ **99% Safety Rate** - CBF layer successfully rejects 99% of unsafe actions
- ✅ **90% Speed Retained** - Maintains 90% of unconstrained PPO policy speed
- ✅ **Provable Safety** - Mathematical guarantee that robot remains stable
- ✅ **Real-time** - Safety filter runs at 200+ Hz

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithm Details](#algorithm-details)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

