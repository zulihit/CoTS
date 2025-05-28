<div align="center">
  <table>
    <tr>
      <td><h1>CoTS: Collaborative Tree Search Enhancing Embodied Multi-Agent Collaboration</h3></td>
    </tr>
  </table>
</div>


<p align="center">
<a href="https://arxiv.org/abs/XXXX.XXXXX" alt="arXiv">
    <img src="https://img.shields.io/badge/paper-Coming--soon-b31b1b.svg?style=flat" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.x%20%7C%202.x-673ab7.svg" alt="Tested PyTorch Versions"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-4caf50.svg" alt="License"></a>
</p>

<p align="center">
<b>CVPR 2025</b>  
<br><br>
<img src="assets/cots.png" width="600">
</p>

---



This repository contains the official implementation of our CVPR 2025 paper:

**Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration**  
_Lizheng Zu, Lin Lin*, Song Fu*, Na Zhao, Pan Zhou_  

> CoTS is a novel decision-making framework designed to improve embodied multi-agent collaboration via tree-structured reasoning and reflective evaluation, built as a modular enhancement over CoELA.

---

## 🧠 Overview
CoTS builds upon the modular framework of [CoELA](https://umass-embodied-agi.github.io/CoELA/) by introducing **tree-based planning**, **collaborative reasoning**, and **reflective decision updates**. Through Monte Carlo Tree Search (MCTS) guided by large language models (LLMs), CoTS enables agents to:

- Reason over future trajectories via simulation.
- Reflect on decision quality using custom scoring mechanisms.
- Adaptively refine multi-agent plans with minimal supervision.



---

## 📄 Paper
- Paper (CVPR 2025): [Coming soon]
- Project Website: [Coming soon]

---

## 🔧 Installation
Please refer to the original environment setup instructions from [CoELA](https://github.com/umass-embodied-agi/CoELA), as CoTS operates within the same `ThreeDWorld Multi-Agent Transport (TDW-MAT)` environment.

```bash
cd tdw_mat
conda create -n cots python=3.9
conda activate cots
pip install -e .
```

We also recommend using the fine-tuned vision detection model provided by CoELA. For instructions, see `tdw_mat/README.md`.

---

## 🚀 Run CoTS
Scripts for running experiments with CoTS are available in `tdw_mat/scripts`.

To run CoTS in the TDW-MAT environment:

```bash
./scripts/run_cots.sh
```

This will launch a two-agent collaborative task using CoTS as the reasoning backbone.

---

## 🔍 Key Features
- **Tree-Structured Planning**: Agents explore multiple potential action paths before execution.
- **Reflective Evaluation**: Using Dis_Score and Task_Score, agents self-evaluate and prune bad trajectories.
- **MCTS-LLM Integration**: Seamless integration of Monte Carlo Tree Search with LLM-based policy priors.

---

## 📊 Metrics
CoTS performance is evaluated with the same metrics as CoELA:

- **Transport Rate (TR)**: Successful object delivery ratio.
- **Efficiency Improvement (EI)**: Performance gain relative to non-LLM or static baseline agents.

Additional metrics:
- **Decision Consistency**: How often agents maintain coherent plans across reflective rounds.
- **Exploration Depth**: Average planning tree depth per decision.

---

## 🌍 Environment: TDW-MAT
We adopt the TDW Multi-Agent Transport environment introduced in CoELA, which features:
- Multiple rooms with task-relevant object and container layouts.
- Visual perception via Mask-RCNN detection.
- Communication-enabled action space.

See [CoELA TDW-MAT](https://github.com/umass-embodied-agi/CoELA) for installation and environment structure.

---

## 📽️ Demos and Qualitative Results
We demonstrate qualitative behaviors of CoTS agents in cooperative scenarios:

- Role negotiation via LLM chat
- Adaptive task switching
- Reflective corrections of inefficient plans

![Qualitative Results](assets/cots_qualitative.png)

More demos and videos: [Coming soon]

---

## 📚 Citation
If you find CoTS helpful in your research, please consider citing:

```bibtex
@inproceedings{zu2025cots,
  title={Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration},
  author={Zu, Lizheng and Lin, Lin and Fu, Song and Zhao, Na and Zhou, Pan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

---

## 🤝 Acknowledgements
This work builds on the foundation of [CoELA](https://github.com/umass-embodied-agi/CoELA). We thank the original authors for releasing their code and environment.

---
