# 🔥 Ember! 🐲
**Emb**odied P**er**ception

## Repository Structure
- `ember`: Core utils used by many experiments
- `experiments`: Contains medium-to-large scale experiments / models which are in a too "early" state for integration as core components
- `notebooks`: Short one-off demos, tests, and experiments
- `scripts`: Like `notebooks` but meant for CLI interfacing


## Python setup
```bash
conda create --name ember python=3.11
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install streamlit opencv-python-headless kornia
```
