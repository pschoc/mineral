# ppo

# Setup

```bash
conda create -n fp python=3.8
conda activate fp

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

pip install wheel==0.38.4 setuptools==66
pip install gym==0.19.0 hydra-core hydra_colorlog wandb termcolor tensorboard moviepy h5py

# fix `OSError: Pillow was built without XCB support`
conda remove Pillow && pip install -U Pillow

cd third_party/isaacgym/python
pip install .
cd -
cd third_party/IsaacGymEnvs
# comment out gym and rl_games in INSTALL_REQUIRES
pip install .
cd -
```

# Train

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a -s "-screen 0 256x256x24 -ac +extension GLX +render -noreset" \
  python -m mineral.ppo.run \
  task=Cartpole \
  logdir="workdir/Cartpole/$(date +%Y%m%d-%H%M%S)" \
  env_render=True train.ppo.save_video_every=25 train.ppo.save_video_consecutive=5
 
CUDA_VISIBLE_DEVICES=0 xvfb-run -a -s "-screen 0 256x256x24 -ac +extension GLX +render -noreset" \
  python -m mineral.ppo.run \
  task=AllegroHand logdir="debug/AllegroHand$(date +%Y%m%d-%H%M%S)"
```
