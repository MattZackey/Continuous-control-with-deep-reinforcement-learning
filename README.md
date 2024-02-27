# Continuous control with deep reinforcement learning
The following is an example of applying the [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) algorithm to the [Pendulum swingup](https://www.gymlibrary.dev/environments/classic_control/pendulum/) environment from Open AI gymnasium. The goal is to swing the Pendulum into an upright position.

To come: Applying DDPG to several MuJoCo and Box2D environments. 

# Training results

<div>
    <img src="Pendulum_results/run20.gif" alt="Image 1" title="Episode 20" style="width: 45%; display: inline-block;">
    <img src="Pendulum_results/run40.gif" alt="Image 2" title="Episode 40" style="width: 45%; display: inline-block;">
</div>

The following figure shows the score the agent achieves per espisode of training.

![Results](https://github.com/MattZackey/Deep-Deterministic-Policy-Gradient/blob/main/Training%20results.png?raw=true) 
