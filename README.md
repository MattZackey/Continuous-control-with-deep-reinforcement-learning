# Continuous control with deep reinforcement learning
The following is the [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) algorithm applied to the Pendulum environment.

# Pendulum

The goal is to swing the Pendulum into an upright position.

<div style="display: flex;">

  <div style="flex: 1; text-align: center;">
    <h3>Episode 20</h3>
    <div style="border: 1px solid black; padding: 5px; display: inline-block">
      <img src="Pendulum_results/run20.gif" alt="Image 1" style="max-width: 50%; width: 150px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Episode 40</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="Pendulum_results/run40.gif" alt="Image 2" style="max-width: 50%; width: 150px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Episode 200</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="Pendulum_results/run200.gif" alt="Image 2" style="max-width: 50%; width: 150px;">
    </div>
  </div>

</div>

The following figure shows the score the agent achieves per espisode of training.

![Results](https://github.com/MattZackey/Deep-Deterministic-Policy-Gradient/blob/main/Results%20Pendulum.png?raw=true) 
