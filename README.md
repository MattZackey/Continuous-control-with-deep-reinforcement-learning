# Continuous control with deep reinforcement learning
The following is the [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) algorithm applied to several OpenAI gym environments.

# Lunar Lander

<div style="display: flex;">

  <div style="flex: 1; text-align: center;">
    <h3>Episode 100</h3>
    <div style="border: 1px solid black; padding: 5px; display: inline-block">
      <img src="Lunarlander_results/run100.gif" alt="Image 1" style="max-width: 70%; width: 400px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Episode 500</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="Lunarlander_results/run500.gif" alt="Image 2" style="max-width: 70%; width: 400px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Episode 2000</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="Lunarlander_results/run2000.gif" alt="Image 3" style="max-width: 70%; width: 400px;">
    </div>
  </div>

</div>

The following figure shows the score the agent achieves per espisode of training.

![Results](https://github.com/MattZackey/Deep-Deterministic-Policy-Gradient/blob/main/Results%20Lunar%20Lander.png?raw=true)

# Mountain Car Continuous

The goal is to reach the flag in the least amount of time possible.

<div style="display: flex;">

  <div style="flex: 1; text-align: center;">
    <h3>Episode 50</h3>
    <div style="border: 1px solid black; padding: 5px; display: inline-block">
      <img src="MountainCar_results/run50.gif" alt="Image 1" style="max-width: 70%; width: 400px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Episode 150</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="MountainCar_results/run150.gif" alt="Image 2" style="max-width: 70%; width: 400px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Episode 300</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="MountainCar_results/run300.gif" alt="Image 3" style="max-width: 70%; width: 400px;">
    </div>
  </div>

</div>

The following figure shows the score the agent achieves per espisode of training.

![Results](https://github.com/MattZackey/Deep-Deterministic-Policy-Gradient/blob/main/Results%20Mountain%20Car.png?raw=true) 

# Pendulum

The goal is to swing the Pendulum into an upright position.

<div style="display: flex;">

  <div style="flex: 1; text-align: center;">
    <h3>Episode 20</h3>
    <div style="border: 1px solid black; padding: 5px; display: inline-block">
      <img src="Pendulum_results/run20.gif" alt="Image 1" style="max-width: 70%; width: 400px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Episode 40</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="Pendulum_results/run40.gif" alt="Image 2" style="max-width: 70%; width: 400px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Episode 200</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="Pendulum_results/run200.gif" alt="Image 3" style="max-width: 70%; width: 400px;">
    </div>
  </div>

</div>

The following figure shows the score the agent achieves per espisode of training.

![Results](https://github.com/MattZackey/Deep-Deterministic-Policy-Gradient/blob/main/Results%20Pendulum.png?raw=true) 
