# Continuous control with deep reinforcement learning
The following is an example of applying the [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) algorithm to the [Pendulum swingup](https://www.gymlibrary.dev/environments/classic_control/pendulum/) environment from Open AI gymnasium. The goal is to swing the Pendulum into an upright position.

To come: Applying DDPG to several MuJoCo and Box2D environments. 

# Training results

<div style="display: flex; justify-content: space-between;">
    <div style="border: 1px solid black; padding: 10px; text-align: center;">
        <h2>Heading 1</h2>
        <img src="Pendulum_results/run20.gif" alt="Image 1" title="Title for Image 1" style="width: 100%;">
    </div>
    <div style="border: 1px solid black; padding: 10px; text-align: center;">
        <h2>Heading 2</h2>
        <img src="Pendulum_results/run40.gif" alt="Image 2" title="Title for Image 2" style="width: 100%;">
    </div>
</div>

The following figure shows the score the agent achieves per espisode of training.

![Results](https://github.com/MattZackey/Deep-Deterministic-Policy-Gradient/blob/main/Training%20results.png?raw=true) 
