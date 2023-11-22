# Soft_Actor_Critic
Implementation of Soft_Actor_Critic


# Requirements
gym == 0.22.0

gym-robotics

pytorch-lightining == 1.6.0

torch == 2.0.1

# Collab installations
!apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    xvfb \
    libosmesa6-dev \
    software-properties-common \
    patchelf
    

!pip install \
    free-mujoco-py \
    gym==0.22 \
    pytorch-lightning==1.6.0 \
    optuna \
    pyvirtualdisplay \
    PyOpenGL \
    PyOpenGL-accelerate\
    gym-robotics


# Description
SAC is an actor-critic method that uses a deterministic policy to achieve the most optimal path of maximizing reward and two critic networks that get the value of a state (s') based on a current state (s) and policy action π(s) = a, Q(s, π(s)). The twin critic networks similar to Twin Delay DDPG architecture are independent of each other and are responsible of giving minimal changes to the policy network without overstimating the state value, the targets of the twin critic are also use with minimal values to adjust both independent critic networks to ensure stability and avoid overstimating the values of states. The policy in SAC is deterministic and to ensure that the agent is able to explore, the SAC architecture implements Entropy Regularization which in terms forces the network to seek higher entropy, therefor increasing the noise (randomness) and allowing the agent to seek stochastic actions that can lead to exploration in the environment.

# Environment
robotic_arm

# Architecture
Soft Actor Critic

# optimizer
Policy: AdamW

Twin Critic: AdamW

# loss function
Policy: Difference of action distribution to state value loss

Value: smooth L1 loss function with combine twin critic loss

# Video Result:


https://github.com/Santiagor2230/Soft_Actor_Critic/assets/52907423/609a12c5-e14d-4c35-a41a-ba96bdbc93eb

