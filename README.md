If you want to learn more about this project, please [read the full paper here (PDF)](CARLA_PPO_PAPER.pdf)

In a nutshell, we used PPO with two different reward functions:

$$\mathbf{R_1(s_t)}=r^{\text{cross}} + r^{\text{speed}} + r^{\text{collision}}$$

$$r^{\text{cross}} = \begin{cases}
    -a & \text{if } |\text{lane offset}| > 2 \\
    0 & \text{otherwise}
    \end{cases}$$

$$r^{\text{collision}} = - \text{collision}$$
$$r^{\text{speed}} = b (\text{speed}) + \text{movement bonus} + \text{acceleration bonus}$$

And,

$$\mathbf{R_2(s_t)}=r^{\text{risk}} + r^{\text{collision}} + r^{\text{waypoint progress}} + r^{\text{target progress}} $$

  $$r^{\text{risk}} =
    \begin{cases}
    -c|\text{lane offset}| & \text{if } |\text{lane offset}| > 0.1 \\
    0 & \text{otherwise}
    \end{cases}$$
    


While the Anxious reward structure encouraged safer behavior, it also made the agent too conservative.
Just making the agent "more anxious" about drifting wasn’t enough to solve the problem—getting
good driving behavior clearly needs more careful reward design or different algorithms altogether.

<img width="336" height="600" alt="image" src="https://github.com/user-attachments/assets/8f272d88-cbb0-4b43-8bb2-a49ca256757f" />

# CARLA PPO Repository Location

This repository is located within the CARLA simulator's PythonAPI directory structure. Specifically, after my fresh download on windows, it is found at:

    Z:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\CARLA_PPO

    
