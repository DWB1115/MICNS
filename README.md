# MICNS

This repository (currently) contains Python3 code to reproduce main experiments of the method described in our NeurIPS'24 paper [Avoiding Undesired Futures with Minimal Cost in Non-Stationary Environments](https://www.lamda.nju.edu.cn/duwb/to%20appear.txt). Check this [poster](https://www.lamda.nju.edu.cn/duwb/MICNS-NeurIPS24/MICNS_poster.pdf) for quick introduction.


##  Organization
1. `Rh_Solver.py` realizes the rehearsal-learning framework that updates parameters, suggests decision actions and evaluate the quality of the suggested actions.
2. `main_market.py` is designed to implement experiments on synthetic data.
3. `main_bermuda.py` is designed to implement experiments on Bermuda data, which has been used in some related research.


## Code for comparative methods
The code for the comparative methods is available in their official implementations, including [rehearsal learning](https://www.lamda.nju.edu.cn/qint/publication/NIPS23_rehearsal/code.zip), [DDPG](https://github.com/ghliu/pytorch-ddpg), [PPO](https://github.com/nikhilbarhate99/PPO-PyTorch) and [SAC](https://github.com/pranz24/pytorch-soft-actor-critic).

---------------

If you have questions or comments about anything related to this work, please do not hesitate to contact [Wen-Bo Du](http://www.lamda.nju.edu.cn/duwb/).