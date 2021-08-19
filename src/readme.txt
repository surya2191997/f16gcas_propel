Dependencies:
1. Openai Spinningup, pytorch, numpy
2. csaf
3. Openai Gym
4. Jeevana's Code(smpolicysynth) - already included in the repo
5. pyOpt - already included in the repo

Overview:
-> smpolicy_propel.py contains the code for all the parts(training the neural policy, projecting trajectories from the neural policy into a state machine, and PROPEL using the projection operator)
-> smpolicy_propel_notebook.py is the same code in notebook format
-> ddpg_model.pth is the saved pytorch neural policy
-> f16_sm is the saved state machine obtained from propel

Directions to run:
python3 smpolicy_propel.py

