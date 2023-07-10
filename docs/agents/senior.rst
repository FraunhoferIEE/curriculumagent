.. currentmodule:: curriculumagent.senior


Senior
=======

The *Senior* agent is the final stage of the Curriculum Agent and consist of a Deep Reinforcement Learning (DRL)
model based on the Proximal Policy Optimization (`PPO <https://arxiv.org/abs/1707.06347>`_.).
In the agent itself is trained with the same action set as the :mod:`~curriculumagent.tutor`, however
instead of a greedy approach the agents selects the actions based on the reward of the Grid2Op environment. Further,
in order to achieve a faster training result, the *Senior* agent receives the weights of the :doc:`junior`
agent for a warm start.

In the *Senior* agent is trained with the `(RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_ framework.
For the training, use the :mod:`~curriculumagent.senior.Senior` and run the train method. Afterwards, you
can save the model of either the last checkpoint or a selected one of your choice. After training the PPO, the model
only needs to be transferred to the submission directory in order to have a completed agent.
