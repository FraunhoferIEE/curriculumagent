Agent Architecture
==================

The agent is split into four sub agents which depend on each other with the goal to improve the final performance.
What follows is a description and an illustration on how the agents interact with each other.

.. figure:: /_static/illustration.png
    :alt: The four stages of the binbinchen agent.

    Overview of the binbinchen agent. Read from top left to bottom right.
    (Source: original binbinchen agent presentation from `binbinchen <https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution>`_)

The first two agents, the :doc:`agents/teacher` and the :doc:`agents/tutor` are greedy expert agents responsible
for generating a reduced actions space(RAS) and experience in form of ``(action, observation)`` pairs.

With the experience from the *Tutor*, the :doc:`agents/junior` is then going to train a neural network which tries to clone
the behaviour of the *Tutor*.

Finally the trained neural network of the *Junior* is used by senior as a starting point for training the
:doc:`agents/senior` which tries to choose actions less greedily and with consideration of the future.

Together with some expert rules, like reconnection of lines the *Senior* is then used as the final agent, reaching
the best performance. That agent is defined in :mod:`curriculumagent.submission.my_agent`.

Paper
==================
If you want a detailed explanation of each agent and their mechanisms, we again refer to the paper
`Managing power grids through topology actions: A comparative study between advanced rule-based and reinforcement learning agents <https://doi.org/10.1016/j.egyai.2023.100276>`_


Baseline and Pipeline overview
-----------------

Considering that the full `Teacher-Tutor-Junior-Senior` pipeline might seem a little bit excessive, we
provide within the package a baseline module, described in :doc:`baseline`. Here you can first initialize the Baseline
and either import and retrain an already existing model, or train all of the steps yourself.