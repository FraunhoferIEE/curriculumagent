Agent Architecture
==================

The agent is split into four sub agents which depend on each other with the goal to improve the final performance.
What follows is a description and an illustration on how the agents interact with each other.

.. figure:: /_static/illustration.png
    :alt: The four stages of the binbinchen agent.

    Overview of the binbinchen agent. Read from top left to bottom right.
    (Source: original binbinchen agent presentation)

The first two agents, the :doc:`agents/teacher` and the :doc:`agents/tutor` are greedy expert agents responsible
for generating a reduced actions space(RAS) and experience in form of ``(action, observation)`` pairs.

With the experience from the *Tutor*, the :doc:`agents/junior` is then going to train a neural network which tries to clone
the behaviour of the *Tutor*.

Finally the trained neural network of the *Junior* is used by senior as a starting point for training the
:doc:`agents/senior` which tries to choose actions less greedily and with consideration of the future.

Together with some expert rules, like reconnection of lines the *Senior* is then used as the final agent, reaching
the best performance. That agent is defined in :mod:`curriculumagent.submission.my_agent`.

Pipeline overview
-----------------

To produce the final agent all these modules have to work together and produce intermediary results.
The following image outlines the structure of the agent:

.. figure:: /_static/pipeline.png
    :alt: The architecture of the pipeline.

    The different modules of the agent being executued, producing different intermediary files to reach
    the end result.
