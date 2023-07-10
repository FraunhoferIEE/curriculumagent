.. currentmodule:: curriculumagent.junior

Junior
=======

The *Junior* sub agent is the first deep learning method in CurriculumAgent pipeline with the goal
to mimic the actions of the greedy :doc:`tutor` agent. The purpose of this agent is to fit a sequential
neural network, i.e., the weights of the network, on the input data of the Grid2Op environment.
After a successful training, the weights are then used for the :doc:`senior` in order to warm start the Deep
Reinforcement Learning approach. Accordingly, the *Junior* sub agent plays a vital role in the
curriculum approach.

Usage
-----

Overall, the agent is trained on the experience of the :doc:`tutor` agent. For this reason, the experience output of
the :mod:`~curriculumagent.tutor.collect_tutor_experience` is first separated into a training, validation and test set via
the :func:`~curriculumagent.junior.junior_student.load_dataset`. Thereafter, the junior class :class:`~curriculumagent.junior.junior_student.Junior`
can be used for training and evaluating the deep learning model.

Under consideration that the parameters of the junior agent are in many cases similar, one can alternatively just run
the :meth:`~curriculumagent.junior.junior_student.Junior.train` method, which combines the collection and the training.
Note that if the :mod:`~curriculumagent.tutor.tutors.general_tutor` was used with multiple action sets, one has to provide
these sets for the junior as well.

Structure of the  Junior Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The *Junior* sub agent is based on a `Tensorflow (Keras) <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_
sequential model and has the following structure:

.. include:: /_static/report.txt
    :literal:
