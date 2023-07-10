.. currentmodule:: curriculumagent.tutor

Tutor
=====

The *Tutor* uses the reduced space by the :doc:`teacher` to generate experience, usable by the :doc:`junior`.
It acts very similar to the :ref:`simple greedy teacher <SimpleGreedyTeacher>`,
by looping through all the actions in the reduced action space and choosing the one with the best load reduction.

It is contained in the :mod:`~curriculumagent.tutor` module and consists of two implementations, the
:class:`tutors.original_tutor.Tutor` and :class:`tutors.general_tutor.GeneralTutor` class. That class is then being used by
the :mod:`~curriculumagent.tutor.collect_tutor_experience` to generate action, observation pairs for the :doc:`junior`.

Further it is possible to sequentially test different action sets.
In the example of the NeurIPS 2020 challenge, the original agent by binbinchen used an actionspace of 208 unitary
actions, which were split into two subsets of 62 and 146 actions. Instead of running through all of them equally, the tutor started by only considering the first 62 actions and stopped
when a good action was found that reduced the max load/rho below 0.99. Only when those 62 actions failed in that task it took a look and simulated the outcome of the 146 actions.


While the tutor implementation from above only handles unitary actions properly, the :mod:`~curriculumagent.tutor.tutors.general_tutor`
is aimed to also work with various action sets, including tuple and triple actions. For this feature one just passes
multiple action sets into the :class:`tutors.general_tutor.GeneralTutor`. The actions are then iteratively analyzed and in
case of the tuple and triple actions split and executed sequentially.

Usage
-----

Execute the tutor search by either running :mod:`~curriculumagent.tutor.tutors.general_tutor` or
:mod:`~curriculumagent.tutor.tutors.original_tutor`. One major argument regarding the computational impact is the num_chronics parameter,
which specifies the chroncis for the search and therefore also impacts the quality of the tutor experience.
After running the tutor search, one has to execute
:func:`curriculumagent.tutor.collect_tutor_experience.prepare_default_junior_dataset`
in order to prepare the dataset for the :doc:`junior` agent.
