.. currentmodule:: curriculumagent.baseline

L2RPN-Baseline
=======

The :mod:`curriculumagent.baseline.CurriculumAgent` is a simplified interface to run either a part or the full
`Teacher-Tutor-Junior-Senior` pipeline. This CurriculumAgent interface is also part of the L2RPN-Baseline and can be
found in the `l2rpn-baselines <https://github.com/rte-france/l2rpn-baselines>`_ repository.

To run the Curriculum agent first import the agent with: :mod:`from curriculumagent.baseline import CurriculumAgent`.
Afterwards, initialize the agent with your preferred Grid2Op Environment  :mod:`agent = CurriculumAgent(...)`.
Now you either have two options:

1. You can load a previously trained model and actions with :mod:`agent.load(...)` and then use the agent with
:mod:`agent.act(...)`. If you want to retrain the model on the same environment you can call :mod:`agent.train(...)`, e.g., when you have new
chronics.

2. Alternatively, if you don't have no model and/or actions you can train the full pipeline with
:mod:`agent.train_full_pipeline(...)`. This will generate both actions and after a while a new model.
Note that this might take some time.

After the training, you can create the output agent similar to the `MyAgent` method by calling
:mod:`agent.save(...)`. Further, if you want to create a submission folder, you can call
:mod:`agent.create_submission(...)`. Note that in the baseline folder, we created some the default
training results on the IEEE14 grid, called "l2rpn_case14_sandbox".