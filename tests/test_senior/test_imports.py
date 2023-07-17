import pytest

class TestingImports:
    """
    Each Agent should be imported directly form its directory
    """
    def test_senior(self):
        """
        Test if the Senior import works
        """
        from curriculumagent.senior import Senior

    def test_junior(self):
        """
        Test if the junior import works
        """
        from curriculumagent.junior import Junior

    def test_tutor(self):
        """
        Test if the tutor import Works
        """
        from curriculumagent.tutor import Tutor
        from curriculumagent.tutor import NminusOneTutor


    def test_teacher(self):
        """
        Test if N-1 Teacher is imported
        """
        from curriculumagent.teacher import Teacher