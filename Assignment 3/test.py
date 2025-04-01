import joblib
import numpy as np
import os
import requests
import time
import unittest

from app import app
from score import score

sent = "This is a test text"

class TestScoreFunction(unittest.TestCase):
    """Test cases for the score function."""

    def setUp(self):
        """Set up the test environment."""
        file_path = "C://Users/paulanwesha01/Documents/2_Sem4/Applied ML/Assignment 2/joblib_files/Support_Vector_Machine.joblib"
        self.mlp = joblib.load(file_path)
        self.threshold = 0.5

    def test_smoke(self):
        """Check if score function returns values properly."""
        label, prop = score(sent, self.mlp, self.threshold)
        self.assertIsNotNone(label)
        self.assertIsNotNone(prop)

    def test_format(self):
        """Check if the type of data meets certain requirements."""
        label, prop = score(sent, self.mlp, self.threshold)
        self.assertIsInstance(sent, str)
        self.assertIsInstance(self.threshold, float)
        self.assertIsInstance(label, np.int64)
        self.assertIsInstance(prop, np.float64)

    def test_pred_value(self):
        """Check if the label value is in {0,1}."""
        label, prop = score(sent, self.mlp, self.threshold)
        self.assertIn(label, [0, 1])

    def test_propensity_value(self):
        """Check if propensity lies in [0,1]."""
        label, prop = score(sent, self.mlp, self.threshold)
        self.assertGreaterEqual(prop, 0)
        self.assertLessEqual(prop, 1)

    def test_prop_test_0(self):
        """If threshold is 0, prediction becomes 1."""
        label, prop = score(sent, self.mlp, 0)
        self.assertEqual(label, 1)

    def test_prop_test_1(self):
        """If threshold is 1, prediction becomes 0."""
        label, prop = score(sent, self.mlp, 1)
        self.assertEqual(label, 0)

    def test_spam(self):
        """Test obvious spam."""
        label, prop = score("You have won a million dollars. Click on this link to redeem it.", self.mlp, self.threshold)
        self.assertEqual(label, 1)

    def test_ham(self):
        """Test obvious ham."""
        label, prop = score("Dogs are cute.", self.mlp, self.threshold)
        self.assertEqual(label, 0)
        

class TestFlaskApp(unittest.TestCase):
    """Test cases for the Flask application."""

    def setUp(self):
        """Set up the test environment."""
        pass

    def test_flask(self):
        """Test the Flask application."""
        # Launch the Flask app using os.system
        os.system('python app.py &')

        # Wait for the app to start up
        time.sleep(1)

        # Make a request to the endpoint
        response = requests.get('http://127.0.0.1:5000/')
        print(response.status_code)

        # Assert that the response is what we expect
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.text, str)

        # Shut down the Flask app using os.system
        os.system('kill $(lsof -t -i:5000)')

if __name__ == '__main__':
    unittest.main()
