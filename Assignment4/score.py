import joblib
from sentence_transformers import SentenceTransformer

class SentimentClassifier:
    """
    A sentiment classifier based on a trained model and sentence embeddings.

    Attributes:
        model_path (str): Path to the trained model file.
        model: The loaded trained model.
        sentence_encoder: Sentence transformer model for encoding text.
    """

    def __init__(self, model_path: str):
        """
        Initialize the SentimentClassifier.

        Args:
            model_path (str): Path to the trained model file.
        """
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def score_sentiment(self, text: str, threshold: float = 0.5) -> (bool, float):
        """
        Classify the sentiment of the input text.

        Args:
            text (str): The input text to be classified.
            threshold (float): Threshold for classification. Default is 0.5.

        Returns:
            tuple: A tuple containing a boolean indicating the predicted sentiment (True for positive, False for negative) and
            a float representing the propensity score for the predicted class.
        """
        try:
            # Transform the input text using the sentence encoder
            embedding = self.sentence_encoder.encode([text])
            
            # Predict the sentiment and propensity score for the input text
            prediction = self.model.predict(embedding)[0]
            propensity = self.model.predict_proba(embedding)[:, 1][0]
            
            return prediction, propensity
        except Exception as e:
            raise ValueError(f"Error scoring sentiment: {str(e)}")

if __name__ == "__main__":
    try:
        classifier = SentimentClassifier("Assignment2/Support_Vector_Machine.joblib")
        text = "This is a great product!"
        label, propensity = classifier.score_sentiment(text)
        sentiment = "Positive" if label else "Negative"
        print(f"The sentiment of '{text}' is {sentiment} with a propensity score of {propensity:.2f}")
    except ValueError as ve:
        print(ve)
