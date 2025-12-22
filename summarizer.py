# summarizer.py

from transformers import pipeline

# Make sure this class is defined directly in this file
class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initializes the summarizer with a pre-trained model.
        Args:
            model_name (str): The name of the pre-trained model to use.
                              Common choices: "facebook/bart-large-cnn", "sshleifer/distilbart-cnn-12-6",
                              "t5-small", "t5-base".
        """
        print(f"Loading summarization pipeline with model: {model_name}...")
        try:
            self.summarizer = pipeline("summarization", model=model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have an internet connection and the model name is correct.")
            self.summarizer = None # Handle cases where model loading fails

    def summarize_text(self, text, max_length=150, min_length=40, do_sample=False):
        """
        Generates a summary for the given text.

        Args:
            text (str): The input text to summarize.
            max_length (int): The maximum length of the generated summary.
            min_length (int): The minimum length of the generated summary.
            do_sample (bool): Whether to use sampling for generation (can increase creativity but less deterministic).

        Returns:
            str: The generated summary.
        """
        if not self.summarizer:
            return "Error: Summarization model not loaded."

        max_model_input_length = 1024 # Example for BART-based models like bart-large-cnn

        # Tokenize the input text to get its actual token length
        input_ids = self.summarizer.tokenizer.encode(text, return_tensors='pt')
        if len(input_ids[0]) > max_model_input_length:
            print(f"Warning: Input text is too long ({len(input_ids[0])} tokens). Truncating for summarization.")
            # Truncate input_ids to the max_model_input_length
            input_ids = input_ids[:, :max_model_input_length]
            # Decode back to text for consistent input to pipeline, though pipeline can handle input_ids
            # For simplicity, we'll let the pipeline handle truncation internally if needed
            # or pass the original text and trust the pipeline's internal truncation for now.
            # A more robust solution would be chunking, but for this example, the warning suffices.


        try:
            # The pipeline automatically handles tokenization and model input preparation
            summary = self.summarizer(
                text, # Pass the original text, pipeline handles internal truncation based on model limits
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample
            )[0]['summary_text']
            return summary
        except Exception as e:
            return f"An error occurred during summarization: {e}"

# This part is for testing summarizer.py directly, it does NOT interact with app.py
if __name__ == "__main__":
    summarizer = TextSummarizer()
    if summarizer.summarizer: # Only proceed if model loaded
        sample_text = """
        Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by animals and humans. Example tasks in which AI is used include speech recognition, computer vision, translation between natural languages, and other mappings of inputs.

        AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative AI (e.g., ChatGPT and other large language models), and competing at the highest level in strategic game systems (such as chess and Go).

        John McCarthy coined the term "artificial intelligence" in 1956. AI research was born out of the notion that human intelligence "can be so precisely described that a machine can be made to simulate it". This raises philosophical arguments about the nature of the mind and the ethics of creating artificial beings endowed with human-like intelligence, issues which have been explored by myth, fiction, and philosophy since antiquity.
        """
        summary = summarizer.summarize_text(sample_text, max_length=100, min_length=30)
        print("\nOriginal Text:")
        print(sample_text)
        print("\nGenerated Summary:")
        print(summary)