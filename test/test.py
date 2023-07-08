import torch
import unittest
from transformers import TrainingArguments, Trainer
from LongNet.model import LongNetTokenizer, LongNet

class TestLongNetModels(unittest.TestCase):
    def setUp(self):
        self.model = LongNet()
        self.tokenizer = LongNetTokenizer()
        self.training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=10_000,
            save_total_limit=2,
            logging_steps=500,
            logging_dir='./test_logs',
        )

    def test_language_model(self):
        model = self.model
        train_dataset = self.get_sample_dataset()
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def get_sample_dataset(self):
        # generate a simple dataset for testing
        data = {'target_text': ["This is a test sentence."] * 10}

        # Tokenize dataset
        inputs = self.tokenizer.tokenize(data)
        return inputs

if __name__ == "__main__":
    unittest.main()
