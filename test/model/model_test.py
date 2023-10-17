# from longnet.model import LongNetTokenizer, LongNetSelector
import torch

# from model import LongNetTokenizer,
from longnet.model import LongNetTokenizer, LongNet


class LongNetTest:
    def __init__(self):
        self.longnet_selector = LongNet()
        self.tokenizer = LongNetTokenizer()

    def run_test(self, model_type: str):
        data = {
            "target_text": ["This is a test sentence."] * 2,
            "image": torch.rand(2, 3, 224, 224),  # 2 random images
        }

        inputs = self.tokenizer.tokenize(data)

        if model_type.lower() == "multimodal":
            self._test_model("multimodal", inputs)
        elif model_type.lower() == "language":
            self._test_model("language", inputs)
        else:
            raise ValueError(
                f"Invalid model_type: {model_type}. Please use either 'multimodal' or 'language'."
            )

    def _test_model(self, model_type: str, inputs: dict):
        print(f"Testing {model_type} LongNet model...")
        model = self.longnet_selector.get_model(model_type)
        model(**inputs)
        print(f"{model_type} LongNet model forward pass succeeded!")


# # Now you can use the class like this:
# tester = LongNetTest()
# tester.run_test('multimodal')
# tester.run_test('language')
