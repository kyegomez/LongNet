# from LongNet.model import LongNetTokenizer, LongNetSelector

from model import LongNetTokenizer, LongNetSelector

longnet_selector = LongNetSelector()
tokenizer = LongNetTokenizer()


data = {
    "target_text": ["This is a test sequence"] * 2,
    # 'image': torch.rand(2, 3, 224, 224) # 2 random images
}

#tokenize the inputs
inputs = tokenizer.tokenize(data)

# #test multo-modal model
# print(f"TTesting multi-modal longnet modal")
# model = longnet_selector.get_model("multimodal")
# outputs = model(**inputs)

#test language only longnet model
print("Testing language only model")
model = longnet_selector.get_model("language")
outputs = model(**inputs)


print("Langauge only longnet forward pass succeeed!")