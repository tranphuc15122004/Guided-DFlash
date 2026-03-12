from datasets import load_dataset, Features, Sequence, Value

base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"

files = [
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
]

urls = [base + f for f in files]

dataset = load_dataset("json", data_files={"test": urls})["test"]

print("Dataset loaded:")
print(dataset)

# -------------------------
# Kiểm tra vài sample gốc
# -------------------------
print("\n===== RAW SAMPLES =====")
for i in range(3):
    print(f"\nSample {i}")
    print(dataset[i]["question_content"][:300])


def format_lcb(doc):
    system_prompt = (
        "You are an expert Python programmer. You will be given a question "
        "and will generate a correct Python program that matches the specification "
        "and passes all tests. You will NOT return anything except for the program."
    )

    question_block = f"### Question:\n{doc['question_content']}"

    if doc.get("starter_code"):
        format_message = "### Format: Use the following code structure:"
        code_block = f"```python\n{doc['starter_code']}\n```"
    else:
        format_message = "### Format: Write your code in the following format:"
        code_block = "```python\n# YOUR CODE HERE\n```"

    answer_footer = "### Answer: (use the provided format with backticks)"

    return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"


target_features = Features({
    "turns": Sequence(Value("string"))
})

dataset = dataset.map(
    lambda x: {"turns": [format_lcb(x)]},
    remove_columns=dataset.column_names,
    features=target_features
)

# -------------------------
# Kiểm tra sample sau khi format
# -------------------------
print("\n===== FORMATTED SAMPLES =====")
for i in range(3):
    print(f"\nFormatted Sample {i}")
    print(dataset[i]["turns"][0][:500])

# -------------------------
# Lưu dataset
# -------------------------
dataset.save_to_disk("livecodebench_dataset")

print("\nSaved dataset to disk")
print(dataset)
