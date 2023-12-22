# from huggingface_hub import notebook_login

# notebook_login()
from datasets import load_dataset
from transformers import AutoTokenizer
squad = load_dataset("squad", split="train")
squad = squad.train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()
tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
training_args = TrainingArguments(
    output_dir="/home/alan_moreideas_ca/qlora/my_awesome_qa_model2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# from transformers import create_optimizer

# batch_size = 16
# num_epochs = 2
# total_train_steps = (len(tokenized_squad["train"]) // batch_size) * num_epochs
# optimizer, schedule = create_optimizer(
#     init_lr=2e-5,
#     num_warmup_steps=0,
#     num_train_steps=total_train_steps,
# )

# from transformers import TFAutoModelForQuestionAnswering

# model = TFAutoModelForQuestionAnswering("gpt2")