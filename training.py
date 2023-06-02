import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 데이터 준비
dataset = TextDataset(tokenizer=GPT2Tokenizer.from_pretrained("gpt2"), file_path= r"C:\Users\lego8\capstone_rta\uploaded_files\2000.txt", block_size=1024)
data_collator = DataCollatorForLanguageModeling(tokenizer=GPT2Tokenizer.from_pretrained("gpt2"), mlm=False)

# 모델 초기화
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 학습 설정
training_args = TrainingArguments(
    output_dir="./gpt2_training",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Trainer 객체 생성 및 학습 실행
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
