import torch
import PyPDF2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
import configparser
import os


config = configparser.ConfigParser()
config.read('config.properties')
model_name = config['DEFAULT']['model']


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def train_model_from_pdf(pdf_file_paths,  output_dir, num_train_epochs):
    # model_name="google/flan-t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    concatenated_text = ""
    for pdf_file_path in pdf_file_paths:
        concatenated_text += extract_text_from_pdf(pdf_file_path) + " "

    source_text = f"summarize: {concatenated_text}"

    inputs = tokenizer(source_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    train_dataset = Dataset.from_dict({
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": inputs["input_ids"].clone(),
        "labels": inputs["input_ids"].clone()
    })

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        num_train_epochs=num_train_epochs,
        save_total_limit=3,
        logging_steps=100,
        logging_dir="./logs",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(output_dir)

