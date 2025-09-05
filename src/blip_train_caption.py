import os, json, math, random
from dataclasses import dataclass
from typing import Dict, List, Any


import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


# Config
MODEL_ID = os.environ.get("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
DATA_DIR = os.environ.get("COCO_DIR", "data/coco")
OUTPUT_DIR = os.environ.get("OUT_DIR", "outputs/blip-caption")
SEED = int(os.environ.get("SEED", 42))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 30))


random.seed(SEED); torch.manual_seed(SEED)

def path_from_row(row: Dict[str, Any]) -> str:
    # HF dataset has fields: filename, filepath (train2014/val2014), split
    fn = row["filename"]
    fp = row.get("filepath", "train2014") # e.g., "train2014" | "val2014"
    return os.path.join(DATA_DIR, fp, fn)

class CocoKarpathyDataset(Dataset):
    def __init__(self, hf_split: str, processor):
        # hf_split in {"train", "validation", "test"}
        self.ds = load_dataset("yerevann/coco-karpathy", split=hf_split)
        self.processor = processor
        # expand to (image, caption) pairs
        items = []
        for row in self.ds:
            img_path = path_from_row(row)
        caps = row["sentences"] # list of 5 captions
        for c in caps:
            items.append({"image": img_path, "caption": c})
        self.items = items

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image"]).convert("RGB")
        encoding = self.processor(images=image, text=item["caption"], return_tensors="pt")
        # Flatten batch dim
        return {k: v.squeeze(0) for k, v in encoding.items()}
    
@dataclass
class DataCollator:
    processor: Any
    def __call__(self, features):
        batch = {}
        # images
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        # labels (input_ids)
        input_ids = [f["input_ids"] for f in features]
        attn = [f["attention_mask"] for f in features]
        batch.update(self.processor.pad(text=input_ids, padding=True, return_tensors="pt"))
        batch["pixel_values"] = pixel_values
        return batch
    
def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)


    train_ds = CocoKarpathyDataset("train", processor)
    val_ds = CocoKarpathyDataset("validation", processor)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        remove_unused_columns=False,
        report_to=[],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )


    collator = DataCollator(processor)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))


    # quick sanity: generate on a few val samples
    model.eval()
    examples = [val_ds[i] for i in range(3)]
    with torch.no_grad():
        for ex in examples:
            pixel_values = ex["pixel_values"].unsqueeze(0).to(model.device)
            out = model.generate(pixel_values=pixel_values, max_new_tokens=MAX_TOKENS)
            text = processor.batch_decode(out, skip_special_tokens=True)[0]
            print("\nCAPTION:", text)

if __name__ == "__main__":
    main()