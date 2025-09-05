import os, json, glob
from typing import Dict, List
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration


# COCO eval
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


# Optional SPICE (Java needed). See metrics_spice.py for a safer wrapper.


MODEL_DIR = os.environ.get("MODEL_DIR", "outputs/blip-caption/final")
COCO_DIR = os.environ.get("COCO_DIR", "data/coco")
OUT_JSON = os.environ.get("OUT_JSON", "outputs/coco_captions_predictions.json")
SPLIT = os.environ.get("SPLIT", "test") # Karpathy: test|validation
MAX_NEW_TOKENS = int(os.environ.get("MAX_TOKENS", 30))

def path_from_row(row):
    fn = row["filename"]
    fp = row.get("filepath", "val2014")
    return os.path.join(COCO_DIR, fp, fn)

def generate_predictions():
    ds = load_dataset("yerevann/coco-karpathy", split=SPLIT)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds = []
    for row in tqdm(ds, total=len(ds)):
        img = Image.open(path_from_row(row)).convert("RGB")
        enc = processor(images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS)
            caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            preds.append({"image_id": int(row["cocoid"]), "caption": caption})

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(preds, f)
    print(f"Wrote predictions -> {OUT_JSON}")

def evaluate_with_coco():
    ann_path = os.path.join(COCO_DIR, "annotations", "captions_val2014.json")
    coco = COCO(ann_path)
    cocoRes = coco.loadRes(OUT_JSON)
    cocoEval = COCOEvalCap(coco, cocoRes)

    # Use only images from the Karpathy split selected
    ds = load_dataset("yerevann/coco-karpathy", split=SPLIT)
    img_ids = [int(r["cocoid"]) for r in ds]
    cocoEval.params["image_id"] = img_ids


    cocoEval.evaluate()
    print("\n=== COCO Captioning metrics ===")
    for metric, score in cocoEval.eval.items():
        print(f"{metric}: {score:.3f}")

if __name__ == "__main__":
    if not os.path.exists(OUT_JSON):
        generate_predictions()
    evaluate_with_coco()