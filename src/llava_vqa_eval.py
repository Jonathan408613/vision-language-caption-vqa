import os, json, re
from typing import Dict
import torch
from PIL import Image
from tqdm import tqdm


from transformers import LlavaForConditionalGeneration, AutoProcessor


# Inputs
MODEL_ID = os.environ.get("LLAVA_MODEL", "llava-hf/llava-1.5-7b-hf")
VQA_Q = os.environ.get("VQA_QUESTIONS", "data/vqav2/questions/v2_OpenEnded_mscoco_val2014_questions.json")
VQA_ANN = os.environ.get("VQA_ANN", "data/vqav2/annotations/v2_mscoco_val2014_annotations.json")
IM_ROOT = os.environ.get("VQA_IM_ROOT", "data/vqav2/images")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 24))
N_SAMPLES = int(os.environ.get("N_SAMPLES", 500)) # for quick eval; set to -1 for full


# load official VQA API
from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

# Normalize answer like VQA eval (fallback)
ARTICLES = {"a", "an", "the"}
PUNCS = ";./,\'[]\`~!@#$%^&*()_+-=:{|}?<>\""


def normalize_ans(ans: str) -> str:
    a = ans.lower().strip()
    a = re.sub(f"[{re.escape(PUNCS)}]", "", a)
    a = " ".join([w for w in a.split() if w not in ARTICLES])
    return a

def q_to_path(img_id: int) -> str:
    # e.g., 458752 -> COCO_val2014_000000458752.jpg
    fn = f"COCO_val2014_{img_id:012d}.jpg" if os.path.exists(os.path.join(IM_ROOT, "val2014")) else f"COCO_train2014_{img_id:012d}.jpg"
    sub = "val2014" if "val2014" in fn else "train2014"
    return os.path.join(IM_ROOT, sub, fn)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        low_cpu_mem_usage=True, load_in_4bit=(device=="cuda"), device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)


    vqa = VQA(VQA_ANN, VQA_Q)
    qids = vqa.getQuesIds()
    if N_SAMPLES > 0:
        qids = qids[:N_SAMPLES]

    results = []
    for qid in tqdm(qids):
        q = vqa.loadQA(qid)[0]
        img_path = q_to_path(q["image_id"])
        image = Image.open(img_path).convert("RGB")
        prompt = f"USER: <image>\n{q['question']}\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        ans = processor.decode(out[0], skip_special_tokens=True)
        ans = ans.split("ASSISTANT:")[-1].strip()
        results.append({"question_id": int(qid), "answer": normalize_ans(ans)})

    os.makedirs("outputs", exist_ok=True)
    res_path = "outputs/vqav2_llava_results.json"
    with open(res_path, "w") as f:
        json.dump(results, f)
    print("Saved:", res_path)

    # Evaluate with official metric
    vqaRes = vqa.loadRes(res_path, VQA_Q)
    vqaEval = VQAEval(vqa, vqaRes, n=2) # n=2 per official default
    vqaEval.evaluate()
    print("Overall Accuracy: %.2f" % (vqaEval.accuracy["overall"]))

if __name__ == "__main__":
    main()