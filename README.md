# Multimodal Captioning & VQA — BLIP + LLaVA

End-to-end project for **image captioning** (fine-tuning BLIP) and **visual question answering** (evaluating LLaVA), with reproducible setup, standard datasets, official metrics, and a Gradio demo.

## Features
- 📦 Reproducible environment (Conda + pip)
- 🖼️ Data setup for **COCO 2014** (Karpathy splits) and **VQAv2**
- 🏋️ Captioning: fine-tune `Salesforce/blip-image-captioning-base`
- 📊 Evaluation: **CIDEr, BLEU, METEOR, ROUGE-L** via `pycocoevalcap`, optional **SPICE**
- ❓ VQA: run **LLaVA-1.5-7B** on VQAv2 with the official accuracy metric
- 🎛️ Gradio app for quick, local demos

## Repo Structure
```
multimodal-blip-llava/
├─ env/environment.yml
├─ data/
│  ├─ coco/{train2014,val2014,annotations,karpathy}
│  └─ vqav2/{images,questions,annotations}
├─ src/
│  ├─ blip_train_caption.py
│  ├─ blip_eval_caption.py
│  ├─ llava_vqa_eval.py
│  ├─ app_gradio.py
│  ├─ metrics_spice.py
│  └─ utils.py
├─ scripts/
│  ├─ download_coco_2014.sh
│  ├─ link_vqav2_images.sh
│  └─ prepare_karpathy_cache.py
└─ README.md
```

## Quickstart

### 1) Environment
```bash
conda env create -f env/environment.yml
conda activate mm-visionlang
python -m nltk.downloader punkt omw-1.4 wordnet
```

### 2) Data

* **COCO 2014** images + captions: [https://cocodataset.org/](https://cocodataset.org/)
* **Karpathy splits** are loaded automatically from the HF dataset in the training script.
* **VQAv2** questions/annotations: [https://visualqa.org/download.html](https://visualqa.org/download.html)
  Place files under `data/vqav2/questions/` and `data/vqav2/annotations/`, then symlink images:

```bash
bash scripts/download_coco_2014.sh
bash scripts/link_vqav2_images.sh
```

### 3) Train BLIP (Captioning)

```bash
python -u src/blip_train_caption.py OUT_DIR=outputs/blip-caption
# Model used: https://huggingface.co/Salesforce/blip-image-captioning-base
```

### 4) Captioning Evaluation (COCO Karpathy test split)

```bash
python -u src/blip_eval_caption.py SPLIT=test OUT_JSON=outputs/coco_caps_test.json
```

* Uses **pycocoevalcap** for CIDEr/BLEU/METEOR/ROUGE-L.
* (Optional) **SPICE**: install Java ≥8 and place `SPICE-1.0.jar` under `tools/spice/`, then run `metrics_spice.py`.

### 5) VQA Evaluation (LLaVA on VQAv2)

```bash
# Official VQA API
git clone https://github.com/GT-Vision-Lab/VQA.git tools/VQA
pip install -e tools/VQA/PythonHelperTools

# Evaluate (subset first)
python -u src/llava_vqa_eval.py N_SAMPLES=500 \
  LLAVA_MODEL=llava-hf/llava-1.5-7b-hf \
  VQA_QUESTIONS=data/vqav2/questions/v2_OpenEnded_mscoco_val2014_questions.json \
  VQA_ANN=data/vqav2/annotations/v2_mscoco_val2014_annotations.json \
  VQA_IM_ROOT=data/vqav2/images
```

### 6) Demo App (Gradio)

```bash
python -u src/app_gradio.py
```

* Caption an image with BLIP or ask LLaVA a question about it.
* Default models are configurable in `app_gradio.py`.

## Expected Results (placeholders)

| Task       | Metric                 | Split         | Result            |
| ---------- | ---------------------- | ------------- | ----------------- |
| Captioning | CIDEr / BLEU-4 / SPICE | Karpathy test | ~130, ~38, ~21    |
| VQA        | Accuracy (official)    | VQAv2 val     | ~78%              |

> Tips: if VRAM is tight, reduce `per_device_train_batch_size` or use gradient accumulation. LLaVA inference can run in 4-bit on \~8–12 GB VRAM (slower on CPU).

## References & Useful Links

* **BLIP paper**: [https://arxiv.org/abs/2201.12086](https://arxiv.org/abs/2201.12086)
* **LLaVA paper**: [https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)
* **COCO dataset**: [https://cocodataset.org/](https://cocodataset.org/)
* **VQAv2 dataset & evaluation**: [https://visualqa.org/](https://visualqa.org/)
* **pycocoevalcap** (COCO caption metrics): [https://github.com/salaniz/pycocoevalcap](https://github.com/salaniz/pycocoevalcap)
* **SPICE metric**: [https://github.com/peteanderson80/SPICE](https://github.com/peteanderson80/SPICE)
* **Transformers docs**: BLIP [https://huggingface.co/docs/transformers/model\_doc/blip](https://huggingface.co/docs/transformers/model_doc/blip), LLaVA [https://huggingface.co/docs/transformers/en/model\_doc/llava](https://huggingface.co/docs/transformers/en/model_doc/llava)
* **LLaVA-1.5-7B (HF)**: [https://huggingface.co/llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)

## License and Dataset Terms

* Check licenses/terms for **COCO** and **VQAv2** before use.
* Review model cards/licenses for BLIP and LLaVA; usage may be restricted in commercial settings.

## Acknowledgments

* BLIP by Salesforce Research
* LLaVA by Liu et al.
* COCO Consortium, VQA team, and the maintainers of `pycocoevalcap`
