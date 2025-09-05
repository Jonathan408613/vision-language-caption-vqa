setup:
	conda env create -f env/environment.yml
	conda activate mm-visionlang
	python -m nltk.downloader punkt omw-1.4 wordnet


data-coco:
	bash scripts/download_coco_2014.sh


vqaimages:
	bash scripts/link_vqav2_images.sh


train-blip:
	python -u src/blip_train_caption.py OUT_DIR=outputs/blip-caption


eval-coco:
	python -u src/blip_eval_caption.py SPLIT=test OUT_JSON=outputs/coco_caps_test.json


vqa-eval:
	python -u src/llava_vqa_eval.py N_SAMPLES=500