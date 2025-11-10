.PHONY: normalize sft split report train test serve eval

normalize:
	python scripts/normalize_posts.py --inputs data/raw/sample.jsonl

sft:
	python scripts/build_sft_pairs.py --in_jsonl data/interim/all_posts.jsonl --out_jsonl data/processed/sft_all.jsonl

split:
	python scripts/split_train_val.py --in_jsonl data/processed/sft_all.jsonl --train_out data/processed/sft_train.jsonl --val_out data/processed/sft_val.jsonl

report:
	python scripts/quality_report.py --jsonl data/processed/sft_all.jsonl

train:
	python train/sft_lora_cpu.py

test:
	python scripts/smoke_test.py

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8000

eval:
	python eval/run_eval.py --suite all --endpoint http://localhost:8000/generate
