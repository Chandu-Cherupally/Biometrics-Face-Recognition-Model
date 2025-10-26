
Steps to run the run.py

Naive Method:
python run.py --dataset lfw_dataset_fixed --max_id 200 --method naive --low_res 24 --seed 123 --strict_naive --dump_ids chosen_ids_seed123_max200.txt --out_csv results_v2.csv --save_examples


Bicubic Probe (Low resolution test + Simple Upscaling)
python run.py --dataset lfw_dataset_fixed --max_id 200 --method bicubic_probe --low_res 24 --seed 123 --dump_ids chosen_ids_seed123_max200.txt --out_csv results_v2.csv --save_examples


Downsample gallery (Consistent Low resolution space)
python run.py --dataset lfw_dataset_fixed --max_id 200 --method downsample_gallery --low_res 24 --seed 123 --dump_ids chosen_ids_seed123_max200.txt --out_csv results_v2.csv --save_examples


SR Probe (Super Resolution Recovery)
python run.py --dataset lfw_dataset_fixed --max_id 200 --method sr_probe --low_res 24 --seed 123 --sr_model EDSR_x4.pb --sr_name edsr --sr_scale 4 --dump_ids chosen_ids_seed123_max200.txt --out_csv results_v2.csv --save_examples
