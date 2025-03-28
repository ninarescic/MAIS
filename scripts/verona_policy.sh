python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 100  ../config/verona_spreader.ini spreader
python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 100  ../config/verona_spreader1.ini spreader1

python plot_experiments.py ../data/output/model/history_spreader.zip ../data/output/model/history_spreader1.zip  --label_names "q=1.0,q=0.2" --out_file verona_policy.png --column "I" --title "different spreder's centralities"  
geeqie verona_policy.png
