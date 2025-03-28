python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 1000 ../config/info_verona.ini verona
python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 1000 ../config/info_verona1.ini verona1
python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 1000 ../config/info_verona2.ini verona2
#python run_experiment.py -r --n_repeat 1 ../config/verona_ani.ini verona
python plot_experiments.py ../data/output/model/history_verona.zip ../data/output/model/history_verona1.zip ../data/output/model/history_verona2.zip --label_names verona,verona1,verona2 --out_file verona.png --column "I" 
geeqie verona.png
