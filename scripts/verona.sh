python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 1000 ../config/info_verona.ini demo_verona
FILENAMES=""
LABELS=""
for FILE in `ls ../data/output/model/history_demo_verona_*.zip`
do
    FILENAMES="$FILENAMES $FILE"
    LABEL=`basename $FILE .zip`
    LABELS="${LABELS},$LABEL"
done
LABELS=`echo $LABELS | sed 's/^,//g' | sed 's/history_demo_verona_//g'`

python plot_experiments.py $FILENAMES --label_names $LABELS --out_file verona.png --column "I" 
geeqie verona.png
