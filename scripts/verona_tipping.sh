python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 100 ../config/verona_tipping.ini demo_verona_tipping
FILENAMES=""
LABELS=""
for FILE in `ls ../data/output/model/history_demo_verona_tipping*.zip`
do
    FILENAMES="$FILENAMES $FILE"
    LABEL=`basename $FILE .zip`
    LABELS="${LABELS},$LABEL"
done
LABELS=`echo $LABELS | sed 's/^,//g' | sed 's/history_demo_verona_tipping//g'`
echo $LABELS

python plot_experiments.py $FILENAMES --ymax 37 --label_names $LABELS --out_file verona_tipping.png --column "Active" 
geeqie verona_tipping.png
