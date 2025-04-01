CONFIG=$1
EXP_ID=$2
COLUMN=$3

python run_multi_experiment.py -R ../config/random_seeds.txt --n_jobs 4 --n_repeat 100 ${CONFIG} ${EXP_ID}

FILENAMES=""
LABELS=""
for FILE in `ls ../data/output/model/history_${EXP_ID}_*.zip`
do
    FILENAMES="$FILENAMES $FILE"
    LABEL=`basename $FILE .zip`
    LABELS="${LABELS},$LABEL"
done
LABELS=`echo $LABELS | sed 's/^,//g' | eval sed 's/history_${EXP_ID}//g'`

python plot_experiments.py $FILENAMES --label_names $LABELS --out_file ${EXP_ID}.png --column ${COLUMN} 
#geeqie ${EXP_ID}.png
