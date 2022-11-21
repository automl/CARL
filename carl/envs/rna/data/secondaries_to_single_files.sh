DATAPATH=$1
SECONDARY_FILE=$2

NUM_SECONDARIES=$(cat $SECONDARY_FILE | wc -l )
i=1;
while [[ i -le $NUM_SECONDARIES ]];
	do awk "NR==$i{print;exit}" $SECONDARY_FILE > $DATAPATH/$i.rna;
	let i=$i+1;
done
