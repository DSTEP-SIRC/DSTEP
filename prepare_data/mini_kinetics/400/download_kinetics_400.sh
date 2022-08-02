mkdir /algo/CBI_inputs/Kinetics-400
cd /algo/CBI_inputs/Kinetics-400
while read one;
do
    echo $one
    wget "$one"
done < $1
