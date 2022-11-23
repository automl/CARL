cd data/

mkdir rfam_taneda
cd rfam_taneda

wget rna.eit.hirosaki-u.ac.jp/modena/v0028/linux/modena.dataset.tar.gz

tar -xf modena.dataset.tar.gz

rm -f modena.dataset.tar.gz
rm -rf ct_version

i=1
while [[ i -le 30 ]]; do
    if  [ $i == 23 ]; then
	:
    elif [[ $i -le 9 ]]; then
        cat RF0000$i* > $i.rna;
    elif [[ $i -le 22 ]]; then
        cat RF000$i* > $i.rna;
    else
	cat RF000$i* > $(($i - 1)).rna;
    fi
    let i=$i+1;
done

rm -f *.ss
