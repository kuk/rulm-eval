
Dev env

```bash

python -m venv ~/.venvs/rulm-eval
source ~/.venvs/rulm-eval/bin/activate

pip install -r requirements.txt

pip install ipykernel
python -m ipykernel install --user --name rulm-eval
```

Data

```
mkdir data

# rsg
wget --no-check-certificate https://russiansuperglue.com/tasks/download
unzip download
rm -r download __MACOSX
mv combined data/rsg

# rucola
mkdir data/rucola
wget https://github.com/RussianNLP/RuCoLA/raw/main/data/in_domain_dev.csv -P data/rucola
wget https://github.com/RussianNLP/RuCoLA/raw/main/data/out_of_domain_dev.csv -P data/rucola
```
