clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.DS_Store' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} \;

pull-data:
	python src/data/make_dataset.py\
			--train data/train/train.json\
			--validation data/validation/validation.json\
			--test data/test/test.json\
			--dataset data\
		   	--max_workers 10\
			--max_images 1000\
			--resize 256

setup:
	mkdir -p data/train data/test data/validation
	cp ~/.kaggle/competitions/imaterialist-challenge-fashion-2018/train.json.zip data/
	cp ~/.kaggle/competitions/imaterialist-challenge-fashion-2018/validation.json.zip data/
	cp ~/.kaggle/competitions/imaterialist-challenge-fashion-2018/test.json.zip data/
	unzip 'data/*.zip' -d data/
	rm data/*.zip
	mv data/train.json data/train/train.json
	mv data/validation.json data/validation/validation.json
	mv data/test.json data/test/test.json
