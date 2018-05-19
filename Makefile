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
		   	--max_workers 1\
			--max_images 100\
			--resize 256
