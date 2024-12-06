# CS-4742-Term-Project
Term project for CS 4742 term project where we try and predicate a person's personality type based off of the way they speak dataset we are using for this term project can be found on kaggle: https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset?resource=download

# How to get Started


### How to Train the model - 

By Default the train_model.py will use an existing synthetic_personality_data.csv file if it exists in /data/ instead of overwriting it
run -
```
python train_model.py
```

### Arguments to put after - 

`--augment-synthetic-data` - 
Adds on to the existing data set in /data/synthetic_personality_data.csv .

`--target-count 10000'` - 
Overrides the target amount that it will generate, the default is whichever is the highest in the dataset.
Will generate 10000 lines per personality.

Example - 
```
python train_model.py --augment-synthetic-data --target-count 10000
```

`--overwrite-synthetic-data` - 
Overwrites the existing .csv file in /data/ and generates a new one - 

Example - 
```
python train_model.py --overwrite-synthetic-data
```

wait until it is finished training, it will save checkpoints in case something happens so not much progress is lost

  ## Run the flask app to start the dev server - 

Run - 
```
python app.py
```

### Open web browser - 

Goto - 
```
http://127.0.0.1:5000
```

### How to test the bot in the terminal - 

Make sure to do this in a seperate console than app.py is running in otherwise it won't work

Run - 
```
python model_test.py
```
