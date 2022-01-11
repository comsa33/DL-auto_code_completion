# auto_code_complete v1.3

## purpose and usage
> `auto_code_complete` is a auto word-completetion program which allows you to customize it on your needs.
> the model for this program is a combined model of a deep-learning NLP(Natural Language Process) model structure called 'GRU(gated recurrent unit)' and 'LSTM(Long Short Term Memory)'.

> the model for this program is one of the deep-learning NLP(Natural Language Process) model structure called 'GRU(gated recurrent unit)'.

## data preprocessing
![data-preprocess](https://user-images.githubusercontent.com/61719257/148905258-e8a4195e-cb00-4493-94cb-b7982e6885d5.png)

## model structure
![model-structure](https://user-images.githubusercontent.com/61719257/148905156-6476fdcb-447b-4f34-8c11-c2f1159b3009.png)

## how to use (terminal)
![auto-code1](https://user-images.githubusercontent.com/61719257/148905376-389b7a14-cded-438c-b628-fc3d4e48e745.gif)
![auto-code2](https://user-images.githubusercontent.com/61719257/148905396-5168c558-06cd-4e7e-8b33-46b6bd159086.gif)
- first, download the repository on your local environment.
- install the neccessary libraries on your dependent environment.
> `pip install -r requirements.txt`
- change your working directory to  `auto-complete/` and execute the line below
> `python -m auto_complete_model`
- it will require for you to enter the data you want to train with the model 
```
ENTER THE CODE YOU WANT TO TRAIN IN YOUR MODEL : tensorflow tf.keras tf.keras.layers LSTM
==== TRAINING START ====
2022-01-08 18:24:14.308919: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
Epoch 1/100
3/3 [==============================] - 1s 59ms/step - loss: 4.7865 - acc: 0.0532
Epoch 2/100
3/3 [==============================] - 0s 62ms/step - loss: 3.9297 - acc: 0.2872
Epoch 3/100
3/3 [==============================] - 0s 58ms/step - loss: 2.9941 - acc: 0.5532
...
Epoch 31/100
3/3 [==============================] - 0s 75ms/step - loss: 0.2747 - acc: 0.8617
Epoch 32/100
3/3 [==============================] - 0s 65ms/step - loss: 0.2700 - acc: 0.8298
==== TRAINING DONE ====
Now, Load the best weights on your model.
```
- if you input your dataset successfully, it will ask for any uncompleted word to be entered.

```
ENTER THE UNCOMPLETED CODE YOU WANT TO COMPLETE : t tf te l la li k ke tf.kera tf.keras.l
t  - best recommendation : tensorflow
		 - all recommendations :  ['tensorflow']
tf  - best recommendation : tf.keras
		 - all recommendations :  ['tfkeras', 'tf.keras']
te  - best recommendation : tensorflow
		 - all recommendations :  ['tensorflow']
l  - best recommendation : list
		 - all recommendations :  ['list', 'layers']
la  - best recommendation : lange
		 - all recommendations :  ['layers', 'lange']
li  - best recommendation : list
		 - all recommendations :  ['list']
k  - best recommendation : keras
		 - all recommendations :  ['keras']
ke  - best recommendation : keras
		 - all recommendations :  ['keras']
tf.kera  - best recommendation : tf.keras
		 - all recommendations :  []
tf.keras.l  - best recommendation : tf.keras.layers
		 - all recommendations :  ['tf.keras.layers']
```
- it will return the best matched word to complete and other recommendations
```
Do you want to check only the recommendations? (y/n) : y
['tensorflow'], 
['tfkeras', 'tf.keras'], 
['tensorflow'], 
['list', 'layers'], 
['layers', 'lange'], 
['list'], 
['keras'], 
['keras'], 
[], 
['tf.keras.layers']
```

## version update & issues

### v1.2 update
2022.01.08
- change deep-learning model from GRU to GRU+LSTM to improve the performance
> By adding the same structrue of new LSTM layers to concatenate before the output layer to an existing model, it shows faster learning and better accuracies in predicting matched recommendations for given incomplete words. 

### v1.3.1 update
2022.01.09
- fix the glitches in data preprocessing
> We solved the problem that it wouldn't add a new dataset on an existing dataset.
- add `plot_history` function in a model class

### v1.3.2 update
2022.01.10
- add `model_save`,`model_load` mode in order that users can save and load their model while training a customized model
- add `data_split` mode so that the big data can be trained seperately.
```python
samp_model = auto_coding(new_code=samp_text,
                      # verbose=0,
                       batch_size=100,
                       epochs=200,
                       patience=10,
                       model_summary=True,
                       model_save=True,
                       model_name='samp_test', # samp_test/samp_test.h5
                       model_load=True,
                       data_split=True,
                       data_split_num=3 # the number into which users want to split the data
                      )
```

### v1.3.3 update
2022.01.11
- add new metrics `Accuracy for Recommendations` to evaluate the model's instant performance when predicting the recommendation list for words.
```
t  - best match : tf
	 - all recommendations :  ['tensorflow', 'tf']
tup  - best match : tuple
	 - all recommendations :  []
p  - best match : pd
	 - all recommendations :  ['plt', 'pd', 'pandas']
li  - best match : list
	 - all recommendations :  []
d  - best match : dataset
	 - all recommendations :  ['dic', 'dataset']
I  - best match : Import
	 - all recommendations :  []
so  - best match : sort
	 - all recommendations :  ['sort']
m  - best match : matplotlib.pyplot
	 - all recommendations :  []
Accuracy for Best:  0.875
Accuracy for Recommendations :  1.0
```
