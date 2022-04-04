# auto_code_complete v1.3

## Purpose and Usage
> `auto_code_complete` is a deep-learning based auto word-completetion program which allows you to customize it on your needs.
> the model for this program is a combined model of a deep-learning NLP(Natural Language Process) model structure called 'GRU(gated recurrent unit)' and 'LSTM(Long Short Term Memory)'.

> the model for this program is one of the deep-learning NLP(Natural Language Process) model structure called 'GRU(gated recurrent unit)'.

## Data Preprocessing
![data-preprocess](https://user-images.githubusercontent.com/61719257/148905258-e8a4195e-cb00-4493-94cb-b7982e6885d5.png)

## Model Structure
![model-structure](https://user-images.githubusercontent.com/61719257/148905156-6476fdcb-447b-4f34-8c11-c2f1159b3009.png)

## Evaluations
> 입력된 데이터에 따라 모델 output layer 에서의 target class 가 천차만별이다.
> 
> 
> 아주 적은 몇줄의 코드를 훈련시킬 경우에는 100~500 개 사이의 클래스를 분류해야하고, 200줄 가량의 (파일 하나정도 분량) 코드는 4000~6000 개 사이의 클래스를 분류해야한다. 혹은 tensorflow 와 같은 라이브러리를 통째로 주요 명령어들만 뽑아 훈련시킬 경우라고 하더라도 최소 12만개 이상의 클래스를 분류해야하기 때문에 단순한 accuracy 만을 주요 평가지표로 고려하기에는 모델의 성능을 완벽하게 설명하기 어렵다고 판단했다.
> 
> 또한, 훈련셋과 검증, 테스트 셋으로 나누어 모델성능을 확인하는 일반적인 경우와 다르게 새로운 코드를 생성하는 것이 주요 목적이 아닌 이미 기존에 학습된 코드들을 얼마나 정확하고 다양하게 추천해줄 수 있는 기능을 기대하고 있기 때문에 과적합에 대한 부분은 우려대상에서 제외되었다.
> 
> 추가로 바로 앞서 언급한 모델의 주요목적 때문에 검증셋과 테스트셋이 결국 훈련셋과 동일하게 사용되었다.
> 

### a. outputs from the output layer

최종 출력층을 거쳐 softmax 함수까지 처리를 마친 outputs들 중 일정 threshold = 0.025 이상의 결과들만 추천하도록 했다.

그렇게 함으로써 최종적인 결과는 가장 높은 softmax 값의 결과 + softmax 결과 > 0.025 이상의 결과로 분류해서 추천하게 된다. 즉 입력값에 대한 the best match 와 recommendations 로 결과를 보여주는 것이다.

### b. Accuracy For Recommendations

앞서 언급한 것처럼 기존의 `accuracy` 만으로는 정확히 프로그램의 성능을 설명하기 어렵기 때문에 `AFR(Accuracy for Recommendations : 추천 정확도)` 를 사용하여 모델을 평가했다.

이 새로운 평가지표는 훈련을 마친 모델에 테스트 데이터를 입력했을때 나온 결과로 최종 평가를 하도록 하였다.

> 추천된 목록 중 실제 훈련된 코드(진짜 존재하는 코드)와 똑같은 단어나 문구가 얼마나 있는지를 측정하는 방법
> 
- AFR 공식은 아래와 같이 계산한다.
    
    > <img width="81" alt="Screen Shot 2022-04-04 at 10 43 59" src="https://user-images.githubusercontent.com/61719257/161460584-a24fbaa8-f918-489f-98b7-0e4c8ce4d734.png">
    
    
    > N_true : 추천 목록에 있는 코등 중에서 실제 존재하는 코드의 수 
    > 
    > N_pred : 입력 당 예측된 추천 목록에 있는 총 코드의 개수
    > 
    > N : 총 테스트 데이터의 수

|DATA		|Accuracy for Best	|Accuracy for Recommendations	|
|---------------|-----------------------|-------------------------------|
|Custom Small Data|0.875|1.0|
|Big Data(TF open source)|0.58|0.79|

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
