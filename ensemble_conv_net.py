import numpy as np # linear algebra
import pandas as pd
from subprocess import check_output
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

def process(train_file, test_file, output_file, val_split, kernel_window_size, drop_out_1, drop_out_2, drop_out_3, pool_size_1, pool_size_2, batch_size):	
	mnist_dataset = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')	
	n_raw = mnist_dataset.shape[0]
	n_val = int(n_raw * val_split + 0.5)
	n_train = n_raw - n_val

	np.random.shuffle(mnist_dataset)
	x_val, x_train = mnist_dataset[:n_val,1:], mnist_dataset[n_val:,1:]
	y_val, y_train = mnist_dataset[:n_val,0], mnist_dataset[n_val:,0]

	x_train = x_train.astype("float32")/255.0
	x_val = x_val.astype("float32")/255.0
	y_train = np_utils.to_categorical(y_train)
	y_val = np_utils.to_categorical(y_val)

	n_classes = y_train.shape[1]
	x_train = x_train.reshape(n_train, 28, 28, 1)
	x_val = x_val.reshape(n_val, 28, 28, 1)

	model = Sequential()

	model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (28, 28, 1), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(filters = 32, kernel_size = (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
	model.add(Activation('relu'))
	model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	datagen = ImageDataGenerator(zoom_range = 0.1,
								height_shift_range = 0.1,
								width_shift_range = 0.1,
								rotation_range = 20)
								
	model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics = ["accuracy"])
	callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto'),
				ModelCheckpoint('mnist.h5', monitor='val_loss', save_best_only=True, verbose=0)]
	hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 64),
							   steps_per_epoch = n_train/100, 
							   epochs = 100, 
							   verbose = 2,  
							   validation_data = (x_val, y_val),
							   callbacks = callbacks)
	mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')
	x_test = mnist_testset.astype("float32")/255.0
	n_test = x_test.shape[0]
	x_test = x_test.reshape(n_test, 28, 28, 1)
	y_test = model.predict(x_test, batch_size=64)
	y_index = np.argmax(y_test,axis=1)
	return y_index	
			
if __name__ == '__main__':
	df_out = pd.DataFrame()	
	num_models = 2
	#this bit will be replaced by jason file soon
	train_file = "input/train.csv"
	test_file = "input/test.csv"
	output_file = 'cnn_ensemble.csv'
	val_split = 0.1
	kernel_window_size = [3,5]
	drop_out_1 = [0.25,0.20]
	drop_out_2 = [0.25,0.20]
	drop_out_3 = [0.4,0.5]
	pool_size_1 = [2,2]
	pool_size_2 = [2,2]
	batch_size = [64,128]
	for i in range(0, num_models):
		y_index = process(train_file,test_file,output_file,val_split,kernel_window_size[i],drop_out_1[i],drop_out_2[i],drop_out_3[i],pool_size_1[i],pool_size_2[i],batch_size[i])
		df_out[i] = y_index
	df_out['mode'] = df_out.mode(axis = 1)
	
	with open(output_file, 'w') as f :
		f.write('TestId,Label\n')
		for i in range(0,y_index.shape[0]) :
			f.write("".join([str(i+1),',',str(df_out['mode'][i]),'\n']))
