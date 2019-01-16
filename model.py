import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import wget
import zipfile
import csv

local = 0
download = 0
arquitecture = 1

def get_array_Images(name_path,images,measurements):

    lines = []

    if download ==1:
        with zipfile.ZipFile(name_path+".zip", 'r') as zip_ref:
            zip_ref.extractall("./")
        print("Finish Unziping")
     
    with open ('./'+ name_path +'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
            source_path = line[0]
            filename = source_path.split("\\")[-1]
            current_path = './'+ name_path +'/IMG/' + filename
            image=mpimg.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)

    return images, measurements



if download ==1:
    wget.download("https://docs.google.com/uc?export=download&id=1KIVaoiAmXkuRSBusvfKfgYZxcXHhg6VP")
    wget.download("https://docs.google.com/uc?export=download&id=14yRvt5VkfBjaMFss7IK5Ew12RpBbJTS1")
    wget.download("https://docs.google.com/uc?export=download&id=1zBGIyYXSxlbmB8LJrWjavKBQaX_3ctYI")
    print("Finish Downloading")

images = []
measurements = []
images,measurements = get_array_Images('first_track_1_lap_forward',images,measurements)
images,measurements = get_array_Images('first_track_1_lap_clockwise',images,measurements)
images,measurements = get_array_Images('first_track_1_lap_forward_2nd',images,measurements)

print("Finish building images array")

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)

if local ==1:
    plt.imshow(images[len(images)-1])
    plt.show()
else:

    print("Start Training")

    from keras.layers import Dense, Lambda, Cropping2D

    if arquitecture == 1:
        from keras.models import Sequential
        from keras.layers import Flatten, Dropout
        from keras.layers import Conv2D

        model = Sequential()
        #Preprocessing
        model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((70,25),(0,0))))

        #Arquitecture
        model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
        model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
        model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
        model.add(Conv2D(64,(3,3),activation="relu"))
        model.add(Conv2D(64,(3,3),activation="relu"))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))

    elif arquitecture == 2:

        from keras.applications.inception_v3 import InceptionV3
        from keras.layers import Input, GlobalAveragePooling2D
        from keras.models import Model
        # Using Inception with ImageNet pre-trained weights
        weights_flag = 'imagenet'
        inception = InceptionV3(weights=weights_flag, include_top=False,input_shape=(160,320,3))
        for layer in inception.layers:
            layer.trainable = True

        # Makes the input placeholder layer 32x32x3 for CIFAR-10
        cifar_input = Input(shape=(160,320,3))

        #normalizing the image to have values between -1 an 1 with a mean of 0
        normalize = Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3))(cifar_input)

        inp = inception(normalize)

        new_layer_1 = GlobalAveragePooling2D()(inp)

        new_layer_2 = Dense(512, activation = 'relu')(new_layer_1)
        new_layer_3 = Dense(10, activation = 'relu')(new_layer_2)
        predictions = Dense(1, activation = 'softmax')(new_layer_3)

        model = Model(inputs=cifar_input, outputs=predictions)
    
    #Trainning
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
    print("Finish Training")

    model.save('model.h5')
    print("Saved model")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import wget
import zipfile
import csv

local = 0
download = 0
arquitecture = 1

def get_array_Images(name_path,images,measurements):

    lines = []

    if download ==1:
        with zipfile.ZipFile(name_path+".zip", 'r') as zip_ref:
            zip_ref.extractall("./")
        print("Finish Unziping")
     
    with open ('./'+ name_path +'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
            source_path = line[0]
            filename = source_path.split("\\")[-1]
            current_path = './'+ name_path +'/IMG/' + filename
            image=mpimg.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)

    return images, measurements



if download ==1:
    wget.download("https://docs.google.com/uc?export=download&id=1KIVaoiAmXkuRSBusvfKfgYZxcXHhg6VP")
    wget.download("https://docs.google.com/uc?export=download&id=14yRvt5VkfBjaMFss7IK5Ew12RpBbJTS1")
    wget.download("https://docs.google.com/uc?export=download&id=1zBGIyYXSxlbmB8LJrWjavKBQaX_3ctYI")
    print("Finish Downloading")

images = []
measurements = []
images,measurements = get_array_Images('first_track_1_lap_forward',images,measurements)
images,measurements = get_array_Images('first_track_1_lap_clockwise',images,measurements)
images,measurements = get_array_Images('first_track_1_lap_forward_2nd',images,measurements)

print("Finish building images array")

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)

if local ==1:
    plt.imshow(images[len(images)-1])
    plt.show()
else:

    print("Start Training")

    from keras.layers import Dense, Lambda, Cropping2D

    if arquitecture == 1:
        from keras.models import Sequential
        from keras.layers import Flatten, Dropout
        from keras.layers import Conv2D

        model = Sequential()
        #Preprocessing
        model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((70,25),(0,0))))

        #Arquitecture
        model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
        model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
        model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
        model.add(Conv2D(64,(3,3),activation="relu"))
        model.add(Conv2D(64,(3,3),activation="relu"))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))

    elif arquitecture == 2:

        from keras.applications.inception_v3 import InceptionV3
        from keras.layers import Input, GlobalAveragePooling2D
        from keras.models import Model
        # Using Inception with ImageNet pre-trained weights
        weights_flag = 'imagenet'
        inception = InceptionV3(weights=weights_flag, include_top=False,input_shape=(160,320,3))
        for layer in inception.layers:
            layer.trainable = True

        # Makes the input placeholder layer 32x32x3 for CIFAR-10
        cifar_input = Input(shape=(160,320,3))

        #normalizing the image to have values between -1 an 1 with a mean of 0
        normalize = Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3))(cifar_input)

        inp = inception(normalize)

        new_layer_1 = GlobalAveragePooling2D()(inp)

        new_layer_2 = Dense(512, activation = 'relu')(new_layer_1)
        new_layer_3 = Dense(10, activation = 'relu')(new_layer_2)
        predictions = Dense(1, activation = 'softmax')(new_layer_3)

        model = Model(inputs=cifar_input, outputs=predictions)
    
    #Trainning
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
    print("Finish Training")

    model.save('model.h5')
    print("Saved model")
