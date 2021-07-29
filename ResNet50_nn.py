from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pickle

BATCH_SIZE = 128

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=10,
                                   zoom_range=0.05,
                                   shear_range=0.15,
                                   fill_mode="nearest",
                                   horizontal_flip=False,)


training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(32, 32),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical',
                                                 color_mode='rgb')

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(32, 32),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical',
                                            color_mode='rgb')

base_model = ResNet50(weights="imagenet", include_top=False,
                      input_tensor=Input(shape=(32, 32, 3)))

head_model = base_model.output
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(units=512, activation="relu")(head_model)
head_model = Dropout(0.3)(head_model)
head_model = Dense(units=37, activation="softmax")(head_model)

model = Model(inputs=base_model.inputs, outputs=head_model)


for layer in base_model.layers:
    layer.trainable = False


opt = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(x=training_set, validation_data=test_set, epochs=50, callbacks=[early_stop])

test_set.reset()
training_set.reset()


for layer in base_model.layers[40:]:
    layer.trainable = True


opt = Adam(learning_rate=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(x=training_set, validation_data=test_set, epochs=20, callbacks=[early_stop])

model.save('model/classifier.h5')
labels = training_set.class_indices
f = open('model/labels.pickle', 'wb')
f.write(pickle.dumps(labels))
f.close()
