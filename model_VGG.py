import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply, Attention, Embedding, LSTM, Bidirectional, GlobalAveragePooling1D, Concatenate, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16

def build_model(im_shape, vocab_size, num_answers, big_model):
  # Load pre-trained VGG16 as a feature extractor
  base_model = VGG16(include_top=False, input_shape=im_shape)
  base_model.trainable = False  # Freeze the convolutional base

  # Create new model on top
  im_input = Input(shape=im_shape)
  # Data augmentation
  #x1 = RandomFlip("horizontal")(im_input)
  #x1 = RandomRotation(0.1)(x1)
  #x1 = RandomZoom(0.1)(x1)
  #x1 = Conv2D(16, 5, padding='same', kernel_regularizer=l2(0.001))(x1)
  #x1 = MaxPooling2D()(x1)
  #x1 = Conv2D(32, 5, padding='same', kernel_regularizer=l2(0.001))(x1)
  #x1 = MaxPooling2D()(x1)
  x1 = base_model(im_input)  # Use the base model as a feature extractor
  x1 = Flatten()(x1)
  x1 = Dense(256, activation='relu')(x1)
  x1 = Dropout(0.5)(x1) # Add dropout

  # The question network
  q_input = Input(shape=(vocab_size,))
  # x2 = Dense(32, activation='tanh')(q_input)
  # x2 = Dense(32, activation='tanh')(x2)
  x2 = Embedding(vocab_size, 32)(q_input) # Embedding
  x2 = Bidirectional(LSTM(32, return_sequences=True))(x2) # LSTM
  
  # Attention merge
  #x2_att = Attention()([x2, x2]) 
  x2 = GlobalAveragePooling1D()(x2)
  x2 = Dropout(0.5)(x2)  # average pool
  # Merge -> output
  out = Concatenate()([x1, x2])
  out = Dense(128, activation='relu')(out)
  out = Dropout(0.5)(out)  # Add dropout
  # out = Multiply()([x1, x2])
  # out = Dense(32, activation='tanh')(out)
  out = Dense(num_answers, activation='softmax')(out)

  model = Model(inputs=[im_input, q_input], outputs=out)
  #model.compile(tf.keras.optimizers.legacy.Adam(lr=5e-4, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # model.compile(RMSprop(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
  #lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
  #              initial_learning_rate=1e-2,
  #              decay_steps=10000,
  #              decay_rate=0.9)
  #opt = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
  #model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

  return model

