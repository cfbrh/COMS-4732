import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply, Attention, Embedding, LSTM, Bidirectional, GlobalAveragePooling1D, Concatenate, Dropout, BatchNormalization, DataAugmentation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

def build_model(im_shape, vocab_size, num_answers, big_model):
  # Data augmentation
  data_augmentation = DataAugmentation([
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.1),
  ])
  # The CNN
  im_input = Input(shape=im_shape)
  x1 = data_augmentation(im_input)
  x1 = Conv2D(16, 5, padding='same', kernel_regularizer=l2(0.001))(x1)
  x1 = MaxPooling2D()(x1)
  x1 = Conv2D(32, 5, padding='same', kernel_regularizer=l2(0.001))(x1)
  x1 = MaxPooling2D()(x1)
  if big_model:
    x1 = Conv2D(64, 3, padding='same')(x1)
    x1 = MaxPooling2D()(x1)
  x1 = Flatten()(x1)
  x1 = Dense(64, activation='relu')(x1)
  x1 = Dropout(0.5)(x1) # Add dropout

  # The question network
  q_input = Input(shape=(vocab_size,))
  # x2 = Dense(32, activation='tanh')(q_input)
  # x2 = Dense(32, activation='tanh')(x2)
  x2 = Embedding(vocab_size, 32)(q_input) # 使用Embedding层而不是Dense层
  x2 = Bidirectional(LSTM(32, return_sequences=True))(x2) # 使用LSTM编码序列 
  
  # Attention融合 
  x2_att = Attention()([x2, x2]) 
  x2 = GlobalAveragePooling1D()(x2_att)
  x2 = Dropout(0.5)(x2)  # 平均池化
  # Merge -> output
  out = Concatenate()([x1, x2])
  out = Dense(64, activation='relu')(out)
  out = Dropout(0.5)(out)  # Add dropout
  # out = Multiply()([x1, x2])
  # out = Dense(32, activation='tanh')(out)
  out = Dense(num_answers, activation='softmax')(out)

  model = Model(inputs=[im_input, q_input], outputs=out)
  model.compile(Adam(lr=5e-4, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
  
  # model.compile(RMSprop(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
  #lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
  #              initial_learning_rate=1e-2,
  #              decay_steps=10000,
  #              decay_rate=0.9)
  #opt = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
  #model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

  return model

# Setup early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
