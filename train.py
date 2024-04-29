from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import argparse
from model import build_model
from prepare_data import setup

# Support command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--big-model', action='store_true', help='Use the bigger model with more conv layers')
parser.add_argument('--use-data-dir', action='store_true', help='Use custom data directory, at /data')
args = parser.parse_args()

if args.big_model:
  print('Using big model')
if args.use_data_dir:
  print('Using data directory')

# Prepare data
train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs, test_Y, im_shape, vocab_size, num_answers, _, _, _ = setup(args.use_data_dir)

print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers, args.big_model)
checkpoint = ModelCheckpoint('model-0426.h5', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
# TensorBoard for monitoring logs
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)


print('\n--- Training model...')
model.fit(
  [train_X_ims, train_X_seqs],
  train_Y,
  validation_data=([test_X_ims, test_X_seqs], test_Y),
  shuffle=True,
  epochs=20,
  callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
)
