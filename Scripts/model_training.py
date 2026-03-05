import tensorflow as tf
from tensorflow.keras import layers, models
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE #SMOTE
from imblearn.under_sampling import RandomUnderSampler #RandomUnderSampler
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
processed_dir = os.path.join(root_dir, "Processed_Dataset")
#model code
def create_model():
    model = models.Sequential([
        layers.Input(shape=(960, 3)),

        #Layer 1
        layers.Conv1D(64, kernel_size=64, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        #Layer 2
        layers.Conv1D(128, kernel_size=32, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),

        # Layer 3
        layers.Conv1D(256, kernel_size=16, activation='relu'),
        layers.Flatten(), #This keeps the 'timing' of the events better than GlobalAverage

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), #Stronger dropout for small subject count
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #learning rate=0.0001
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
   


subjects = ["AP01", "AP02", "AP03", "AP04", "AP05"]
all_fold_results = []

for i, test_sub in enumerate(subjects):
    print(f"\nFOLD {i+1}/5: Testing on {test_sub}")

    #Identify Training Subjects- leave one out for testing(1,2,3,4,5)
    train_subs = [s for s in subjects if s != test_sub]
    def feature_scale(X):
    #Iterate through each of the 3 channels
      for i in range(X.shape[2]):
          min_val = np.min(X[:, :, i])
          max_val = np.max(X[:, :, i])
          if max_val - min_val != 0:
              X[:, :, i] = (X[:, :, i] - min_val) / (max_val - min_val)
      return X

    #load train data
    X_train_list = [feature_scale(np.load(os.path.join(processed_dir, f"X_{s}.npy"))) for s in train_subs]
    X_test = feature_scale(np.load(os.path.join(processed_dir, f"X_{test_sub}.npy")))
    # Load and Concatenate Training Data
    X_train_list = [np.load(os.path.join(processed_dir, f"X_{s}.npy")) for s in train_subs]
    y_train_list = [np.load(os.path.join(processed_dir, f"y_{s}.npy")) for s in train_subs]

    X_train = np.concatenate(X_train_list, axis=0);
    y_train = np.concatenate(y_train_list, axis=0)

    #Load Test Data
    X_test = np.load(os.path.join(processed_dir, f"X_{test_sub}.npy"))
    y_test = np.load(os.path.join(processed_dir, f"y_{test_sub}.npy"))
    #Under-sampling the majority class (Normal) to balance the dataset for training
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_flat = X_train.reshape(n_samples, n_timesteps * n_features)

    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled_flat, y_resampled = rus.fit_resample(X_train_flat, y_train)

    #Reshape back for Conv1D
    X_resampled = X_resampled_flat.reshape(-1, n_timesteps, n_features)

    #Train with a lr=0.0001; 30 epochs
    model = create_model()
    model.fit(X_resampled, y_resampled, epochs=30, batch_size=3)


    #Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    #Store Results for this Fold
    all_fold_results.append((y_test, y_pred))

#concantanate all results of all folds
total_y_true = np.concatenate([res[0] for res in all_fold_results])
total_y_pred = np.concatenate([res[1] for res in all_fold_results])
unique, counts = np.unique(total_y_true, return_counts=True)
print(dict(zip(unique, counts)))

#Calculate Metrics-accuracy, prescion, recall
accuracy = accuracy_score(total_y_true, total_y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(total_y_true, total_y_pred, average='macro')

#print(f"Overall Accuracy: {accuracy:.4f}")
#print(f"Macro Precision:  {precision:.4f}")
#print(f"Macro Recall:     {recall:.4f}")