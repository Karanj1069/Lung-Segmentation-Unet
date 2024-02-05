It appears that you have a comprehensive lung segmentation project structured into three main parts: training, model architecture, and evaluation. Below is a summary of each section:

### 1. **Training (`train.py`):**
- **Environment Setup:**
  - Sets TensorFlow environment variables.
  - Imports necessary libraries.

- **Functions:**
  - `create_dir(path)`: Creates a directory if it doesn't exist.
  - `load_data(path, split=0.1)`: Loads and splits the dataset into training, validation, and test sets.
  - `read_image(path)`: Reads and preprocesses an image.
  - `read_mask(path1, path2)`: Reads and preprocesses masks.
  - `tf_parse(x, y1, y2)`: TensorFlow dataset parsing function.
  - `tf_dataset(X, Y1, Y2, batch=8)`: Creates a TensorFlow dataset.

- **Main Script:**
  - Seeds for reproducibility.
  - Creates a directory for storing model files.
  - Sets hyperparameters.
  - Loads and splits the dataset.
  - Builds and compiles the U-Net model.
  - Defines callbacks (ModelCheckpoint, ReduceLROnPlateau, CSVLogger).
  - Trains the model on the dataset.

### 2. **Model Architecture (`model.py`):**
- Defines U-Net architecture using TensorFlow/Keras.
- Contains functions for convolutional blocks, encoder and decoder blocks, and the overall U-Net model.

### 3. **Metrics (`metrics.py`):**
- Defines custom evaluation metrics for the model, including Intersection over Union (IoU), Dice coefficient, and Dice loss.

### 4. **Evaluation (`eval.py`):**
- Similar environment setup as in training.
- Loads the trained model.
- Reads the test dataset.
- Predicts lung masks using the loaded model.
- Saves the original image, ground truth, and predicted mask side by side for evaluation.

### Note:
- Ensure that the code is structured into separate files (`train.py`, `model.py`, `metrics.py`, `eval.py`) or adjust accordingly.
- Some functions and imports related to model training and evaluation are commented out, and you might need to uncomment them based on your needs.

Make sure you have the required dataset paths and file locations configured properly in your project directory.
