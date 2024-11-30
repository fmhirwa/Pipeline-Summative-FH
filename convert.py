import numpy as np

def load_images(file_path):
    with open(file_path, 'rb') as f:
        # Read and parse IDX file headers
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        # Load image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read and parse IDX file headers
        magic, num = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        # Load label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Paths to the IDX files
train_images_path = './archive/train-images.idx3-ubyte'
train_labels_path = './archive/train-labels.idx1-ubyte'
test_images_path = './archive/t10k-images.idx3-ubyte'
test_labels_path = './archive/t10k-labels.idx1-ubyte'

# Load the data
train_images = load_images(train_images_path)
train_labels = load_labels(train_labels_path)
test_images = load_images(test_images_path)
test_labels = load_labels(test_labels_path)

# Save as .npy files
np.save('./archive/train_images.npy', train_images)
np.save('./archive/train_labels.npy', train_labels)
np.save('./archive/test_images.npy', test_images)
np.save('./archive/test_labels.npy', test_labels)

print("MNIST data has been converted to .npy files and saved!")
