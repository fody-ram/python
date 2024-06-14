import os
from collections import Counter

def get_class_distribution(data_dir):
  """Calculates class distribution (number of images per class) in a directory."""
  class_counts = Counter()
  for image_class in os.listdir(data_dir):
    class_counts[image_class] = len(os.listdir(os.path.join(data_dir, image_class)))
  return class_counts

# Specify paths to your data directories (replace with your actual paths)
data_dir_train = 'C:\\my files\\IIUM\\6\\fyp_1\\FYP\\datasets\\archive (5)\\OCT2017\\train'
data_dir_test = 'C:\\my files\\IIUM\\6\\fyp_1\\FYP\\datasets\\archive (5)\\OCT2017\\test'
data_dir_val = 'C:\\my files\\IIUM\\6\\fyp_1\\FYP\\datasets\\archive (5)\\OCT2017\\val'

# Calculate class distributions for each dataset
train_class_counts = get_class_distribution(data_dir_train)
test_class_counts = get_class_distribution(data_dir_test)
val_class_counts = get_class_distribution(data_dir_val)

# Calculate overall class distribution
total_class_counts = Counter()
total_class_counts.update(train_class_counts)
total_class_counts.update(test_class_counts)
total_class_counts.update(val_class_counts)

# Calculate and print overall ratios
total_images = sum(total_class_counts.values())
print("Overall Class Distribution Ratios:")
for class_label, count in total_class_counts.items():
  ratio = count / total_images
  print(f"Class: {class_label}, Ratio: {ratio:.2f}")
