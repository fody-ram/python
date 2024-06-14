import os

def get_total_image_count(data_dir):
  """Calculates the total number of images in a directory."""
  total_count = 0
  for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    total_count += len(os.listdir(class_path))
  return total_count

# Specify data directory paths
data_dir_train = 'C:\\my files\\IIUM\\6\\fyp_1\\FYP\\datasets\\archive (5)\\OCT2017\\train'
data_dir_test = 'C:\\my files\\IIUM\\6\\fyp_1\\FYP\\datasets\\archive (5)\\OCT2017\\test'

# Calculate total image counts for all sets
total_train_images = get_total_image_count(data_dir_train)
total_test_images = get_total_image_count(data_dir_test)

# Calculate overall split ratios (assuming all classes in both sets)
total_images = total_train_images + total_test_images
train_ratio = (total_train_images / total_images) * 100
test_ratio = (total_test_images / total_images) * 100

print(f"Estimated Train-Test-Validation Split Ratio:")
print(f"- Train: {train_ratio:.2f}%")
print(f"- Test: {test_ratio:.2f}%")
