# Report

# Contribution
- 50-50 contribution on this project, the reason most commits are made by one user is due to the live collaborative environment we used during the project.
# Design Choices for AbstractDataset Class

## 1. Attribute Choices:

- `root` and `data_type`:
  - **Design Choice:** Accessible Attributes
  - **Reason:** `root` stores the root location where data is stored, and `data_type` represents the type of data (e.g., image or audio). Both attributes are accessible within the class for managing dataset-related operations.

- `labels` and `data`:
  - **Design Choice:** Protected Attributes
  - **Reason:** Both `labels` and `data` are marked as protected attributes, denoted by a single leading underscore. This choice is made to indicate that these attributes are intended for internal use within the class. They can be accessed within the class and its subclasses but are not part of the public interface. If the user wishes to access this data, then can through the corresponding getter and setter functions denoted by the @property decorator.

## 2. Method Choices:

- `_load_labels`:
  - **Design Choice:** Protected Method
  - **Reason:** The method `_load_labels` is marked as protected, as it is intended for internal use within the class. It handles the loading of labels and can be accessed by other methods within the class.

- `__len__` and `__getitem__`:
  - **Design Choice:** Public Methods
  - **Reason:** Both `__len__` and `__getitem__` are essential methods for interacting with the dataset. They are made public to allow users to retrieve the length of the dataset and access individual data points using the standard Python indexing syntax.

- `load_data_eager` and `load_data_lazy`:
  - **Design Choice:** Public Methods
  - **Reason:** These methods provide users with the flexibility to load data either eagerly (loading all data into memory at once) or lazily (loading data from disk only when needed). They are made public to allow users to choose the loading strategy based on their requirements.

- `_load_data`:
  - **Design Choice:** Protected Method
  - **Reason:** `_load_data` is marked as protected to indicate that it is intended for internal use within the class. It handles the actual loading of data and can be accessed by other methods within the class.

- `split`:
  - **Design Choice:** Public Method
  - **Reason:** The `split` method is made public to allow users to split the dataset into training and test sets based on a specified ratio. This method provides a convenient way for users to partition the data for model training and evaluation.

## 3. Design Considerations:

- **Flexibility for Different Data Types:**
  - The inclusion of the `data_type` attribute allows the class to handle various types of data, such as images or audio. 

---
# Design Choices for Dataset Class

## 1. Attribute Choices:

- `_dataloader`, `_data_paths`:
  - **Design Choice:** Private Attributes
  - **Reason:** These attributes are designated as private to encapsulate internal details and prevent direct access from outside the class.

## 2. Property Choices:

- `dataloader` and `data_paths`:
  - **Design Choice:** Properties
  - **Reason:** These properties are used to access the private attributes `_dataloader` and `_data_paths`, respectively. The use of properties allows controlled access and potential validation.

## 3. Method Choices:

- `__len__`, `__getitem__`, `__iter__`:
  - **Design Choice:** Public Methods
  - **Reason:** These methods are intended for external use. `__len__` provides the length of the dataset, `__getitem__` allows accessing a specific data point, and `__iter__` facilitates iterating through the dataset.

- `load_data_eager`, `load_data_lazy`, `_load_data`:
  - **Design Choice:** Delegated Methods
  - **Reason:** These methods delegate the loading of data to the internal `_dataloader` object, promoting modularity and encapsulation.

- `split`:
  - **Design Choice:** Public Method
  - **Reason:** The `split` method is designed for users to split the dataset into training and testing sets. It is made public for external use.

## 4. Design Considerations:

- **Type Checking:**
  - The class includes type checking for input parameters to ensure that `root` is a string, `data_type` is a string, and `dataloader` is an instance of `DataLoader`.

- **Exception Handling:**
  - Exceptions are handled for file operations, such as when the root directory or label file is not found.

---

# Design Choices for LabeledDataset Class (Subclass of Dataset)

## 1. Attribute Choices:

- `_root` and `_label_file`:
  - **Design Choice:** Private Attributes
  - **Reason:** These attributes are designated as private to encapsulate internal details and prevent direct access from outside the class.

## 2. Method Choices:

- `_load_labels`:
  - **Design Choice:** Private Method
  - **Reason:** This method is designed to handle the loading of labels from a CSV file. It is kept private to manage internal operations.

- `__len__` and `__getitem__`:
  - **Design Choice:** Public Methods
  - **Reason:** These methods are intended for external use. `__len__` provides the length of the dataset, and `__getitem__` allows accessing a specific data point.

## 3. Design Considerations:

- **Sorting and Formatting Labels:**
  - Labels loaded from the CSV file are sorted alphabetically and formatted, ensuring consistency and ease of use.

---

# Design Choices for UnlabeledDataset Class (Subclass of Dataset)

## 1. Attribute Choices:

- `_root`:
  - **Design Choice:** Private Attribute
  - **Reason:** This attribute is designated as private to encapsulate internal details and prevent direct access from outside the class.

## 2. Method Choices:

- `_load_labels`:
  - **Design Choice:** Private Method
  - **Reason:** This method is designed to handle the loading of labels, which are set to `None` for unlabeled data. It is kept private to manage internal operations.

---

# Design Choices for HierarchicalDataset Class (Subclass of Dataset)

## 1. Attribute Choices:

- `_root`:
  - **Design Choice:** Private Attribute
  - **Reason:** This attribute is designated as private to encapsulate internal details and prevent direct access from outside the class.

## 2. Method Choices:

- `_load_labels`:
  - **Design Choice:** Private Method
  - **Reason:** This method is designed to handle the loading of labels based on the directory structure. It is kept private to manage internal operations.

- `__len__` and `__getitem__`:
  - **Design Choice:** Public Methods
  - **Reason:** These methods are intended for external use. `__len__` provides the total number of data points, and `__getitem__` allows accessing a specific data point.

## 3. Design Considerations:

- **Exception Handling for Directory:**
  - Exceptions are handled in case the root directory does not exist.

---

# Design Choices for DataLoader Class

## 1. Attribute Choices:

- `_root`, `_data_type`, `_data_paths`, `_labels`:
  - **Design Choice:** Protected Attributes
  - **Reason:** These attributes are used internally within the class and its subclasses for managing the data loading process. Making them protected ensures that they are accessible within the class hierarchy but not directly accessible outside the class.

## 2. Method Choices:

- `__init__`, `load_data_eager`, `load_data_lazy`, `_load_data`:
  - **Design Choice:** Initialization and Loading Methods
  - **Reason:** The `__init__` method initializes the DataLoader object with essential parameters. `load_data_eager` and `load_data_lazy` are abstract methods for loading data either eagerly or lazily, depending on the subclass implementation. The `_load_data` method is a utility method for loading individual data points from files.

## 3. Design Considerations:

- **File Path Handling:**
  - The class handles file paths for accessing data stored in directories. It constructs file paths based on the root directory and directory contents.

- **Data Type Flexibility:**
  - The `DataLoader` class is designed to handle different types of data (e.g., images, audio) based on the specified data type parameter.
  - The reason this class was made was to reduce the size of the Dataset class for readability and flexibility.

---

# Design Choices for UnlabeledDataLoader Class (Subclass of DataLoader)

## 1. Attribute Choices:

- Inherits attributes from DataLoader.

## 2. Method Choices:

- `load_data_eager` and `load_data_lazy`:
  - **Design Choice:** Custom Loading Methods
  - **Reason:** These methods are implemented to load unlabeled data either eagerly or lazily. They iterate through data directories, load individual data points, and store them in memory or yield them as a generator.

## 3. Design Considerations:

- **Unlabeled Data Handling:**
  - The class is specialized in loading unlabeled data. It doesn't require label information for loading data points.

---

# Design Choices for LabeledDataLoader Class (Subclass of DataLoader)

## 1. Attribute Choices:

- Inherits attributes from DataLoader.

## 2. Method Choices:

- `load_data_eager` and `load_data_lazy`:
  - **Design Choice:** Custom Loading Methods
  - **Reason:** These methods are implemented to load labeled data along with corresponding labels either eagerly or lazily. They ensure that data points are loaded along with their associated labels for supervised learning tasks.

## 3. Design Considerations:

- **Labeled Data Handling:**
  - The class is specialized in loading labeled data. It requires label information for associating labels with data points during the loading process.

---

# Design Choices for HierarchicalDataLoader Class (Subclass of DataLoader)

## 1. Attribute Choices:

- Inherits attributes from DataLoader.

## 2. Method Choices:

- `load_data_eager` and `load_data_lazy`:
  - **Design Choice:** Custom Loading Methods
  - **Reason:** These methods are implemented to load hierarchical data structures, such as folders containing subfolders, where each subfolder represents a class label. They handle the hierarchical organization of data and load it along with corresponding labels.

## 3. Design Considerations:

- **Hierarchical Data Handling:**
  - The class is specialized in loading hierarchical data structures. It leverages the directory structure to organize data into classes and loads them accordingly.


# Design Choices for BatchLoader Class

## 1. Attribute Choices:

- `data`, `batch_size`, `shuffle`, `discard_last`, and `indices`:
  - **Design Choice:** Accessible Attributes
  - **Reason:** These attributes are used for managing batch loading operations. `data` contains the input data, `batch_size` specifies the size of each batch, `shuffle` determines whether to shuffle data indices, `discard_last` indicates whether to discard the last incomplete batch, and `indices` stores the indices of the data. All these attributes are accessible within the class using their respective property decorator getter and setters

## 2. Method Choices:

- `__len__`:
  - **Design Choice:** Magic Method
  - **Reason:** `__len__` calculates and returns the number of batches that can be created from the data with the specified batch size. It is made public for external use.

- `__iter__`:
  - **Design Choice:** Magic Method
  - **Reason:** `__iter__` returns an iterator that yields batches of data. It allows users to iterate over batches conveniently and efficiently.

## 3. Design Considerations:

- **Batch Randomization:**
  - The class supports batch randomization based on the `shuffle` attribute. If set to `True`, it shuffles the indices before creating batches, providing randomness in the batch order.

- **Discarding Last Batch:**
  - Users have the option to discard the last incomplete batch if the dataset size is not a multiple of the batch size. This behavior is controlled by the `discard_last` attribute.

- **Error Handling:**
  - Error handling is implemented to handle potential exceptions during batch creation. If an error occurs while shuffling indices or creating a batch, a `RuntimeError` is raised with a descriptive error message.

# Design Choices for AbtractPreprocessor Class

## 1. Attribute Choices:

- `hyperparameters`:
  - **Design Choice:** Accessible Attribute
  - **Reason:** The `hyperparameters` attribute stores the hyperparameters for the preprocessor. It is accessible within the class for managing and applying hyperparameters during text preprocessing.

## 2. Method Choices:

- `__call__` and `_preprocess`:
  - **Design Choice:** Abstract Methods
  - **Reason:** Both `__call__` and `_preprocess` are abstract methods defined in the `ABC` class. They are left to be implemented by subclasses. `__call__` represents the main preprocessing operation on which data is applied, and `_preprocess` is an internal utility method.

## 3. Design Considerations:

- **Hyperparameter Handling:**
  - The class is designed to store hyperparameters and provide a structure for text preprocessing. Subclasses are expected to implement the actual preprocessing logic.

---

# Design Choices for CentreCrop Class (Subclass of AbtractPreprocessor)

## 1. Attribute Choices:

- `width_param` and `height_param`:
  - **Design Choice:** Accessible Attributes
  - **Reason:** These attributes store the width and height parameters for center cropping. They are accessible within the class for specifying the crop size.

## 2. Method Choices:

- `__call__` and `_preprocess`:
  - **Design Choice:** Public and Protected Methods
  - **Reason:** `__call__` is a public method that represents the main center cropping operation. It uses the protected `_preprocess` method for handling the actual cropping logic.

## 3. Design Considerations:

- **Parameter Validation:**
  - The class checks if the specified crop size is valid for the given image dimensions. If not, it raises a `ValueError` with a descriptive error message.

- **Image Cropping Logic:**
  - The center cropping logic is implemented in the `_preprocess` method. It calculates the crop coordinates based on the specified width and height parameters.

---

# Design Choices for RandomCrop Class (Subclass of CentreCrop)

## 1. Attribute Choices:

- Inherits attributes from CentreCrop.

## 2. Method Choices:

- `_preprocess`:
  - **Design Choice:** Customization for Random Crop
  - **Reason:** `_preprocess` is overridden in the `RandomCrop` class to implement random cropping logic. It generates random crop coordinates within the image dimensions.

## 3. Design Considerations:

- **Random Cropping Logic:**
  - The `_preprocess` method generates random crop coordinates within the image dimensions. This allows for random cropping of images by randomly generating the coordinates of the respective bounding box, and ensure it is within range.

---

# Design Choices for PitchShift Class (Subclass of AbtractPreprocessor)

## 1. Attribute Choices:

- `pitch_factor` and `sample_rate`:
  - **Design Choice:** Accessible Attributes
  - **Reason:** These attributes store the pitch shifting factor and sample rate of the audio. They are accessible within the class for specifying the audio transformation parameters.

## 2. Method Choices:

- `__call__` and `_preprocess`:
  - **Design Choice:** Magic and Protected Methods
  - **Reason:** `__call__` is a magic method representing the main pitch shifting operation. It uses the protected `_preprocess` method for handling the actual pitch shifting logic.

## 3. Design Considerations:

- **Pitch Shifting Logic:**
  - The pitch shifting logic is implemented in the `_preprocess` method using the librosa library. It applies pitch shifting to the input audio based on the specified pitch factor and sample rate.

---

# Design Choices for MelSpectrogram Class (Subclass of AbtractPreprocessor)

## 1. Attribute Choices:

- `sample_rate` and `file_name`:
  - **Design Choice:** Accessible Attributes
  - **Reason:** These attributes store the sample rate of the audio and the file name for saving the spectrogram image. They are accessible within the class for specifying parameters and managing the spectrogram output.

## 2. Method Choices:

- `__call__` and `_preprocess`:
  - **Design Choice:** Public and Protected Methods
  - **Reason:** `__call__` is a public method representing the main mel spectrogram computation. It uses the protected `_preprocess` method for handling the actual spectrogram generation logic.

## 3. Design Considerations:

- **Spectrogram Computation:**
  - The mel spectrogram computation is performed in the `_preprocess` method using the librosa library. It computes the mel spectrogram of the input audio data and visualizes it using matplotlib.

---

# Design Choices for PreprocessingPipeline Class (Subclass of AbtractPreprocessor)

## 1. Attribute Choices:

- `preprocessors` and `data`:
  - **Design Choice:** Accessible Attributes
  - **Reason:** These attributes store the list of preprocessors in the pipeline and the preprocessed data, respectively. They are accessible within the class for managing the preprocessing pipeline and accessing the processed data.

## 2. Method Choices:

- `__call__` and `_preprocess`:
  - **Design Choice:** Public and Protected Methods
  - **Reason:** `__call__` is a public method representing the main preprocessing pipeline operation. It uses the protected `_preprocess` method for applying the preprocessors in sequence.

## 3. Design Considerations:

- **Pipeline Composition:**
  - The class allows the composition of multiple preprocessors into a preprocessing pipeline.
