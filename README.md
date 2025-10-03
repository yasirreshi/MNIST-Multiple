# MNIST Classification Project

This project documents an iterative journey to build and optimize a Convolutional Neural Network (CNN) for the MNIST dataset. It explores three distinct model architectures, tracks their performance, and refines the training process to meet specific constraints on parameter count and accuracy.

## Project Structure

- `model.py`, `model_v1.py`, `model_v2.py`: Python files defining the three different CNN architectures.
- `train_xpu.py`: The main training script. It is designed to be flexible and can run on Intel XPUs (Arc GPUs), NVIDIA CUDA GPUs, or CPUs. It handles data loading, augmentation, training, testing, and logging.
- `utils.py`: Contains helper functions for plotting metrics, analyzing model performance (parameter count, receptive field), and generating visualizations like confusion matrices.
- `requirements.txt`: A list of Python packages required to run the project.
- `analytics/`: A directory containing all the output from the training runs.
  - `pre_training/`: Contains model summaries, hyperparameter configurations, and data samples saved before training starts.
  - `post_training/`: Contains performance plots, confusion matrices, classification reports, and misclassified image samples generated after training is complete.
  - `training.log`, `training_v1.log`, `training_v2.log`: Log files for each training session, capturing the console output.

## Models Summary & Architectural Journey

The project evolved through three major model versions, each with a specific goal.

### Model 1: `model.py` (The Baseline)

- **Objective**: To establish a simple, functional CNN as a performance benchmark without modern regularization techniques.
- **Architectural Philosophy**: A traditional, sequential stack of 8 convolutional layers followed by a dense, fully-connected layer for classification. This "heavy head" is a classic design but often contains the majority of a model's parameters.
- **Key Components**:
  - `nn.Conv2d`: Used for all feature extraction.
  - `nn.MaxPool2d`: A single pooling layer to reduce spatial dimensions.
  - `nn.Linear`: A large final classifier layer.
  - **Omissions**: No `BatchNorm` or `Dropout`.
- **Performance**:
  - **Parameters**: 21,538
  - **Best Accuracy**: 99.36%
- **Analysis**: This model performed surprisingly well, proving the feature extraction capabilities of a deep convolutional stack. However, its high parameter count and lack of regularization make it inefficient and prone to overfitting on more complex datasets.

### Model 2: `model_v1.py` (The Lightweight Redesign)

- **Objective**: To drastically reduce the parameter count below 8,000 while maintaining high accuracy, proving that efficiency is possible.
- **Architectural Philosophy**: This model introduces modern, efficient CNN design patterns. The core innovation is the replacement of the fully-connected head with a **Global Average Pooling (GAP)** layer.
- **Key Components**:
  - `nn.BatchNorm2d`: Added after each convolution to stabilize training and improve convergence.
  - `nn.Dropout`: Added to regularize the network and prevent overfitting.
  - `nn.AvgPool2d` (GAP): The final convolutional feature maps are averaged into a single value per channel, which directly feeds into the final classification output. This eliminates the need for a large `Linear` layer.
- **Performance**:
  - **Parameters**: 6,308 (a 70% reduction from the baseline)
  - **Best Accuracy**: 99.18%
- **Analysis**: This model successfully demonstrated that a lightweight architecture could achieve comparable (though slightly lower) accuracy. The massive parameter reduction with only a minor accuracy trade-off highlights the efficiency of the GAP layer.

### Model 3: `model_v2.py` (The Constrained Optimizer)

- **Objective**: To build the most efficient model possible, targeting **<6,000 parameters** while pushing for the highest possible accuracy (approaching 99.5%).
- **Architectural Philosophy**: A carefully balanced and refined architecture. It uses two pooling layers for more aggressive spatial reduction, allowing the network to build a sufficient receptive field with fewer channels in the early layers. The channel progression is meticulously managed to stay within the strict parameter budget.
- **Key Components**:
  - **Dual `MaxPool2d` Layers**: Provides a more structured down-sampling approach, growing the receptive field efficiently.
  - **Managed Channel Growth**: The number of channels increases modestly (e.g., 1 -> 8 -> 10), peaks in the middle (12 channels), and then reduces before the final classification.
  - **`nn.AdaptiveAvgPool2d` (GAP)**: Retains the efficient GAP head from `model_v1`.
- **Performance**:
  - **Parameters**: 7,718 (per analytics export)
  - **Final RF**: 56 (covers the 28x28 input comfortably via GAP)
  - **Best Accuracy**: 99.15% (observed on a prior ~7.7k-parameter variant; expect similar with this layout).
- **Analysis**: This model represents the pinnacle of the optimization journey, achieving the lowest parameter count while maintaining a strong, deep architecture. It is a testament to principled CNN design, where every layer and channel is chosen with the parameter budget in mind.

### Hyperparameter & Training Strategy

- **Optimizer**: `SGD` with momentum (0.9) was used for all models, providing stable and reliable convergence.
- **Scheduler**: `ReduceLROnPlateau` was used to dynamically adjust the learning rate based on validation loss, allowing the model to fine-tune itself when training plateaued.
- **Initial Learning Rate**: Current runs use an initial LR of `0.05` (see `train_xpu.py`).
- **Data Augmentation**: `RandomRotation` and `RandomAffine` were used to create variations in the training data, helping the model generalize better.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    If you are using an Intel Arc GPU, make sure to install the `intel-extension-for-pytorch` package as well.

2.  **Run Training**:
    You can train any of the three models by passing the model's file name (without the `.py` extension) as an argument to the training script.

    - To train the baseline model:
      ```bash
      python train.py --model model
      ```
    - To train the `v1` model:
      ```bash
      python train.py --model model_v1
      ```
    - To train the `v2` model:
      ```bash
      python train.py --model model_v2
      ```

    The script will automatically detect the best available hardware (XPU, CUDA, or CPU) and run the training. All results, logs, and plots will be saved in the `analytics` directory.
