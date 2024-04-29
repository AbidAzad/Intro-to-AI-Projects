import numpy as np
import scipy.optimize as opt
import time
import random
import warnings
from scipy.optimize import OptimizeWarning
warnings.filterwarnings("ignore", category=OptimizeWarning)
import os
import pickle
import matplotlib.pyplot as plt

# Function to convert characters to integers
def char_to_int_mapper(character):
    conversion_map = {' ': 0, '+': 1, '#': 2}
    return conversion_map.get(character, -1)

# Recursive function to convert data to integers
def recursive_int_conversion(data):
    return char_to_int_mapper(data) if type(data) != np.ndarray else np.array(list(map(recursive_int_conversion, data)))

# Function to load data file randomly
def load_data_randomly(file_path: str, order_list: list, w: int, h: int):
    loaded_items = []
    with open(file_path, "r") as data_file:
        for i in order_list:
            data_file.seek(i * (w + 1) * h, 0)
            pic_data = [[c for c in data_file.readline().rstrip('\n')] for _ in range(h)]
            loaded_items.append(Picture(np.array(pic_data), w, h))
    return loaded_items

# Function to load label file randomly
def load_labels_randomly(file_path: str, order_list: list):
    file_handle = open(file_path, "r")
    labels = []
    for i in order_list:
        file_handle.seek(2 * i, 0)
        labels.append(file_handle.read(1))
    return labels

# Function to load data file
def load_data(file_path: str, total_num: int, w: int, h: int):
    data_items = []
    file_handle = open(file_path, "r")
    for i in range(total_num):
        file_handle.seek(i * (w + 1) * h, 0)
        pic_data = []
        for line_counter in range(h):
            pic_data.append([ch for ch in file_handle.readline() if ch != '\n'])
        data_items.append(Picture(np.array(pic_data), w, h))
    return data_items

# Function to load label file
def load_labels(file_path: str, total_num: int):
    file_handle = open(file_path, "r")
    labels = []
    for i in range(total_num):
        file_handle.seek(2 * i, 0)
        labels.append(file_handle.read(1))
    return labels

# Class representing a picture
class Picture:
    def __init__(self, data, width: int, height: int):
        self.width = width
        self.height = height
        if data is None:
            data = [[' ' for i in range(self.width)] for j in range(self.height)]
        self.pixels = np.rot90(recursive_int_conversion(data), -1)

    def getPixel(self, column, row):
        return self.pixels[column][row]

# Counter class with additional functionalities
class Counter(dict):
    def __getitem__(self, index):
        # Ensure that accessing a non-existing key returns 0 instead of raising a KeyError
        self.setdefault(index, 0)
        return dict.__getitem__(self, index)

    def argMax(self):
        # Return the key with the maximum value in the counter
        if not self:
            return None
        return max(self, key=self.get)

    def copy(self):
        # Return a copy of the counter
        return Counter(dict.copy(self))

    def __mul__(self, y):
        # Dot product of two counters, useful for calculating the score in the perceptron classifier
        return sum(self[key] * y.get(key, 0) for key in self)

    def __add__(self, y):
        # Element-wise addition of two counters, useful for updating weights in the perceptron classifier
        result = Counter(self)
        for key, value in y.items():
            result[key] += value
        return result

    def __sub__(self, y):
        # Element-wise subtraction of two counters, useful for updating weights in the perceptron classifier
        result = Counter(self)
        for key, value in y.items():
            result[key] -= value
        return result

# Perceptron classifier class
class PerceptronClassifier:
    def __init__(self, categories, dataType, dataUsage, useSavedLearnWeightData):
        # Initialize PerceptronClassifier with categories
        self.categories = categories
        # Initialize weights for each category using Counter
        self.weights = {cat: Counter() for cat in categories}
        self.data_type = dataType
        self.data_usage = dataUsage
        self.weights_folder = "learnedWeights"
        self.usedLearnedWeights = False
        self.useSavedLearnWeightData = useSavedLearnWeightData

    def train(self, train_data, train_labels, val_data, val_labels):
        if(self.useSavedLearnWeightData):
            # Create the learnedWeights folder if it doesn't exist
            os.makedirs(self.weights_folder, exist_ok=True)

            # Check if weights file exists
            weights_file_path = os.path.join(self.weights_folder, f"Perceptron{self.data_type}Weights({self.data_usage} Percent Train Data Usage).pkl")
            if os.path.exists(weights_file_path):
                # Load weights from file
                self.usedLearnedWeights = True
                with open(weights_file_path, 'rb') as file:
                    self.weights = pickle.load(file)
                return
            else:
                print("Could not find Existing Saved Learned Weights. Will Create One Now...", end = "")
        # Set learning rate and maximum number of iterations
        learn_rate = 1
        max_iterations = 10
        # Get features from training data
        self.features = train_data[0].keys()

        # Initialize weights for each category and feature
        for cat in self.categories:
            # Initialize bias weight
            self.weights[cat][0] = 0.1
            # Initialize feature weights
            for key in self.features:
                self.weights[cat][key] = 0.5

        # Variables to store best weights and accuracy
        best_weights, best_accuracy = {}, 0

        # Loop through maximum iterations
        for _ in range(max_iterations):
            all_passed = True
            # Iterate through each training data point
            for i, data in enumerate(train_data):
                # Calculate result for each category
                result = {cat: self.weights[cat] * data + self.weights[cat][0] for cat in self.categories}

                # Predicted category is the one with maximum result
                predicted_cat = max(result, key=result.get)
                
                # Update weights if prediction is incorrect
                if predicted_cat != int(train_labels[i]):
                    if result[predicted_cat] > 0:
                        # If prediction is incorrect and its score is positive, decrease its weights
                        self.weights[predicted_cat] -= data
                        self.weights[predicted_cat][0] -= learn_rate
                    if result[int(train_labels[i])] < 0:
                        # If correct label's score is negative, increase its weights
                        self.weights[int(train_labels[i])] += data
                        self.weights[predicted_cat][0] += learn_rate
                    all_passed = False

            # Classify validation data
            predictions = self.classify(val_data)
            # Calculate accuracy
            correct = sum(guess == int(label) for guess, label in zip(predictions, val_labels))
            accuracy = correct / len(val_labels)

            # Update best weights and accuracy if current accuracy is better
            if accuracy > best_accuracy:
                best_weights = self.weights
                best_accuracy = accuracy

            # If all data points are correctly classified, stop training
            if all_passed:
                break

        # Set weights to the best weights found during training
        self.weights = best_weights

        if(self.useSavedLearnWeightData):
            # Save weights to file
            with open(weights_file_path, 'wb') as file:
                pickle.dump(self.weights, file)

    def classify(self, data):
        # Classify data based on the highest score from each category
        return [max(self.categories, key=lambda cat: self.weights[cat] * pic + self.weights[cat][0]) for pic in data]

# Two-layer neural network classifier class
class TwoLayerNeuralNetworkClassifier:
    def __init__(self, output_classes, input_size, hidden_size, output_size, data_size, regularization_param, dataType, dataUsage, useSavedLearnWeightData):
        # Initialize classifier parameters
        self.input_size = input_size  # Number of input features
        self.hidden_size = hidden_size  # Number of neurons in the hidden layer
        self.output_size = output_size  # Number of output classes
        self.data_size = data_size  # Size of the dataset
        self.output_classes = output_classes  # List of output classes
        self.regularization_param = regularization_param  # Regularization parameter
        self.data_type = dataType
        self.data_usage = dataUsage
        self.weights_folder = "learnedWeights"
        self.usedLearnedWeights = False
        self.useSavedLearnWeightData = useSavedLearnWeightData

        # Initialize activation matrices
        self.input_activation = np.ones((self.input_size + 1, data_size))  # Input layer activation matrix
        self.hidden_activation = np.ones((self.hidden_size + 1, data_size))  # Hidden layer activation matrix
        self.output_activation = np.ones((self.output_size, data_size))  # Output layer activation matrix

        # Initialize bias
        self.bias = np.ones((1, data_size))  # Bias for each data point

        # Initialize weight change matrices
        self.input_change = np.zeros((self.hidden_size, self.input_size + 1))  # Change in input layer weights
        self.output_change = np.zeros((self.output_size, self.hidden_size + 1))  # Change in output layer weights

        # Initialize weight epsilon values for random weight initialization
        self.hidden_epsilon = np.sqrt(6.0 / (self.input_size + self.hidden_size))  # Epsilon for hidden layer weights
        self.output_epsilon = np.sqrt(6.0 / (self.input_size + self.output_size))  # Epsilon for output layer weights

        # Initialize weights randomly within epsilon range
        self.input_weights = np.random.rand(self.hidden_size, self.input_size + 1) * 2 * self.hidden_epsilon - self.hidden_epsilon  # Hidden layer weights
        self.output_weights = np.random.rand(self.output_size, self.hidden_size + 1) * 2 * self.output_epsilon - self.output_epsilon  # Output layer weights

    def set_regularization_param(self, regularization_param):
        # Set regularization parameter
        self.regularization_param = regularization_param

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1.0 / (1.0 + np.exp(-x))

    def dsigmoid(self, y):
        # Derivative of sigmoid function
        return y * (1.0 - y)

    def feed_forward(self, weight_vector):
        # Feed-forward propagation through the network
        self.input_weights = weight_vector[0:self.hidden_size * (self.input_size + 1)].reshape((self.hidden_size, self.input_size + 1))  # Extract input layer weights
        self.output_weights = weight_vector[-self.output_size * (self.hidden_size + 1):].reshape((self.output_size, self.hidden_size + 1))  # Extract output layer weights

        # Compute activations of hidden layer
        hidden_z = self.input_weights.dot(self.input_activation)  # Weighted sum of inputs to hidden layer
        self.hidden_activation[:-1, :] = self.sigmoid(hidden_z)  # Activation of hidden layer neurons

        # Compute activations of output layer
        output_z = self.output_weights.dot(self.hidden_activation)  # Weighted sum of inputs to output layer
        self.output_activation = self.sigmoid(output_z)  # Activation of output layer neurons

        # Calculate cost and regularization
        cost_matrix = self.output_truth * np.log(self.output_activation) + (1 - self.output_truth) * np.log(1 - self.output_activation)  # Cost matrix
        regulations = (np.sum(self.output_weights[:, :-1] ** 2) + np.sum(self.input_weights[:, :-1] ** 2)) * self.regularization_param / 2  # Regularization term
        return (-cost_matrix.sum() + regulations) / self.data_size  # Overall cost

    def back_propagate(self, weight_vector):
        # Back-propagation to compute gradients
        self.input_weights = weight_vector[0:self.hidden_size * (self.input_size + 1)].reshape((self.hidden_size, self.input_size + 1))  # Extract input layer weights
        self.output_weights = weight_vector[-self.output_size * (self.hidden_size + 1):].reshape((self.output_size, self.hidden_size + 1))  # Extract output layer weights

        # Compute error at output layer
        output_error = self.output_activation - self.output_truth  # Output error

        # Compute error at hidden layer
        hidden_error = self.output_weights[:, :-1].T.dot(output_error) * self.dsigmoid(self.hidden_activation[:-1:])  # Hidden layer error

        # Compute weight changes
        self.output_change = output_error.dot(self.hidden_activation.T) / self.data_size  # Change in output layer weights
        self.input_change = hidden_error.dot(self.input_activation.T) / self.data_size  # Change in input layer weights

        # Apply regularization to weight changes
        self.output_change[:, :-1].__add__(self.regularization_param * self.output_weights[:, :-1])  # Regularization for output layer weights
        self.input_change[:, :-1].__add__(self.regularization_param * self.input_weights[:, :-1])  # Regularization for input layer weights

        # Concatenate and return weight changes
        return np.append(self.input_change.ravel(), self.output_change.ravel())

    def train(self, train_data, train_labels, valid_data, valid_labels):
        if(self.useSavedLearnWeightData):
            # Create the learnedWeights folder if it doesn't exist
            os.makedirs(self.weights_folder, exist_ok=True)

            # Check if weights file exists
            inputWeights_file_path = os.path.join(self.weights_folder, f"NeuralNetwork{self.data_type}InputWeights({self.data_usage} Percent Train Data Usage).pkl")
            outputWeights_file_path = os.path.join(self.weights_folder, f"NeuralNetwork{self.data_type}OutputWeights({self.data_usage} Percent Train Data Usage).pkl")
            if os.path.exists(inputWeights_file_path):
                # Load weights from file
                self.usedLearnedWeights = True
                with open(inputWeights_file_path, 'rb') as file:
                    self.input_weights = pickle.load(file)
                with open(outputWeights_file_path, 'rb') as file:
                    self.output_weights = pickle.load(file)
                return
            else:
                print("Could not find Existing Saved Learned Weights. Will Create One Now...", end = "")

        # Training the neural network
        self.train_data = train_data
        self.train_labels = train_labels
        self.valid_data = valid_data
        self.valid_labels = valid_labels

        # Extract features from training data
        self.size_train = len(train_data)
        features_train = [list(datum.values()) for datum in train_data]
        train_set = np.array(features_train, dtype=np.int32)

        # Set input activations and output truth values
        self.input_activation[:-1, :] = train_set.transpose()
        self.output_truth = self.labelMatrix(train_labels)

        # Combine weights into a single vector
        weight_vector = np.append(self.input_weights.ravel(), self.output_weights.ravel())

        # Use conjugate gradient optimization to minimize cost function
        iteration = 100
        weight_vector = opt.fmin_cg(self.feed_forward, weight_vector, fprime=self.back_propagate, maxiter=iteration, disp=False)

        # Separate weights back into input and output weights
        self.input_weights = weight_vector[0:self.hidden_size * (self.input_size + 1)].reshape((self.hidden_size, self.input_size + 1))
        self.output_weights = weight_vector[-self.output_size * (self.hidden_size + 1):].reshape((self.output_size, self.hidden_size + 1))
        
        if(self.useSavedLearnWeightData):
            # Save weights to file
            with open(inputWeights_file_path, 'wb') as file:
                pickle.dump(self.input_weights, file)
            with open(outputWeights_file_path, 'wb') as file:
                pickle.dump(self.output_weights, file)

    def classify(self, test_data):
        # Classify test data using trained neural network
        self.size_test = len(test_data)
        features_test = [list(datum.values()) for datum in test_data]
        test_set = np.array(features_test, dtype=np.int32)
        feature_test_set = test_set.transpose()

        # If the number of features in test data is different, reset activation matrices
        if feature_test_set.shape[1] != self.input_activation.shape[1]:
            self.input_activation = np.ones((self.input_size + 1, feature_test_set.shape[1]))
            self.hidden_activation = np.ones((self.hidden_size + 1, feature_test_set.shape[1]))
            self.output_activation = np.ones((self.output_size + 1, feature_test_set.shape[1]))

        # Set input activations for test data
        self.input_activation[:-1, :] = feature_test_set

        # Compute activations for hidden layer
        hidden_z = np.dot(self.input_weights, self.input_activation)
        self.hidden_activation[:-1, :] = self.sigmoid(hidden_z)

        # Compute activations for output layer
        output_z = np.dot(self.output_weights, self.hidden_activation)
        self.output_activation = self.sigmoid(output_z)

        # If output has multiple classes, return class with highest activation, else return binary classification
        if self.output_size > 1:
            return np.argmax(self.output_activation, axis=0).tolist()
        else:
            return (self.output_activation > 0.5).ravel()

    def labelMatrix(self, labels):
        # Convert labels into a matrix with 1 at true class position and 0 elsewhere
        result = np.zeros((len(self.output_classes), self.data_size))
        for i in range(self.data_size):
            result[int(labels[i]), i] = 1
        return result

# Function for basic feature extraction for digits
def digitFeatureExtractor(pic: Picture):
    return {(x, y): int(pic.getPixel(x, y) > 0) for x in range(28) for y in range(28)}

# Function for basic feature extraction for faces
def faceFeatureExtractor(pic: Picture):
    return {(x, y): int(pic.getPixel(x, y) > 0) for x in range(60) for y in range(70)}

if __name__ == '__main__':
    # Loop to allow user to select classifier type
    while True:
        usr_in = input("Enter Classifier Type (Perceptron, Neural Network): ")
        if usr_in == "Perceptron":
            classif_types = ["Perceptron"]
            break
        elif usr_in == "Neural Network":
            classif_types = ["Neural Network"]
            break
        else:
            print("Invalid input. Please try again.")
    
    # Loop to allow user to select data type
    while True:
        usr_in = input("Enter Data Type (Face, Digit): ")
        if usr_in == "Face":
            data_typ = "Face"
            out_classes = range(2)
            # Define file paths for face data
            train_path = "data/facedata/facedatatrainlabels"
            valid_path = "data/facedata/facedatavalidationlabels"
            test_path = "data/facedata/facedatatestlabels"
            raw_train = "data/facedata/facedatatrain"
            train_label = "data/facedata/facedatatrainlabels"
            raw_valid = "data/facedata/facedatavalidation"
            valid_label = "data/facedata/facedatavalidationlabels"
            raw_test = "data/facedata/facedatatest"
            test_label = "data/facedata/facedatatestlabels"
            width = 60
            height = 70
            break
        elif usr_in == "Digit":
            data_typ = "Digit"
            out_classes = range(10)
            # Define file paths for digit data
            train_path = "data/digitdata/traininglabels"
            valid_path = "data/digitdata/validationlabels"
            test_path = "data/digitdata/testlabels"
            raw_train = "data/digitdata/trainingimages"
            train_label = "data/digitdata/traininglabels"
            raw_valid = "data/digitdata/validationimages"
            valid_label = "data/digitdata/validationlabels"
            raw_test = "data/digitdata/testimages"
            test_label = "data/digitdata/testlabels"
            width = 28
            height = 28
            break
        else:
            print("Invalid input. Please try again.")

    #Number of iterations to run for each training data usage. Not applicable if using saved weights.
    NUM_ITER_STATS = 10

    # Loop to allow user to used Saved Weight on File
    while True:
        usr_in = input("Use Saved Learned Weight Data? Saying No will Run Stats Instead. (Yes, No): ")
        if usr_in == "Yes":
            useSavedLearnWeightData = True
            NUM_ITER_STATS = 1
            break
        elif usr_in == "No":
            useSavedLearnWeightData = False
            break
        else:
            print("Invalid input. Please try again.")        
    
    map_dict = {"Face": faceFeatureExtractor, "Digit": digitFeatureExtractor}

    # Loop through classifier types
    for classif_typ in classif_types:
        classifier = None
        # Initialize the classifier based on the type
        if classif_typ == "Perceptron":
            print(f"Initializing Perceptron Classifier")
        else:
            print(f"Initializing Two-Layer Neural Network Classifier")
        # Create a directory to save the plots if it doesn't exist
        if not os.path.exists('stats'):
            os.makedirs('stats')

        # Dictionary to store data for plotting
        plot_data = {
            'trainSetSz': [],
            'avg_training_times': [],
            'avg_test_errors': [],
            'std_dev_test_errors': []
        }
        # Loop through different percentages of training data used
        for trainDataUsage in [round(i * 0.1, 1) for i in range(1, 11)]:
            #Utilized for Stat Tracking!
            training_times = []
            validation_errors = []
            test_errors = []

            # Loop for random iterations
            for rand_time in range(NUM_ITER_STATS):
                tr_data, tr_labels, val_data, val_labels, tst_data, tst_labels = [None] * 6

                # Calculate sizes of datasets
                trainSetSz = int(len(open(train_path, "r").readlines()) * trainDataUsage)
                valSetSz = int(len(open(valid_path, "r").readlines()))
                testSetSz = int(len(open(test_path, "r").readlines()))

                # Generate random order for training data
                rand_ord = random.sample(range(trainSetSz), trainSetSz)

                # Load training data and labels
                raw_tr_data = load_data_randomly(raw_train, rand_ord, width, height)
                tr_labels = load_labels_randomly(train_label, rand_ord)

                # Load validation data and labels
                raw_val_data = load_data(raw_valid, valSetSz, width, height)
                val_labels = load_labels(valid_label, valSetSz)

                # Load test data and labels
                raw_tst_data = load_data(raw_test, testSetSz, width, height)
                tst_labels = load_labels(test_label, testSetSz)

                print(f"Current Classifier Type: {'Perceptron' if classif_typ == 'Perceptron' else 'Two-Layer Neural Network'}\n"
                        f"Training Data Used: {trainDataUsage * 100:.1f}%\n"
                        f"Training Set Size: {trainSetSz}\n"
                        f"Validation Set Size: {valSetSz}\n"
                        f"Test Set Size: {testSetSz}\n"
                        f"\tExtracting features...", end="")
             
                # Initialize classifier if using neural network
                if classif_typ == "Neural Network":
                    classifier = TwoLayerNeuralNetworkClassifier(out_classes, width * height, 100, len(out_classes), trainSetSz, 1, data_typ, int(trainDataUsage * 100), useSavedLearnWeightData)
                elif classif_typ == "Perceptron":
                    classifier = PerceptronClassifier(out_classes, data_typ, int(trainDataUsage * 100), useSavedLearnWeightData)
                # Extract features from raw data
                tr_data = list(map(map_dict[data_typ], raw_tr_data))
                val_data = list(map(map_dict[data_typ], raw_val_data))
                tst_data = list(map(map_dict[data_typ], raw_tst_data))

                print("Feature extraction complete.")

                print("\tTraining...", end="")
                start_t = time.time()
                # Train the classifier
                classifier.train(tr_data, tr_labels, val_data, val_labels)
                end_t = time.time()
                print("Training complete.")
                if(classifier.usedLearnedWeights):
                    print("\tUtilized Learned Weights from File.")
                else:
                    print(f"\tTraining Time: {end_t - start_t:.2f} s")

                print("\tValidating the Model...", end="")
                # Validate the model
                predictions = classifier.classify(val_data)
                correct = [predictions[i] == int(val_labels[i]) for i in range(len(val_labels))].count(True)
                print("Validation complete.")
                print(f"\t\tCorrect Predictions on Validation Set: {correct}/{len(val_labels)} ({100.0 * correct / len(val_labels):.2f}%).")

                print("\tTesting the Model...", end="")
                # Test the model
                predictions = classifier.classify(tst_data)
                correct = [predictions[i] == int(tst_labels[i]) for i in range(len(tst_labels))].count(True)
                print("Testing complete.")
                print(f"\t\tCorrect Predictions on Test Set: {correct}/{len(tst_labels)} ({100.0 * correct / len(tst_labels):.2f}%).\n")
                
                # After each iteration, calculate training time, validation error, and test error
                training_times.append(end_t - start_t)
                validation_error = 1.0 - (correct / len(val_labels))
                test_error = 1.0 - (correct / len(tst_labels))
                validation_errors.append(validation_error)
                test_errors.append(test_error)
            
            if(not useSavedLearnWeightData):
                # Calculate average of training times
                avg_training_time = np.mean(training_times)

                # Calculate average and standard deviation of test errors
                avg_test_error = np.mean(test_errors)
                std_dev_test_error = np.std(test_errors)

                # Append values to dictionary
                plot_data['trainSetSz'].append(trainSetSz)
                plot_data['avg_training_times'].append(avg_training_time)
                plot_data['avg_test_errors'].append(avg_test_error)
                plot_data['std_dev_test_errors'].append(std_dev_test_error)

                # Print the results
                print(f"Results of running {classif_typ} with {trainDataUsage * 100:.1f}% {data_typ} Data:\n")
                print(f"Average Training Time: {avg_training_time:.2f} s")
                print(f"Average Prediction Error: {avg_test_error:.2f}")
                print(f"Standard Deviation of Prediction Error: {std_dev_test_error:.2f}\n")
        
        if(not useSavedLearnWeightData):
            # Plotting
            plt.figure(figsize=(10, 6))

            # Plotting Average Training Time vs num of data points (trainSetSz)
            plt.subplot(1, 2, 1)
            plt.plot(plot_data['trainSetSz'], plot_data['avg_training_times'], marker='o')
            plt.title(f'{classif_typ} Model with {data_typ} Data')
            plt.xlabel('Number of Data Points in Training Set')
            plt.ylabel('Average Training Time (s)')

            # Plotting Average Prediction error (with standard deviation) vs num of data points (trainSetSz)
            plt.subplot(1, 2, 2)
            plt.errorbar(plot_data['trainSetSz'], plot_data['avg_test_errors'], yerr=plot_data['std_dev_test_errors'], fmt='o', linestyle='-')
            plt.title(f'{classif_typ} Model with {data_typ} Data')
            plt.xlabel('Number of Data Points in Training Set')
            plt.ylabel('Average Prediction Error')

            # Saving plots with unique filename
            plt.tight_layout()
            plt.savefig(f'stats/{classif_typ} Model with {data_typ} Data.png')
            plt.close()