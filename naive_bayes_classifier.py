import pandas as pd
import math


class NaiveBayes:
    """
    Naive Bayes classifier for continuous and discrete features using pandas
    """

    def __init__(self, continuous: list):
        """
        :param continuous: list containing a bool for each feature column in the input data. True if a feature column
                           contains a continuous feature, False if discrete.
        """
        self.continuous = continuous
        self.class_priors = {}
        self.feature_stats = {}
        self.classes = []
        self.feature_names = []

    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities. Depending on self.continuous the
        features can be treated with the respective algorithm.
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """
        # Get feature names (all columns except target)
        self.feature_names = [col for col in data.columns if col != target_name]
        
        # Get unique classes
        self.classes = data[target_name].unique()
        
        # Calculate class priors P(C)
        class_counts = data[target_name].value_counts()
        total_samples = len(data)
        self.class_priors = {cls: count / total_samples for cls, count in class_counts.items()}
        
        # Initialize feature statistics storage
        self.feature_stats = {cls: {} for cls in self.classes}
        
        # For each class, calculate feature statistics
        for cls in self.classes:
            class_data = data[data[target_name] == cls]
            
            for idx, feature in enumerate(self.feature_names):
                if self.continuous[idx]:
                    # For continuous features: calculate mean and standard deviation
                    mean = class_data[feature].mean()
                    std = class_data[feature].std()
                    # Handle zero std by adding small epsilon
                    if std == 0 or pd.isna(std):
                        std = 1e-6
                    self.feature_stats[cls][feature] = {'mean': mean, 'std': std, 'type': 'continuous'}
                else:
                    # For discrete features: calculate probability for each value
                    value_counts = class_data[feature].value_counts()
                    total_count = len(class_data)
                    
                    # Get all possible values for this feature across all classes
                    all_values = data[feature].unique()
                    
                    # Calculate probabilities with Laplace smoothing
                    probabilities = {}
                    for value in all_values:
                        # Laplace smoothing: (count + 1) / (total + num_unique_values)
                        count = value_counts.get(value, 0)
                        probabilities[value] = (count + 1) / (total_count + len(all_values))
                    
                    self.feature_stats[cls][feature] = {'probabilities': probabilities, 'type': 'discrete'}

    def _gaussian_probability(self, x, mean, std):
        """
        Calculate Gaussian probability density function
        :param x: value
        :param mean: mean of the distribution
        :param std: standard deviation of the distribution
        :return: probability density
        """
        exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted (not containing label columns)
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """
        results = []
        
        for _, row in data.iterrows():
            class_probabilities = {}
            
            # Calculate posterior probability for each class
            for cls in self.classes:
                # Start with prior probability (in log space to avoid underflow)
                log_prob = math.log(self.class_priors[cls])
                
                # Multiply by conditional probabilities of each feature
                for idx, feature in enumerate(self.feature_names):
                    feature_value = row[feature]
                    feature_stat = self.feature_stats[cls][feature]
                    
                    if feature_stat['type'] == 'continuous':
                        # Gaussian probability for continuous features
                        prob = self._gaussian_probability(
                            feature_value,
                            feature_stat['mean'],
                            feature_stat['std']
                        )
                        # Avoid log(0)
                        if prob > 0:
                            log_prob += math.log(prob)
                        else:
                            log_prob += math.log(1e-10)
                    else:
                        # Discrete probability
                        prob = feature_stat['probabilities'].get(feature_value, 1e-10)
                        log_prob += math.log(prob)
                
                class_probabilities[cls] = log_prob
            
            # Convert log probabilities back to probabilities and normalize
            max_log_prob = max(class_probabilities.values())
            exp_probs = {cls: math.exp(log_prob - max_log_prob) 
                        for cls, log_prob in class_probabilities.items()}
            
            total_prob = sum(exp_probs.values())
            normalized_probs = {cls: prob / total_prob for cls, prob in exp_probs.items()}
            
            # Prediction is the class with highest probability
            prediction = max(normalized_probs, key=normalized_probs.get)
            
            # Create result row
            result_row = {f'P({cls})': normalized_probs[cls] for cls in self.classes}
            result_row['prediction'] = prediction
            results.append(result_row)
        
        return pd.DataFrame(results)

    def evaluate_on_data(self, data: pd.DataFrame, test_labels: str):
        """
        Predicts a test DataFrame (including labels) and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels: str Name of the label column in data
        :return: tuple of overall accuracy and confusion matrix values
        """
        # Get true labels
        true_labels = data[test_labels].values
        
        # Get features only (exclude label column)
        features = data.drop(columns=[test_labels])
        
        # Make predictions
        predictions_df = self.predict_probability(features)
        predicted_labels = predictions_df['prediction'].values
        
        # Calculate accuracy
        accuracy = sum(true_labels == predicted_labels) / len(true_labels)
        
        # Calculate confusion matrix
        confusion_matrix = {}
        for true_cls in self.classes:
            confusion_matrix[true_cls] = {}
            for pred_cls in self.classes:
                count = sum((true_labels == true_cls) & (predicted_labels == pred_cls))
                confusion_matrix[true_cls][pred_cls] = count
        
        return accuracy, confusion_matrix