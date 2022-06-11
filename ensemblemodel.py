import numpy as np
import pandas as pd

ensemble_options = ["majority", "average", "weighted"]

class EnsembleModel():
    """
    The EnsembleModel class takes trained and fit base models and aggregates their predictions on a test dataset.
    
    Attributes:
        base_model_probs (list): A list of base model prediction on test data. These are the inputs for the ensemble model
        ensemble_method (str): The type of model aggregation, options are ["majority", "average", "weighted"]
        proba_ensemble (array): The aggregated probability for the class based on the ensemble method
        predictions (array): The predicted classes for each observation
    """
    def __init__(self, base_model_probs: list, ensemble_method: str='majority'):

        # data validation checks
        assert len(base_model_probs) > 0, 'Must include list of predictions'
        num_models = len(base_model_probs)
        num_obs = base_model_probs[0].shape[0]
        self.proba_base = np.zeros([num_obs, num_models])
        
        
        for i, model_probs in enumerate(base_model_probs):
            assert model_probs.shape[1] == 2, 'Predictions must be an n x 2 array'
            
            self.proba_base[:, i] = model_probs[:,1]
        
        # Print summary message
        print('-- Ensemble model loaded --')
        print('No. of models included:', num_models)
        print('No. of predictions per model:', num_obs)
        return
        
    def predict_positive(self, ensemble_method: str='majority'):
        """
        Aggregates the base predictions using the ensemble method. Calculates a probability for each class.

        Arguments:
            ensemble_method (str): The type of model aggregation, options are ["majority", "average", "weighted"]

        Returns:
            self.proba_ensemble
        """
        assert ensemble_method in ensemble_options, 'Ensemble method must be in ["majority", "average", "weighted"]'
        self.ensemble_method = ensemble_method
        
        # Apply the ensemble method to aggregate base model predictions
        if self.ensemble_method == 'majority':
            # Round each model's prediction to nearest int and average scores per observation/row
            # For each row/observation, round to the nearest integer again to take the majority class
            self.proba_ensemble = np.mean(np.rint(self.proba_base), axis=1)

        elif self.ensemble_method == 'average':
            # Take the average probability for all model predictions per observation/row
            self.proba_ensemble = np.mean(self.proba_base, axis=1)
            
        elif self.ensemble_method == 'weighted':
            print('Not build yet.')
            return

        else:
            print('Not a valid ensemble method.')
            return
            
        return self.proba_ensemble
    
    def predict_class(self, ensemble_method: str='majority', threshold: float=0.5):
        """
        Predict the outcome for each observation based on the class probablities from predictions.

        Arguments:
            ensemble_method (str): The type of model aggregation, options are ["majority", "average", "weighted"]
            threshold (float): The number that determines whether a probability score is assigned 1 or 0

        Returns:
            self.predictions
        """
        
        # Ensure the ensemble class probabilities have been prepared
        self.proba_ensemble = self.predict_positive(ensemble_method)
        
        self.predictions = (self.proba_ensemble >= threshold).astype(int)
        
        return self.predictions

########    Classification scoring function    ########
## Calculate the metrics based on the predicted and actual labels
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

def ScoreMetrics(y_actual, y_prob, threshold=0.5):
    """ Calculate the metrics for binary classification
    
    Parameters:
        y_actual: Actual label
        y_prob: Predicted probability for Positive
        threshold: Probability score to classify as 1
        
    Returns:
        metrics: Dict of the standard scoring metrics (Acc, Prec, Rec, F1, Spec, NPV, ROC, Gini)
    """
    
    y_pred = y_prob >= threshold
    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred).ravel()
    
    metrics = {
        'Accuracy': (tp+tn)/(tp+tn+fp+fn),
        'Precision': tp/(tp+fp),
        'Recall': tp/(tp+fn), # Also known as sensitivity
        'F1': 2 * tp/(tp+fp)*tp/(tp+fn) / (tp/(tp+fp) + tp/(tp+fn)), # 2 x Precision*Recall / (Precision+Recall)
        'Specifity': tn/(tn+fp),
        'Neg-Pred Value': tn/(tn+fn),
        'ROC_AUC_Score': roc_auc_score(y_actual, y_prob),
        'Gini': 2*roc_auc_score(y_actual, y_prob) # 2 x ROC_AUC_Score
    }
    
    return metrics
    
def PrintConfusionMatrix(y_actual, y_prob, threshold=0.5):
    """ Print the confusion matrix and scoring metrics
    
    Parameters:
        y_actual: Actual label
        y_prob: Predicted probability for Positive
        threshold: Probability score to classify as 1
        
    Returns:
        None: Visual only
    """
    
    y_pred = y_prob >= threshold
    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred).ravel()
    
    print(f'----------------  ## CONFUSION MATRIX ##  ----------------')
    print(f'                                    True Labels')
    print(f'    Predicted Labels  |  True Positive  |  True Negative')
    print(f' Predicted Positive   |        {tp}     |        {fp}   ')
    print(f' Predicted Negative   |        {fn}     |        {tn}   ')
    print(f'----------------------------------------------------------')
    
    metrics = ScoreMetrics(y_actual, y_prob, threshold)
    for metric, result in metrics.items():
        print(f'{metric}: {result:.3f}')
        
    return