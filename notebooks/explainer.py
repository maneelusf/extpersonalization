import random
<<<<<<< HEAD
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def pd_to_interaction(df, config):
    """
    Convert a pandas DataFrame row to an interaction dictionary.
    
    Args:
        df (pd.DataFrame): A single-row pandas DataFrame containing the interaction data.
        config: config of Recbole model
    Returns:
        dict: A dictionary containing the interaction data with keys "user_id", "item_id", "item_length", and "item_id_list".
    """
    user_id = (df["user_id"].values[0],)
    item_id = df["item_id"].values[0]
    item_id_list = df["item_id_list"].values[0]
    item_length = df["item_length"].values[0]

    interaction = {
        "user_id": np.array([user_id]),
        "item_id": np.array([item_id]),
        "item_length": np.array([item_length]),
        "item_id_list": np.array([item_id_list]),
    }

    # Convert the numpy arrays to PyTorch tensors
    for key, value in interaction.items():
        interaction[key] = torch.tensor(value, dtype=torch.long, device=config["device"])

    return interaction

def plot_shap_values(shap_values, sequence):
    """
    Plot the SHAP values for each event in the sequence.

    Args:
        shap_values (array-like): An array of SHAP values corresponding to the events in the sequence.
        sequence (array-like): The input sequence of events.
    """
    # Normalize SHAP values to the range [0, 1]
    norm = mcolors.Normalize(vmin=np.min(shap_values), vmax=np.max(shap_values))
    cmap = plt.get_cmap("coolwarm")

    plt.figure(figsize=(10, 10))
    for i, shap_value in enumerate(shap_values):
        plt.barh(i, shap_value, color=cmap(norm(shap_value)))

    plt.ylabel("Events in sequence")
    plt.xlabel("Shapley value")
    plt.yticks(range(len(sequence)), [f"Event -{i}" for i in range(len(sequence), 0, -1)])
=======

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_counterfactuals(history_items):
    """
    Generate a list of counterfactual sequences by removing one item at a time
    from the input history_items.

    Args:
        history_items (list): List of historical items.

    Returns:
        list: A list of counterfactual sequences.
    """
    counterfactuals = []
    for i in range(len(history_items)):
        counterfactual = history_items[:i] + history_items[i + 1 :]
        counterfactuals.append(counterfactual)
    return counterfactuals


def evaluate_counterfactuals(
    model, user_id, item_id, batch_size, counterfactuals, predictions, topk=10
):
    """
    Evaluate counterfactual sequences using the model and compute NDCG@k for each counterfactual.

    Args:
        model: The model to be used for generating predictions.
        user_id (int): The user ID.
        item_id (int): The item ID.
        batch_size (int): The batch size.
        counterfactuals (list): List of counterfactual sequences.
        predictions (tensor): Predictions for the original input.
        topk (int, optional): The number of top items to consider. Default is 10.

    Returns:
        np.array: A numpy array of NDCG@k values for each counterfactual sequence.
    """
    ndcg_values = np.zeros(len(counterfactuals))
    for i, cf in enumerate(counterfactuals):
        length = len(cf)
        sample = {
            "user_id": torch.tensor(user_id),
            "item_id": torch.tensor(item_id),
            "history_items": torch.tensor([cf]),
            "lengths": torch.tensor([length]),
            "batch_size": batch_size,
        }
        cf_predictions = model.predict_interaction(sample)
        ndcg_values[i] = ndcg_at_k(
            cf_predictions.cpu().numpy(), predictions.cpu().numpy(), k=topk
        )
    return ndcg_values


def get_most_popular_item(df):
    """
    Get the most popular item from the input DataFrame.

    Args:
        df (pandas.DataFrame): A DataFrame containing item_id column.

    Returns:
        int: The ID of the most popular item.
    """
    item_counts = df["item_id"].value_counts()
    most_popular_item = item_counts.idxmax()
    return most_popular_item


def ndcg_at_k(predicted_items, ground_truth, k=10):
    """
    Calculate NDCG@k for the given predicted items and ground truth.

    Args:
        predicted_items (list or numpy.array): A list or numpy array containing the predicted items.
        ground_truth (list or numpy.array): The ground truth items.
        k (int, optional): The number of top items to consider. Default is 10.

    Returns:
        float: The NDCG@k value.
    """
    if len(predicted_items) == 0:
        return 0

    dcg = 0
    idcg = 0
    for i, item in enumerate(predicted_items[:k]):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)

    for i in range(k):
        idcg += 1 / np.log2(i + 2)

    if idcg == 0:
        return 0
    else:
        return dcg / idcg


def plot_metric_values(metric_values, sequence):
    """
    Plot the metric values for each event in the sequence.
    Args:
        metric_values (array-like): An array of metric values corresponding to the events in the sequence.
        sequence (array-like): The input sequence of events.
    """
    # Normalize metric values to the range [0, 1]
    norm = mcolors.Normalize(vmin=np.min(metric_values), vmax=np.max(metric_values))
    cmap = plt.get_cmap("coolwarm")

    metric_values = [1 - val for val in metric_values]

    plt.figure(figsize=(10, 10))
    for i, metric_value in enumerate(metric_values):
        plt.barh(i, metric_value, color=cmap(norm(metric_value)))

    plt.ylabel("Events in sequence")
    plt.xlabel("Metric value")
    plt.yticks(
        range(len(sequence)), [f"Event -{i}" for i in range(len(sequence), 0, -1)]
    )
>>>>>>> master
    plt.title("Event-level importance")

    # Add colorbar for the colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
<<<<<<< HEAD
    plt.colorbar(sm, label="SHAP Value")
    
    plt.show()
    
def print_topk_movies(shap_values, sequence, dataset, input_interaction,movies, top_k = 10):
    """
    Prints the top-k movies and their genres based on the Shapley values and event features.

    Args:
        shap_values (torch.Tensor): The Shapley values for the events.
        events (torch.Tensor): The event features for which to print the top-k movies.
        dataset (SequenceDataset): The dataset containing the field2token_id dictionary.
    """
    # Get indices of top k absolute values
    top_k_indices = np.abs(shap_values).argsort()[::-1][:top_k]

    # Get top k values using the indices
    top_k_values = shap_values[top_k_indices]
    
    # Convert the event features to integers
    item_mapping = dict([(value, key) for key, value in dataset.field2token_id['item_id'].items()])
    top_events = [sequence[i] for i in top_k_indices]
    
    print("Query movie info:")
    q_movie = movies.loc[movies['item_id:token'] == input_interaction['item_id'].cpu().numpy()[0]].iloc[0]

    # Print the relevant information
    print("Movie Title:", q_movie['movie_title:token_seq'])
    print("Release Year:", q_movie['release_year:token'])
    print("Class:", q_movie['class:token_seq'])
    print()


    print("Top-k movies:")
    for item_id, shap_val in zip(top_events,top_k_values):
        # Get the movie information for the current item_id
        movie_info = movies.loc[movies['item_id:token'] == item_id].iloc[0]

        # Print the relevant information
        print("Movie Title:", movie_info['movie_title:token_seq'])
        print("Release Year:", movie_info['release_year:token'])
        print("Class:", movie_info['class:token_seq'])
        print("SHAP:", shap_val)
        print()

class SeqExplainer:
    def __init__(self, model):
        """
        Initialize the SeqExplainer class.

        Args:
            model: The model to be used for generating explanations.
        """
        self.model = model

    def explain(self, input_interaction, replacement_func, num_samples):
        """
        Generate SHAP values for each event in the input interaction sequence.

        Args:
            input_interaction (dict): A dictionary containing the interaction data.
            replacement_func (callable): A function to generate replacement items for perturbed sequences.
            num_samples (int): The number of samples to use for generating perturbed sequences.

        Returns:
            np.array: A numpy array containing the SHAP values for each event in the input interaction sequence.
        """
        sequence = input_interaction["item_id_list"].cpu().numpy()[0]
        n = len(sequence)
        shap_values = np.zeros(n)
        baseline = replacement_func()
        for i in range(n):
            # Generate perturbed sequences by replacing the i-th item with a new item from the replacement_func
            perturbed_sequences = []
            # for j in range(num_samples):
            #     tmp = sequence.copy()
            #     tmp[i] = baseline
            #     rand_idx = random.sample(range(n), random.randint(0, 5))
            #     tmp = [baseline if i in rand_idx else tmp[i] for i in range(n)]
            #     perturbed_sequences.append(tmp)
            tmp = sequence.copy()
            tmp[i] = baseline
            perturbed_sequences.append(tmp)
#             perturbed_sequences = [
                
#                 [item if idx != i else replacement_func() for idx, item in enumerate(sequence)]
#                 for _ in range(num_samples)
#             ]

            # Calculate the model output for the original input interaction
            original_output = self.model.predict(input_interaction).detach().cpu().numpy()
            perturbed_outputs = []

            # Calculate the model output for each perturbed input interaction
            for perturbed_sequence in perturbed_sequences:
                perturbed_input_interaction = {
                    "user_id": input_interaction["user_id"],
                    "item_id": input_interaction["item_id"],
                    "item_id_list": torch.tensor([perturbed_sequence], device=self.model.device),
                    "item_length": torch.tensor([len(perturbed_sequence)], device=self.model.device),
                }

                perturbed_output = self.model.predict(perturbed_input_interaction).detach().cpu().numpy()
                perturbed_outputs.append(perturbed_output)

            # Compute the SHAP value for the i-th event as the difference between the original output and the average perturbed output
            shap_values[i] = original_output - np.mean(perturbed_outputs)
        return shap_values

    def copy_interaction_to_cpu(self, input_interaction):
        """
        Copy the interaction dictionary to CPU.

        Args:
            input_interaction (dict): A dictionary containing the interaction data.

        Returns:
            dict: A dictionary containing the interaction data with tensors on the CPU.
        """
        cpu_input_interaction = {}
        for key, value in input_interaction.items():
            if hasattr(value, "cpu"):
                cpu_input_interaction[key] = value.cpu()
            else:
                cpu_input_interaction[key] = value
        return cpu_input_interaction

=======
    plt.colorbar(sm, label="Metric Value")

    plt.show()


class SeqExplainer:
    def __init__(self, model, all_item_ids, corpus):
        """
        Initialize the SeqExplainer class.
            Args:
        model: The model to be used for generating explanations.
        all_item_ids (list): A list of all possible item_ids in the dataset.
        corpus: The corpus containing all_df DataFrame with item_id column.
        """

        self.model = model
        self.all_item_ids = all_item_ids
        self.corpus = corpus


    def replacement_func(self):
        """
        A simple replacement function to select the most popular item from the item pool.

        Returns:
            int: The ID of the most popular item.
        """
        return get_most_popular_item(self.corpus.all_df)


    def explain(self, input_interaction, num_samples):
        """
        Generate metric values for each event in the input interaction sequence.

        Args:
            input_interaction (dict): A dictionary containing the interaction data.
            num_samples (int): The number of samples to use for generating perturbed sequences.

        Returns:
            np.array: A numpy array containing the metric values for each event in the input interaction sequence.
        """
        sequence = input_interaction["history_items"].cpu().numpy()[0]
        n = len(sequence)
        metric_values = np.zeros(n)

        for i in range(n):
            # Generate perturbed sequences by replacing the i-th item with a new item from the replacement_func
            perturbed_sequence = [
                item if idx != i else self.replacement_func()
                for idx, item in enumerate(sequence)
            ]

            # Calculate the model output for the original input interaction
            original_output = self.model.predict_interaction(input_interaction)

            # Calculate the model output for each perturbed input interaction
            perturbed_input_interaction = {
                "user_id": input_interaction["user_id"],
                "item_id": input_interaction["item_id"],
                "history_items": torch.tensor([perturbed_sequence]),
                "lengths": torch.tensor([len(perturbed_sequence)]),
                "batch_size": input_interaction["batch_size"],
            }

            perturbed_output = self.model.predict_interaction(perturbed_input_interaction)

            metric_values[i] = ndcg_at_k(
                perturbed_output.cpu().numpy(), original_output.cpu().numpy()
            )
        return metric_values
>>>>>>> master
