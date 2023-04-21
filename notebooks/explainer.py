import random

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
    plt.title("Event-level importance")

    # Add colorbar for the colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
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