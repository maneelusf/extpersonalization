import random
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
    plt.title("Event-level importance")

    # Add colorbar for the colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
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

