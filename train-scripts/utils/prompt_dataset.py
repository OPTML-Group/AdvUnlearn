import pandas as pd
import random

class PromptDataset:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.unseen_indices = list(self.data.index)  # 保存所有未见过的索引

    def get_random_prompts(self, num_prompts=1):
        # Ensure that the number of prompts requested is not greater than the number of unseen prompts
        num_prompts = min(num_prompts, len(self.unseen_indices))

        # Randomly select num_prompts indices from the list of unseen indices
        selected_indices = random.sample(self.unseen_indices, num_prompts)
        
        # Remove the selected indices from the list of unseen indices
        for index in selected_indices:
            self.unseen_indices.remove(index)

        # return the prompts corresponding to the selected indices
        return self.data.loc[selected_indices, 'prompt'].tolist()

    def has_unseen_prompts(self):
        # check if there are any unseen prompts
        return len(self.unseen_indices) > 0
    
    def reset(self):
        self.unseen_indices = list(self.data.index)
        
    def check_unseen_prompt_count(self):
        return len(self.unseen_indices)
    

class CoupledPromptDataset:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.unseen_indices = list(self.data.index)  # 保存所有未见过的索引

    def get_random_prompts(self, num_prompts=1):
        # Ensure that the number of prompts requested is not greater than the number of unseen prompts
        num_prompts = min(num_prompts, len(self.unseen_indices))

        # Randomly select num_prompts indices from the list of unseen indices
        selected_indices = random.sample(self.unseen_indices, num_prompts)
        
        # Remove the selected indices from the list of unseen indices
        for index in selected_indices:
            self.unseen_indices.remove(index)

        retrived_prompts = {}
        retrived_prompts['benign_prompt'] = self.data.loc[selected_indices, 'benign_prompt'].tolist()
        retrived_prompts['harmful_prompt'] = self.data.loc[selected_indices, 'harmful_prompt'].tolist()
        return retrived_prompts

    def has_unseen_prompts(self):
        # check if there are any unseen prompts
        return len(self.unseen_indices) > 0
    
    def reset(self):
        self.unseen_indices = list(self.data.index)
    
    def check_unseen_prompt_count(self):
        return len(self.unseen_indices)
    
    
# Example usage of PromptDataset and CoupledPromptDataset
if __name__ == "__main__":
    dataset = PromptDataset('./data/prompts/train/nude_body_train.csv')
    # check if there are any unseen prompts
    while dataset.has_unseen_prompts():
        # randomly select 2 prompts
        next_prompt = dataset.get_random_prompts(2)
        print(next_prompt)
    
    # Reset the dataset
    dataset.reset()
        
    coupled_dataset = CoupledPromptDataset('./data/prompts/train/harmful_benign_train.csv')
    while coupled_dataset.has_unseen_prompts():
        next_prompt = coupled_dataset.get_random_prompts(5)
        print(next_prompt)