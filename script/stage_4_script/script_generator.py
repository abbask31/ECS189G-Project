from code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator
from code.stage_4_code.Dataset_Loader import Dataset_Loader

data_obj = Dataset_Loader("Joke Dataset", '')
data_obj.dataset_source_folder_path = r'data\stage_4_data\text_classification'
data_obj.task = 'generator'
data_obj.dataset_source_folder_path = r'data\stage_4_data\text_generation'
data_obj.dataset_source_file_name = r'\data'

data = data_obj.load()

method_obj = Method_RNN_Generator()
method_obj.data = data['train']
method_obj.vocab_path = r'data\stage_4_data\text_generation\loaded_data\vocab.json'
method_obj.idx_word_path = r'data\stage_4_data\text_generation\loaded_data\idx_to_word.json'

print('-----Start Training-----')
method_obj.train_model()

print('-----Start Testing-----')
# Modify this to add any starting words to test model
starting_words = [['what', 'do', 'you'], ['what', 'did', 'the'], ['what', 'does', 'a']]

x = ''
while (x != 'X'):
    starting_words = input("Enter starting words separated by spaces: ").strip().split()
    print(method_obj.generate_text(starting_words))
    x = input("To continue, press enter\nTo exit, press X.\n")
