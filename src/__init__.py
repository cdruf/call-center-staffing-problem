from pathlib import Path

MODULE_DIR = Path(__file__).parent
print(f'Module directory {MODULE_DIR}\n')
data_folder_path = MODULE_DIR.parent / 'data'
input_file_path = data_folder_path / 'Inbound Phone Dataset.xlsx'
