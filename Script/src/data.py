from src.utils.libs import *
from src.utils.utils import *
import config

class Data:
  def __init__(self):
    self.input_data = Data.get_prepared_input_data()
    self.data = Data.standardise(self.input_data)
    self.variables = self.data.columns.values
    self.observations = self.data.index.values
    self.no_observations = self.observations.size
    self.no_variables = self.variables.size

    if config.DATA_SAVE_DATA:
      self.data.to_csv(getPath("standardised_data.csv"))

    if config.DATA_SHOW_TEXT_VARIABLES:
      custom_print("Variables", join(self.variables))
    if config.DATA_SHOW_TEXT_OBSERVATIONS:
      custom_print("Observations", join(self.observations))
    if config.DATA_SHOW_TEXT_INPUT_DATA:
      custom_print("Input data", self.input_data)
    if config.DATA_SHOW_TEXT_DATA:
      custom_print("Data (Standardised)", self.data)

  def get_prepared_input_data():
    input_data = pd.read_csv(INPUT_FILE_PATH, index_col=0)
    for col_name in input_data:
      _temp = input_data.get(col_name)
      _temp = _temp.fillna(input_data.get(col_name).mean())
      _temp = _temp.round(2)
      input_data[col_name] = _temp
    return input_data

  def standardise(data: pd.DataFrame):
    for col_name in data:
      means = np.mean(a=data.get(col_name), axis=0)
      stds = np.std(a=data.get(col_name), axis=0)
      data[col_name] = (data.get(col_name) - means)/ stds
    return data