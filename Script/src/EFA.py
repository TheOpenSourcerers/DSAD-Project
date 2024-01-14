from src.utils.libs import *
from src.utils.utils import *
import src.graphics as gf
from src.data import Data as DT
from src.PCA import PCA
import factor_analyzer as fa

class EFA(PCA):
  def __init__(self, Data: DT):
     super().__init__(Data)
     self.correlation_coefficients = np.corrcoef(self.data, rowvar=False)

  def BarlettSphericity(self):
    self.no_factors = np.shape(self.factor_loadings)[0]

    if config.EFA_SHOW_TEXT_FACTORS:
      custom_print("EFA - Number of factors", self.no_factors, False)

    self.factor_loadings_diag = np.diag(self.factor_loadings)
    if config.EFA_SHOW_TEXT_DIAG_FACTOR_LODAGINS:
      custom_print("EFA - Factor loadings", self.factor_loadings_diag, False, True)

    self.estimated_covariance = self.factor_loadings @ np.transpose(self.factor_loadings) + self.factor_loadings_diag
    if config.EFA_SHOW_TEXT_ESTIMATED_COVARIANCE:
      custom_print("EFA - Estimated covariance", self.estimated_covariance, False, True)

    self.estimated_inverse_covariance = np.linalg.inv(self.estimated_covariance) @ self.correlation_coefficients
    if config.EFA_SHOW_TEXT_INVERSE_ESTIMATED_COVARIANCE:
      custom_print("EFA - Estimated inverse covariance", self.estimated_inverse_covariance, False, True)

    self.estimated_inverse_covariance_det = np.linalg.det(self.estimated_inverse_covariance)
    if config.EFA_SHOW_TEXT_INVERSE_EST_COVARIANCE_DET:
      custom_print("EFA - Estimated inverse covariance determinant", self.estimated_inverse_covariance_det, False)

    self.Success = True
    if self.estimated_inverse_covariance_det > 0:
      self.inverse_covariance_trace = np.trace(self.estimated_inverse_covariance)

      # Bartlett's test
      self.chi2 = (self.DataObj.observations.size - 1 - (2 * self.no_factors - 4 * self.DataObj.no_variables - 5) / 6) * \
                          (self.inverse_covariance_trace - np.log(abs(self.estimated_inverse_covariance_det)) - self.no_factors)
      self.dof = self.DataObj.no_variables * (self.DataObj.no_variables - 1) / 2

      self.pval = 1 - scp.stats.chi2.cdf(self.chi2, self.dof)
      if config.EFA_SHOW_TEXT_BARLETT_RESULT:
        custom_print("EFA - Barlett Sphericity Result", f"Result: {Barlett_Interpretation(self.chi2, self.pval)}\nChi squared: {self.chi2}, P-Value: {self.pval}", False)
    else:
        self.Success = False
        self.chi2 = self.pval = np.nan
        if config.EFA_SHOW_TEXT_BARLETT_RESULT:
          custom_print("EFA - Barlett Sphericity Result", "The estimated inverse covariance matrix determinant was 0.", False)

    return self.chi2, self.pval

  def kaiser_meyer_olkin(self):
    self.partial_correlations = np.array([[scp.stats.pearsonr(self.data[i], self.data[j])[0] for j in range(self.data.shape[1])] for i in range(self.data.shape[1])])
    if np.linalg.det(self.correlation_coefficients) < 0.01:
      self.Success = False
      if config.EFA_SHOW_TEXT_KAISER_RESULT:
        custom_print("EFA - Keiser-Meyer-Olkin Result", f"The partial correlations matrix' determinant was {np.linalg.det(self.correlation_coefficients)}.")
      return -1
    self.anti_image = np.linalg.inv(self.correlation_coefficients)
    self.anti_image_diag = np.diag(self.anti_image)
    self.original_diag = np.diag(np.corrcoef(self.data.T))
    self.kmo_index = sum(self.original_diag**2) / (sum(self.original_diag**2) + sum(self.anti_image_diag**2))


    kmo_j = [None]*self.correlation_coefficients.shape[1]
    #KMO per variable (diagonal of the spss anti-image matrix)
    for j in range(0, self.correlation_coefficients.shape[1]):
        kmo_j_num = np.sum(self.correlation_coefficients[:,[j]] ** 2) - self.correlation_coefficients[j,j] ** 2
        kmo_j_denom = kmo_j_num + np.sum(self.partial_correlations[:,[j]] ** 2) - self.partial_correlations[j,j] ** 2
        kmo_j[j] = kmo_j_num / kmo_j_denom

    if config.EFA_SHOW_TEXT_KAISER_RESULT:
      custom_print("EFA - Keiser-Meyer-Olkin result", f"Result: {KMO_Interpretation(self.kmo_index)}\nThe KMO index was {self.kmo_index}\nThe kmo indeces are: {kmo_j}")
    self.Success = True
    return self.kmo_index

  def run(self):
    self.BarlettSphericity()
    self.kaiser_meyer_olkin()