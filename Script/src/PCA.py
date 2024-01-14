from src.utils.libs import *
from src.utils.utils import *
import src.graphics as gf
from src.data import Data as DT

class PCA:
  def __init__(self, Data: DT):
    self.DataObj = Data
    self.data = Data.data.values

    # Compute variance/covariance of data
    cov = np.cov(m=self.data, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(a=cov)

    # Sort the eigen values and vectors in descending order
    k_desc = [k for k in reversed(np.argsort(eigen_values))]
    self.eigen_values = eigen_values[k_desc]
    self.eigen_vectors = eigen_vectors[:, k_desc]

    # Regularization of eigen vectors
    for j in range(self.eigen_vectors.shape[1]):
        minCol = np.min(a=self.eigen_vectors[:, j], axis=0)
        maxCol = np.max(a=self.eigen_vectors[:, j], axis=0)
        if np.abs(minCol) > np.abs(maxCol):
            self.eigen_vectors[:, j] = (-1) * self.eigen_vectors[:, j]

    self.principal_components = self.data @ self.eigen_vectors
    self.factor_loadings = self.eigen_vectors * np.sqrt(self.eigen_values)
    self.explained_variance = pow(self.principal_components, 2)
    self.scores = self.principal_components / np.sqrt(self.eigen_values)
    self.qualitative_observations = np.transpose(self.explained_variance.T / np.sum(self.explained_variance, axis=1))
    self.contributions_of_observations = self.explained_variance / (self.data.shape[0] * self.eigen_values)
    self.cummulative_scores = np.cumsum(a=np.square(self.scores), axis=1)
    self.components = ['C'+str(i+1) for i in range(self.principal_components.shape[1])]

  def run(self):
    self.print()

    # Variance
    if config.PCA_SHOW_GRAPH_VARIANCE:
      PCA.principalComponentsPlot(eigenvalues=self.eigen_values)

    # Principal Components
    principal_components = pd.DataFrame(data=self.principal_components, index=self.DataObj.observations, columns=self.components)
    if config.PCA_SAVE_PRINCIPAL_COMPONENTS:
      principal_components.to_csv(getPath("PCA/principal_components.csv"))

    # Factor loadings
    factor_loadings = pd.DataFrame(data=self.factor_loadings, index=self.DataObj.variables, columns=self.components)
    if config.PCA_SAVE_FACTOR_LOADINGS:
      factor_loadings.to_csv(getPath("PCA/factor_loadings.csv"))
    if config.PCA_SHOW_GRAPH_FACTOR_LOADINGS:
      gf.correlogram(matrix=factor_loadings, title="Correlogram of factor loadings")

    # Scores
    scores = pd.DataFrame(data=self.scores, index=self.DataObj.observations, columns=self.components)
    if config.PCA_SAVE_SCORES:
      scores.to_csv(getPath("PCA/scores.csv"))
    if config.PCA_SHOW_GRAPH_SCORES:
      gf.correlogram(matrix=scores, title='Correlogram of scores')

    # Quality of points representations
    qualitative_observations = pd.DataFrame(data=self.qualitative_observations, index=self.DataObj.observations, columns=self.components)
    if config.PCA_SAVE_QUALITY_OF_POINTS:
      qualitative_observations.to_csv(getPath("PCA/qualitative_observations.csv"))
    if config.PCA_SHOW_GRAPH_QUALITY_OF_POINTS:
      gf.correlogram(matrix=qualitative_observations, title='Quality of points representation')

    # Contribution of observations to the axes' variance
    contributions_of_observations = pd.DataFrame(data=self.contributions_of_observations, index=self.DataObj.observations, columns=self.components)
    if config.PCA_SAVE_CONTRIBUTION_OF_OBSERVATIONS:
      contributions_of_observations.to_csv(getPath("PCA/contributions_of_observations.csv"))
    if config.PCA_SHOW_GRAPH_CONTRIBUTION_OF_OBSERVATIONS:
      gf.correlogram(matrix=contributions_of_observations, title="Contribution of observations to the axes' variance")

    # Commmonalities
    commonalities = pd.DataFrame(data=self.cummulative_scores, index=self.DataObj.observations, columns=self.components)
    if config.PCA_SAVE_COMMONALITIES:
      commonalities.to_csv(getPath("PCA/commonalities.csv"))
    if config.PCA_SHOW_GRAPH_COMMONALITIES:
      gf.correlogram(matrix=commonalities, title='Coorelogram of commonalities')

  def print(self):
    if config.PCA_SHOW_TEXT_DATA:
      custom_print("PCA - Data", self.data, False, True)
    if config.PCA_SHOW_TEXT_EIGEN_VALUES:
      custom_print("PCA - Eigen values", self.eigen_values, False, True)
    if config.PCA_SHOW_TEXT_EIGEN_VECTORS:
      custom_print("PCA - Eigen vectors", self.eigen_vectors, False, True)
    if config.PCA_SHOW_TEXT_PRINCIPAL_COMPONENTS:
      custom_print("PCA - Principal components", self.principal_components, False, True)
    if config.PCA_SHOW_TEXT_FACTOR_LOADINGS:
      custom_print("PCA - Factor loadings", self.factor_loadings, False, True)
    if config.PCA_SHOW_TEXT_EXPLAINED_VARIANCE:
      custom_print("PCA - Explained variance", self.explained_variance, False, True)
    if config.PCA_SHOW_TEXT_SCORES:
      custom_print("PCA - Scores", self.scores, False, True)
    if config.PCA_SHOW_TEXT_QUALITATE_OBSERVATIONS:
      custom_print("PCA - Qualitate observations", self.qualitative_observations, False, True)
    if config.PCA_SHOW_TEXT_CONTRIBUTIONS_OF_OBSERVATIONS:
      custom_print("PCA - Contributions of observations", self.contributions_of_observations, False, True)
    if config.PCA_SHOW_TEXT_CUMMULATIVE_SCORES:
      custom_print("PCA - Cummulative scores", self.cummulative_scores, False, True)

  def principalComponentsPlot(eigenvalues=None):
    XLabel='Principal components'
    YLabel='Eigenvalues (variance)'
    title='Explained variance by the principal components'

    plt.figure(title, figsize=(13, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')

    components = ['C'+str(j+1) for j in range(eigenvalues.shape[0])]
    plt.plot(components, eigenvalues, 'bo-')
    plt.axhline(y=1, color='r')
