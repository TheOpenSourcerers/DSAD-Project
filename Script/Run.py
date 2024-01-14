from src.data import Data
from src.PCA import PCA
from src.EFA import EFA
import src.graphics as gf
import config

data = Data()

pca = PCA(data)
pca.run()

efa = EFA(data)
efa.run()

input("Press anything to show graphs")

gf.showAll()
pca.visualize()


