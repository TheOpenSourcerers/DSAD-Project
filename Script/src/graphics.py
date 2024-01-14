from src.utils.libs import *

def correlogram(matrix=None, dec=1, title='Correlogram', valmin=-1, valmax=1):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrix, dec), vmin=valmin, vmax=valmax, cmap='bwr', annot=True)

def correlationCircle(matrix=None, V1=0, V2=1, dec=1, XLabel=None, YLabel=None, minVal=-1, maxVal=1, title='Correlation Circle'):
    plt.figure(title, figsize=(8, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    T = [t for t in np.arange(0, np.pi*2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y)
    plt.axhline(y=0, color='g')
    plt.axvline(x=0, color='g')
    if XLabel==None or YLabel==None:
        if isinstance(matrix, pd.DataFrame):
            plt.xlabel(matrix.columns[V1], fontsize=14, color='k', verticalalignment='top')
            plt.ylabel(matrix.columns[V2], fontsize=14, color='k', verticalalignment='bottom')
        else:
            plt.xlabel('Var '+str(V1+1), fontsize=14, color='k', verticalalignment='top')
            plt.ylabel('Var '+str(V2+1), fontsize=14, color='k', verticalalignment='bottom')
    else:
        plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')
        plt.ylabel(YLabel, fontsize=14, color='k', verticalalignment='bottom')

    if isinstance(matrix, np.ndarray):
        plt.scatter(x=matrix[:, V1], y=matrix[:, V2], c='r', vmin=minVal, vmax=maxVal)
        for i in range(matrix.shape[0]):
            plt.text(x=matrix[i, V1], y=matrix[i, V2], s='(' +
                    str(np.round(matrix[i, V1], dec))
                    + ', ' + str(np.round(matrix[i, V2], dec)) + ')')

    if isinstance(matrix, pd.DataFrame):
        plt.scatter(x=matrix.iloc[:, V1], y=matrix.iloc[:, V2], c='b', vmin=minVal, vmax=maxVal)
        for i in range(matrix.values.shape[0]):
            plt.text(x=matrix.iloc[i, V1], y=matrix.iloc[i, V2], s='(' +
                    str(np.round(matrix.iloc[i, V1], dec))
                     + ', ' + str(np.round(matrix.iloc[i, V2], dec)) + ')')

def biplot(x: np.ndarray, y: np.ndarray, xlabel='X', ylabel='Y', title='Biplot', l1=None, l2=None):
    f = plt.figure(figsize=(12, 8))
    ax = f.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(x[:, 0], x[:, 1], c='r', label='Set X')
    ax.scatter(y[:, 0], y[:, 1], c='b', label='Set Y')
    if l1 is not None:
        for i in range(len(l1)):
            ax.text(x[i, 0], x[i, 1], l1[i])
    if l2 is not None:
        for i in range(len(l2)):
            ax.text(y[i, 0], y[i, 1], l2[i])
    ax.legend()

def intensity_map(R2, dec=1, title='Intensity Map', valmin=None, valmax=None,):
  plt.figure(title, figsize=(8, 7))
  plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
  sb.heatmap(data=np.round(R2, dec), vmin=valmin, vmax=valmax,
              cmap = 'Blues', annot = True)

def showAll():
  plt.show()