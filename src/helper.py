import matplotlib.pyplot as plt


def plot_model_accuracy(train_history, path):
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.savefig(path)
    
def plot_model_loss(train_history, path):
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.savefig(path)
