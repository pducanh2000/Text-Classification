import matplotlib.pyplot as plt


def plot_log(log):
    plt.figure(figsize=(20, 6))
    #     fig, axes = plt.subplots(1, 2, sharey=True, figsize=(22, 6))
    plt.subplot(1, 2, 1)
    for k in ["Train Loss", "Valid Loss"]:
        plt.plot(log[k])
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.subplot(1, 2, 2)
    for k in ["Train Acc", "Valid Acc"]:
        plt.plot(log[k])
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.show()