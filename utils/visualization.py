import matplotlib.pyplot as plt

def plot_accuracy(ann_acc, snn_acc):

    models = ['ANN', 'SNN']
    accuracy = [ann_acc, snn_acc]

    plt.figure(figsize=(6, 5))

    plt.bar(models, accuracy)

    plt.ylabel('Accuracy (%)')

    plt.title('ANN vs SNN Accuracy')

    plt.ylim(0, 100)

    plt.savefig('./results/accuracy_comparison.png')

    plt.close()


def plot_energy(ann_energy, snn_energy):

    models = ['ANN', 'SNN']
    energy = [ann_energy, snn_energy]

    plt.figure(figsize=(6, 5))

    plt.bar(models, energy)

    plt.ylabel('Energy (pJ)')

    plt.title('ANN vs SNN Energy')

    plt.yscale('log')

    plt.savefig('./results/energy_comparison.png')

    plt.close()
