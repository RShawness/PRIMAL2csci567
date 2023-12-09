import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def create_plot(directory, filename):
    figure_path = os.path.join(directory, filename)

    df = pd.read_csv(figure_path)
    x = df['Step']
    y = df['Value']
    y = savgol_filter(y, 10, 0)
    plt.clf()
    plt.plot(x, y)
    plt.xlabel('Episode')
    plt.rcParams['lines.linewidth'] = 3
    raw_name = filename.split('_')[-1]
    y_label = raw_name[:-4]
    plt.ylabel(y_label)
    plt.rcParams.update({'font.size': 18})
    figure_name = y_label + '.png'
    save_path = os.path.join(directory, "Plots/", figure_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_csv_files_with_metric(metric):
    csv_files = glob.glob(f'*{metric}*.csv')
    plt.clf()
    print(f"Found {len(csv_files)} files")
    for file in csv_files:
        df = pd.read_csv(file)
        x = df['Step']
        y = df['Value']
        y = savgol_filter(y, 10, 0)
        # color is dependent on name. If name contains "Base", use red, else use blue
        if "Base" in file:
            plt.plot(x, y, color='red', label=file)
        else:
            plt.plot(x, y, label=file, color='blue')

    plt.xlabel('Episode')
    plt.rcParams['lines.linewidth'] = 3
    y_label = metric
    plt.ylabel(y_label)
    plt.rcParams.update({'font.size': 18})
    figure_name = f'{metric}_plot.png'
    save_path = os.path.join(os.getcwd(), figure_name)
    plt.legend()
    # make figure larger
    plt.gcf().set_size_inches(10, 6)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_csv_files_with_metric(metric):
    csv_files = glob.glob(f'*{metric}*.csv')
    plt.clf()
    print(f"Found {len(csv_files)} files")
    for file in csv_files:
        df = pd.read_csv(file)
        x = df['instance']
        y = df['target_reached']
        # y = savgol_filter(y, 10, 0)
        # color is dependent on name. If name contains "Base", use red, else use blue
        # if "Base" in file:
        #     plt.plot(x, y, color='red', label=file)
        # else:
        name = file.split('_')[-1][:-4]
        # remove "continuous" from name
        name = name.replace('continuous', '')
        plt.plot(x, y, label=name)

    plt.xlabel('Number of Agents')
    plt.rcParams['lines.linewidth'] = 3
    plt.ylabel("Targets Completed")
    plt.rcParams.update({'font.size': 18})
    figure_name = f'{metric}_plot.png'
    save_path = os.path.join(os.getcwd(), figure_name)
    plt.legend()
    # make figure larger
    plt.gcf().set_size_inches(10, 6)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)


def process_directory(directory):
    try:
        os.mkdir(os.path.join(directory, 'Plots'))
    except:
        print("Plots directory already exists")
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            create_plot(directory, filename)

if __name__ == "__main__":
    # process_directory(sys.argv[1])
    plot_csv_files_with_metric('continuous')
