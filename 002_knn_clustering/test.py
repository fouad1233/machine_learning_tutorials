import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib.patches as patches

# Define K-means function with added tracking of centroids and labels
def Kmeans(features, n_clusters, max_iter=100):
    centroids = features[np.random.choice(range(features.shape[0]), n_clusters, replace=False)]
    all_centroids = [centroids]
    all_labels = []

    for _ in range(max_iter):
        # Calculate distances and assign labels
        distances = np.array([np.linalg.norm(features - centroid, axis=1) for centroid in centroids])
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([features[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # Track changes
        all_centroids.append(new_centroids)
        all_labels.append(labels)

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids, all_labels, all_centroids

# Generate sample data
features = np.vstack((
    np.random.normal([1, 1], 0.2, size=(50, 2)),
    np.random.normal([4, 4], 0.2, size=(50, 2)),
    np.random.normal([8, 1], 0.2, size=(50, 2))
))

n_clusters = 3
labels, final_centroids, all_labels, all_centroids = Kmeans(features, n_clusters)

# Initialize plot
fig, ax = plt.subplots()
colors = ['red', 'blue', 'green']  # Colors for clusters
scat = ax.scatter(features[:, 0], features[:, 1], c='grey', s=30)
centroid_scat = ax.scatter([], [], s=100, c='yellow', marker='X', edgecolor='black')

ax.set_xlim(features[:, 0].min() - 1, features[:, 0].max() + 1)
ax.set_ylim(features[:, 1].min() - 1, features[:, 1].max() + 1)
ax.set_title('K-Means Clustering Animation')

# Button setup
ax_next = plt.axes([0.8, 0.01, 0.1, 0.05])
button_next = Button(ax_next, 'Next')

# Variables to control animation steps
frame = 0

# Update function for animation
def update(frame):
    labels = all_labels[frame]
    centroids = all_centroids[frame]

    # Update scatter plot colors based on labels
    current_colors = [colors[label] for label in labels]
    scat.set_color(current_colors)
    
    # Update centroid locations
    centroid_scat.set_offsets(centroids)

    return scat, centroid_scat

# Event handler for button click
def on_button_clicked(event):
    global frame
    if frame < len(all_labels):
        update(frame)
        plt.draw()
        frame += 1
    else:
        # Show "Animation Finished" message
        ax.text(0.5, 1.05, 'Animation Finished', transform=ax.transAxes, ha="center", color="red", fontsize=12)
        button_next.label.set_text("Finished")
        button_next.on_clicked(None)  # Disable button

# Connect button to the function
button_next.on_clicked(on_button_clicked)

plt.show()
