import diplib as dip
import matplotlib.pyplot as plt
import numpy as np
import cv2
from copy import deepcopy
import matplotlib.cm as cm
from PIL import Image
import random
import math


def segment_image(image):
    # Image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth image
    smoothed_image = dip.MedianFilter(gray)

    # Threshold image
    thresholded_image = dip.IsodataThreshold(smoothed_image)
    # thresholded_image.Show()

    # Distance transform calc
    distancemap = dip.EuclideanDistanceTransform(thresholded_image)

    # Local maxima calculation
    maxima = dip.WatershedMaxima(distancemap)

    # Seeded Watershed on distance map. Use inverse of the distance map
    watershed = dip.SeededWatershed(-distancemap, maxima, thresholded_image, 0)
    inverted_watershed = dip.Invert(watershed)

    # Edge object removal & Small component removal
    # removed_edge = dip.EdgeObjectsRemove(inverted_watershed)
    removed_small = dip.SmallObjectsRemove(inverted_watershed, 2)

    labeled = dip.Label(removed_small, 0)
    labeled_array = np.asarray(labeled)

    # # Random color map for labeled image
    # num_labels = np.max(labeled_array) + 1
    # cmap = cm.get_cmap('tab20b', num_labels)
    # cmap.set_under('black')
    #
    # # Create a 2x4 plot grid
    # fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    #
    # # Plot images with titles
    # axs[0, 0].imshow(image[..., ::-1])  # Convert BGR to RGB for plotting
    # axs[0, 0].set_title('Original Image')
    #
    # axs[0, 1].imshow(smoothed_image, cmap='gray')
    # axs[0, 1].set_title('Smoothed Image')
    #
    # axs[0, 2].imshow(thresholded_image, cmap='gray')
    # axs[0, 2].set_title('Thresholded Image')
    #
    # axs[0, 3].imshow(distancemap, cmap='jet')
    # axs[0, 3].set_title('Distance Transform')
    #
    # axs[1, 0].imshow(maxima, cmap='gray')
    # axs[1, 0].set_title('Seeds from distance transform')
    #
    # axs[1, 1].imshow(watershed, cmap='binary')
    # axs[1, 1].set_title('Watershed segmentated image')
    #
    # axs[1, 2].imshow(removed_small, cmap='gray')
    # axs[1, 2].set_title('Small- and border elements removed')
    #
    # axs[1, 3].imshow(labeled_array, cmap=cmap, vmin=1)
    # axs[1, 3].set_title('Labeled Image')
    #
    # # Hide tick labels for all subplots
    # for ax in axs.flat:
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    # # Show the plot
    # plt.show()

    return labeled_array


def random_rgb():
    rgb = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    return rgb


def calculate_roundness(labeled_image, label):
    # Find the contour of the object
    binary_mask = np.uint8(labeled_image == label)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure at least one contour was found
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)

        # Calculate the roundness using the formula: 4 * pi * area / perimeter^2
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        roundness = (4 * np.pi * area) / (perimeter ** 2)

        return area, perimeter, roundness

    else:
        # If no contour was found, return None
        return None


def choose_closest_cells_to_middle(labeled_image, central_points, amount_of_cells):
    # Calculate the midpoint of the image
    rows, cols = labeled_image.shape
    midpoint_x = cols // 2
    midpoint_y = rows // 2

    # Calculate the distance of each central point from the midpoint
    distances = {}
    for label, point in central_points.items():
        distance = np.sqrt((point[0] - midpoint_x) ** 2 + (point[1] - midpoint_y) ** 2)
        distances[label] = distance

    # Sort the labels based on their distances from the midpoint
    sorted_labels = sorted(distances, key=distances.get)

    # Choose the 15 closest cells to the midpoint
    chosen_labels = sorted_labels[:amount_of_cells]

    return chosen_labels


def neighbor_cell(current_central_points, previous_central_points, label):
    # Obtain x and y centroid coordinates
    prev_x, prev_y = previous_central_points[label]

    distances = {}

    for label, point in current_central_points.items():
        distance = np.sqrt((point[0] - prev_x) ** 2 + (point[1] - prev_y) ** 2)

        distances[label] = distance

    # Sort the labels based on their distances from the given point
    sorted_labels = sorted(distances, key=distances.get)

    return sorted_labels


def measurements(original_image, labeled_image):
    # Find the unique labels in the image
    labels = np.unique(labeled_image)

    # Centroids initialization
    central_points = {}

    # Roundness initialization
    roundness_values = {}

    # Intensity initialization
    intensity_values = {}

    # Area initialization
    area_values = {}

    # Perimeter initialization
    perimeter_values = {}

    # Iterate over each label and calculate the central point
    for label in labels:
        if label != 0:
            # Find the indices where the label appears in the image
            indices = np.where(labeled_image == label)

            # Calculate the central point coordinates
            central_x = np.mean(indices[1])
            central_y = np.mean(indices[0])

            # Add the central point to the dictionary
            central_points[label] = (central_x, central_y)

            # Measure the roundness of the object
            area, perimeter, roundness = calculate_roundness(labeled_image, label)
            area_values[label] = area
            perimeter_values[label] = perimeter
            roundness_values[label] = roundness

            # Measure the average intensity of the object
            intensity = np.mean(original_image[indices])
            intensity_values[label] = intensity

    return central_points, roundness_values, intensity_values, perimeter_values, area_values


def make_gif(filename, image_stack):
    imgs = [Image.fromarray(img) for img in image_stack]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(f"gifs/{filename}.gif", save_all=True, append_images=imgs[1:], duration=250, loop=5)


def distance_calc(point1_x, point2_x, point1_y, point2_y):
    return math.sqrt((point2_x - point1_x) ** 2 + (point2_y - point1_y) ** 2)


def calc_average_cell_distances(distance_list):


    # Initialize a list to store average values
    averages = []

    # Get the length of the sublists (assuming equal length)
    sublist_length = len(distance_list[0])

    # Initialize a list to store average values
    averages = []

    # Iterate over the entry indices
    for i in range(sublist_length):
        total = 0
        count = 0

        # Iterate over the sublists
        for sublist in distance_list:
            total += sublist[i]
            count += 1

        # Calculate average and append to averages list
        average = total / count
        averages.append(average)

    # Print the result
    return averages


if __name__ == "__main__":

    save_labels = []
    gif_images = []
    segment_images = []
    centroid_list = []
    label_list = []
    pixel_distance = []

    visualization_method = "PNG"

    color_list = [random_rgb() for x in range(150)]

    if visualization_method == "PNG":
        # Generate an image to visualize the chosen cells with labeled numbers
        visualization_image = cv2.imread(f"images/control/MTLn3-ctrl0029.tif")
        # visualization_image = cv2.imread(f"images/subject/MTLn3+EGF0029.tif")

    # Control group
    for number in range(30):

        # # Controlgroup
        if number < 10:
            image = cv2.imread(f"images/control/MTLn3-ctrl000{number}.tif")

        else:
            image = cv2.imread(f"images/control/MTLn3-ctrl00{number}.tif")

        # Subject group

        # if number < 10:
        #     image = cv2.imread(f"images/subject/MTLn3+EGF000{number}.tif")
        #
        # else:
        #     image = cv2.imread(f"images/subject/MTLn3+EGF00{number}.tif")

        segmented = segment_image(image)

        # Choosing random cells closest to the middle
        # Choose 15 cells closest to the middle of the image
        if number == 0:
            prev_central_points, prev_roundness_values, prev_intensity_values, prev_perimeter_values, prev_area_values = \
                measurements(image, segmented)

            previous_labels = choose_closest_cells_to_middle(labeled_image=segmented,
                                                             central_points=prev_central_points,
                                                             amount_of_cells=15)

            display_cells = previous_labels

            label_list.append(previous_labels)
            centroid_list.append(prev_central_points)
            print(prev_central_points)

        if number != 0:

            cur_central_points, cur_roundness_values, cur_intensity_values, cur_perimeter_values, cur_area_values = \
                measurements(image, segmented)

            new_labels = []
            closest_point = 0

            for count, label in enumerate(previous_labels):
                closest_point_list = neighbor_cell(previous_central_points=prev_central_points,
                                                   current_central_points=cur_central_points,
                                                   label=label)

                # To not have double points, check the closest that is not already in the list.
                for point in closest_point_list:
                    if point not in new_labels:
                        closest_point = point
                        break

                new_labels.append(closest_point)

            prev_central_points, prev_roundness_values, prev_intensity_values, prev_perimeter_values, prev_area_values = cur_central_points, cur_roundness_values, cur_intensity_values, cur_perimeter_values, cur_area_values
            previous_labels = new_labels

            display_cells = new_labels
            label_list.append(new_labels)
            centroid_list.append(prev_central_points)

        # PNG Visualization

        if visualization_method == "PNG":

            counter = 1

            distance_frame_list = []

            for count, label in enumerate(display_cells):
                current_point = centroid_list[number][label]

                try:
                    cv2.circle(visualization_image, (int(current_point[0]), int(current_point[1])), 2,
                               color_list[count], -1)

                    if number != 0:
                        prev_point = centroid_list[number - 1][label_list[number - 1][count]]

                        cv2.line(visualization_image, (int(prev_point[0]), int(prev_point[1])),
                                 (int(current_point[0]), int(current_point[1])), color_list[count], 1)

                        distance_frame_list.append(
                            distance_calc(point1_x=int(prev_point[0]), point2_x=int(current_point[0]),
                                          point1_y=int(prev_point[1]), point2_y=int(current_point[1])))

                    if number == 29:
                        cv2.putText(visualization_image, str(counter),
                                    (int(current_point[0]) - 10, int(current_point[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 2)

                except:
                    continue

                counter += 1

            pixel_distance.append(distance_frame_list)

        if visualization_method == "GIF":
            visualization_image = image

            counter = 1

            for count, label in enumerate(display_cells):
                point = centroid_list[number][label]
                cv2.circle(visualization_image, (int(point[0]), int(point[1])), 5, color_list[count], -1)
                cv2.putText(visualization_image, str(counter), (int(point[0]) - 10, int(point[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                counter += 1

            plt.imsave("starting_image.png", visualization_image)

            # Display the visualization image
            gif_images.append(visualization_image)
            segment_images.append(segmented)


    # Calculate trajectory data
    dig = calc_average_cell_distances(pixel_distance[1:])
    #print("Distance", dig)

    print(np.average(dig))





    print("")



    vel = [x / 120 for x in dig]
    # print("Vecolity pixel/second", vel)

    print(np.average(vel))





    #for frame in pixel_distance[1:]:





    if visualization_method == "PNG":
        plt.imsave("controlPNG15.png", visualization_image)

    if visualization_method == "GIF":
        make_gif("labeledCONTROL15", gif_images)
        make_gif("segmentedCONTROL15", segment_images)

    # FOR GIF
    #
    #     # Display the visualization image
    #     gif_images.append(visualization_image)
    #     segment_images.append(segmented)
    #
    #
    # make_gif("labeled", gif_images)
    # make_gif("segmented", segment_images )
