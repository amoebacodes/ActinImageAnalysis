# %%
# source code: https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# %%
def getEdgeImage(image):
    edge_image = cv2.Canny(image, np.percentile(image,0.05), np.percentile(image,85))
    edge_image = cv2.dilate(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    edge_image = cv2.erode(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    return edge_image

def line_detection_non_vectorized(image, num_rhos=180, num_thetas=180, t_count=220):
  edge_image = getEdgeImage(image)
  edge_height, edge_width = edge_image.shape[:2]
  edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
  #
  d = np.sqrt(np.square(edge_height) + np.square(edge_width)) # length of diagonal of edge image
  dtheta = 180 / num_thetas # step size
  drho = (2 * d) / num_rhos # step size of rhos
  #
  thetas = np.arange(0, 180, step=dtheta) # thetas to try
  rhos = np.arange(-d, d, step=drho) # rhos to try
  #
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  #
  accumulator = np.zeros((len(rhos), len(rhos)))
  #
#   figure = plt.figure(figsize=(12, 12))
#   subplot1 = figure.add_subplot(1, 4, 1)
#   subplot1.imshow(image)
#   subplot2 = figure.add_subplot(1, 4, 2)
#   subplot2.imshow(edge_image, cmap="gray")
#   subplot3 = figure.add_subplot(1, 4, 3)
#   subplot3.set_facecolor((0, 0, 0))
#   subplot4 = figure.add_subplot(1, 4, 4)
#   subplot4.imshow(image)
  # get Hough space
  for y in range(edge_height): # loop through each pixel in edge image
    for x in range(edge_width):
      if edge_image[y][x] != 0: # if it is an edge pixel
        edge_point = [y - edge_height_half, x - edge_width_half] #[y1,x1]
        ys, xs = [], []
        for theta_idx in range(len(thetas)):
          rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx]) # rho = x1 * cos(theta) + x2 * sin(theta); a curve in Hough space
          theta = thetas[theta_idx]
          rho_idx = np.argmin(np.abs(rhos - rho)) # get a rho that is closest to rho three lines prior; approximate even if it is not a perfect line
          accumulator[rho_idx][theta_idx] += 1 # tried all theta_idx, but only some rho_idx would be incremented
          ys.append(rho)
          xs.append(theta)
        # subplot3.plot(xs, ys, color="white", alpha=0.05) # plot Hough space
  # find the lines
  return sum(sum(accumulator > t_count))
  numLines = 0
  for y in range(accumulator.shape[0]):
    for x in range(accumulator.shape[1]):
      if accumulator[y][x] > t_count: # multiple pixels on the same line in the edge image will have the same rho value in Hough space.
        numLines += 1
        
#         rho = rhos[y] # get this rho
#         theta = thetas[x] # get this theta
#         a = np.cos(np.deg2rad(theta)) # the rest is all for plotting
#         b = np.sin(np.deg2rad(theta))
#         x0 = (a * rho) + edge_width_half
#         y0 = (b * rho) + edge_height_half
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         subplot3.plot([theta], [rho], marker='o', color="yellow")
#         subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2]))

#   subplot3.invert_yaxis()
#   subplot3.invert_xaxis()

#   subplot1.title.set_text("Original Image")
#   subplot2.title.set_text("Edge Image")
#   subplot3.title.set_text("Hough Space")
#   subplot4.title.set_text("Detected Lines")
  plt.show()
  return numLines#, accumulator, rhos, thetas
