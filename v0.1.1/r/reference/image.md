# Image Dataset

Contains 2310 observations of instances from 7 outdoor images

- Type:

  has 7 types of outdoor images, brickface, cement, foliage, grass,
  path, sky, and window.

- X1:

  the column of the center pixel of the region

- X2:

  the row of the center pixel of the region.

- X3:

  the number of pixels in a region = 9.

- X4:

  the results of a line extraction algorithm that counts how many lines
  of length 5 (any orientation) with low contrast, less than or equal to
  5, go through the region.

- X5:

  measure the contrast of horizontally adjacent pixels in the region.
  There are 6, the mean and standard deviation are given. This attribute
  is used as a vertical edge detector.

- X6:

  X5 sd

- X7:

  measures the contrast of vertically adjacent pixels. Used for
  horizontal line detection.

- X8:

  sd X7

- X9:

  the average over the region of (R + G + B)/3

- X10:

  the average over the region of the R value.

- X11:

  the average over the region of the B value.

- X12:

  the average over the region of the G value.

- X13:

  measure the excess red: (2R - (G + B))

- X14:

  measure the excess blue: (2B - (G + R))

- X15:

  measure the excess green: (2G - (R + B))

- X16:

  3-d nonlinear transformation of RGB. (Algorithm can be found in Foley
  and VanDam, Fundamentals of Interactive Computer Graphics)

- X17:

  mean of X16

- X18:

  hue mean

## Usage

``` r
data(image)
```

## Format

A data frame contains 2310 observations and 19 variables
