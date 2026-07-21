# California Housing Dataset

Median house values and demographic/geographic predictors for 20,433
California block groups, derived from the 1990 US Census. Classic
regression benchmark. Originally released by Pace & Barry (1997); this
version drops the categorical \`ocean_proximity\` column (to keep the
bundled schema all-numeric, matching the rest of the package's datasets)
and removes rows with missing \`total_bedrooms\`. The regression target,
\`median_house_value\`, is the last column.

## Usage

``` r
data(california_housing)
```

## Format

A data frame with 20,433 rows and 9 variables.

## Source

Pace, R.K. and Barry, R. (1997) Sparse spatial autoregressions.
Statistics & Probability Letters, 33, 291-297.

## Details

- longitude:

  Longitude of the block group (degrees, negative = west).

- latitude:

  Latitude of the block group (degrees).

- housing_median_age:

  Median age of houses in the block group (years).

- total_rooms:

  Total number of rooms.

- total_bedrooms:

  Total number of bedrooms.

- population:

  Block group population.

- households:

  Number of households.

- median_income:

  Median household income (tens of thousands of USD).

- median_house_value:

  Regression target: median house value (USD).
