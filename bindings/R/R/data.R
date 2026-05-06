#' Crabs Dataset
#'
#' Measurements of 200 crabs from the genus Leptograpsus, with 50 crabs for each of two colour forms and both sexes. Measurements include the frontal lobe size, rear width, carapace length, carapace width, and body depth.
#'
#' @format A data frame with 200 rows and 8 variables:
#' \describe{
#'   \item{Type}{Species code, with `B` indicating blue and `O` indicating orange.}
#'   \item{sex}{Sex of the crab, with `M` indicating male and `F` indicating female.}
#'   \item{FL}{Size of the frontal lobe in mm.}
#'   \item{RW}{Rear width in mm.}
#'   \item{CL}{Carapace length in mm.}
#'   \item{CW}{Carapace width in mm.}
#'   \item{BD}{Body depth in mm.}
#' }
#'
#' @docType data
#' @source Campbell, N.A., and Mahon, R.J. (1974) A multivariate study of variation in two species of rock crab of genus Leptograpsus. Australian Journal of Zoology, 22, 417-425.
#' @name crabs
#'
NULL

#' Australian Crabs Dataset
#'
#' Measurements on rock crabs of the genus Leptograpsus. The dataset contains 200 observations
#'  from two species of crab (blue and orange), there are 50 specimens of each sex of each species,
#'   collected on site at Fremantle, Western Australia.
#' \describe{
#'   \item{Type}{ is the group variable and has 4 groups with the combinations of species and sex (BlueMale, BlueFemale, OrangeMale and OrangeFemale)}.
#'   \item{FL}{the size of the frontal lobe length, in mm}
#'   \item{RW}{rear width, in mm}
#'   \item{CL}{length of midline of the carapace, in mm}
#'   \item{CW}{maximum width of carapace, in mm}
#'   \item{BD}{depth of the body; for females, measured after displacement of the abdomen, in mm}
#' }
#'
#' @docType data
#' @keywords datasets
#' @name crab
#' @usage data(crab)
#' @format A data frame with 200 rows and 6 variables
#' @source Campbell, N. A. & Mahon, R. J. (1974), A Multivariate Study of Variation in Two Species of Rock Crab of genus Leptograpsus, Australian Journal of Zoology 22(3), 417 - 425.
NULL

#' Fish Catch Dataset
#'
#' There are 159 fishes of 7 species are caught and measured. Altogether there are
#' 7 variables.  All the fishes are caught from the same lake(Laengelmavesi) near Tampere in Finland.
#' \describe{
#' 	 \item{Type}{ has 7 fish groups, with 35 cases of Bream, 11 cases of Parkki, 56 cases of Perch 17 cases of Pike, 20 cases of Roach, 14 cases of Smelt and 6 cases of Whitewish.}
#' 	 \item{weight}{ Weight of the fish (in grams)}
#' 	 \item{length1}{ Length from the nose to the beginning of the tail (in cm)}
#' 	 \item{length2}{ Length from the nose to the notch of the tail (in cm)}
#' 	 \item{length3}{ Length from the nose to the end of the tail (in cm)}
#' 	 \item{height}{ Maximal height as \% of Length3}
#' 	 \item{width}{ Maximal width as \% of Length3}
#' }
#'
#' @docType data
#' @keywords datasets
#' @name fishcatch
#' @usage data(fishcatch)
#' @format A data frame with 159 rows and 7 variables
#' @source \url{https://jse.amstat.org/jse_data_archive.htm}
NULL

#' Glass Dataset
#'
#' Contains measurements 214 observations of 6 types of glass; defined in terms of their oxide content.
#' \describe{
#'   \item{Type}{ has 6 types of glasses}
#'   \item{X1}{ refractive index}
#'   \item{X2}{ Sodium (unit measurement: weight percent in corresponding oxide).}
#'   \item{X3}{ Magnesium}
#'   \item{X4}{ Aluminum}
#'   \item{X5}{ Silicon}
#'   \item{X6}{ Potassium}
#'   \item{X7}{ Calcium}
#'   \item{X8}{ Barium}
#'   \item{X9}{ Iron}
#' }
#'
#' @docType data
#' @keywords datasets
#' @name glass
#' @usage data(glass)
#' @format A data frame with 214 rows and 10 variables
NULL

#' Image Dataset
#'
#' Contains  2310 observations of instances from 7 outdoor images
#' \describe{
#'   \item{Type}{ has 7 types of outdoor images, brickface, cement,  foliage, grass, path, sky, and window.}
#'   \item{X1}{ the column of the center pixel of the region}
#'   \item{X2}{ the row of the center pixel of the region. }
#'   \item{X3}{ the number of pixels in a region = 9. }
#'   \item{X4}{ the results of a line extraction algorithm that counts how many lines of length 5 (any orientation) with low contrast, less than or equal to 5, go through the region.}
#'   \item{X5}{ measure the contrast of horizontally adjacent pixels in the region. There are 6, the mean and standard deviation are given. This attribute is used as a vertical edge detector.}
#'   \item{X6}{ X5 sd}
#'   \item{X7}{ measures the contrast of vertically adjacent pixels. Used for horizontal line detection. }
#'   \item{X8}{ sd X7}
#'   \item{X9}{ the average over the region of (R + G + B)/3}
#'   \item{X10}{ the average over the region of the R value.}
#'   \item{X11}{ the average over the region of the B value.}
#'   \item{X12}{ the average over the region of the G value.}
#'   \item{X13}{ measure the excess red: (2R - (G + B))}
#'   \item{X14}{ measure the excess blue: (2B - (G + R))}
#'   \item{X15}{ measure the excess green: (2G - (R + B))}
#'   \item{X16}{ 3-d nonlinear transformation of RGB. (Algorithm can be found in Foley and VanDam, Fundamentals of Interactive Computer Graphics)}
#'   \item{X17}{ mean of X16}
#'   \item{X18}{ hue  mean}
#' }
#'
#' @docType data
#' @keywords datasets
#' @name image
#' @usage data(image)
#' @format A data frame contains 2310 observations and 19 variables
NULL

#' Leukemia Dataset
#'
#' This dataset comes from a study of gene expression in two types of acute leukemias, acute lymphoblastic leukemia (ALL) and acute myeloid leukemia (AML). Gene expression levels were measured using Affymetrix high density oligonucleotide arrays containing 6817 human genes. A dataset containing 72 observations from 3 leukemia type groups.
#' \describe{
#'   \item{Type}{ has 3 groups with 38 cases of B-cell ALL, 25 cases of AML and 9 cases of T-cell ALL}.
#'   \item{Gene1 to Gen 40}{ gene expression levels}
#' }
#' @docType data
#' @keywords datasets
#' @name leukemia
#' @usage data(leukemia)
#' @format A data frame with 72 rows and 41 variables
#' @source Dudoit, S., Fridlyand, J. and Speed, T. P. (2002). Comparison of Discrimination Methods for the Classification of Tumors Using Gene Expression Data. Journal of the American statistical Association 97 77-87.
NULL

#' Lymphoma Dataset
#'
#' Gene expression in the three most prevalent adult lymphoid malignancies: B-cell chronic lymphocytic leukemia (B-CLL), follicular lymphoma (FL), and diffuse large B-cell lym- phoma (DLBCL). Gene expression levels were measured using a specialized cDNA microarray, the Lymphochip, containing genes that are preferentially expressed in lymphoid cells or that are of known immunologic or oncologic importance. This dataset contain 80 observations from 3 lymphoma types.
#' \describe{
#'   \item{Type}{ Group variable has 3 groups with 29 cases of B-cell ALL (B-CLL), 42 cases of diffuse large B-cell lymphoma (DLBCL) and 9 cases of follicular lymphoma (FL)}.
#'   \item{Gene1 to Gen 50}{gene expression}
#'   }
#'
#' @docType data
#' @keywords datasets
#' @name lymphoma
#' @usage data(lymphoma)
#' @format A data frame with 80 rows and 51 variables
#' @source Dudoit, S., Fridlyand, J. and Speed, T. P. (2002). Comparison of Discrimination Methods for the Classification of Tumors Using Gene Ex- pression Data. Journal of the American statistical Association 97 77-87.
NULL

#' NCI60 Dataset
#'
#' cDNA microarrays were used to examine the variation in gene expression among the 60 cell lines.  The cell lines are derived from tumors with different sites of origin. This dataset contain 61 observations and 30 feature variables from 8 different tissue types.
#'
#' \describe{
#'   \item{Type}{ has 8 different tissue types, 9 cases of breast, 5 cases of central nervous system (CNS), 7 cases pf colon, 8 cases of leukemia, 8 cases of melanoma, 9 cases of  non-small-cell lung carcinoma (NSCLC), 6 cases of ovarian and 9 cases of renal.}
#'   \item{Gene1 to Gen 30}{ gene expression information}
#' }
#'
#' @docType data
#' @keywords datasets
#' @name NCI60
#' @usage data(NCI60)
#' @format A data frame with 61 rows and 31 variables
#' @source Dudoit, S., Fridlyand, J. and Speed, T. P. (2002). Comparison of Discrimination Methods for the Classification of Tumors Using Gene Expression Data. Journal of the American statistical Association 97 77-87.
NULL

#' Olive Dataset
#'
#' Contains  572 observations and 10 variables
#' \describe{
#'   \item{Region}{Three super-groups of Italy: North, South and the island of Sardinia }
#'   \item{area}{ Nine collection areas: three from North, four from South and 2 from Sardinia}
#'   \item{palmitic}{ fatty acids percent x 100}
#'   \item{palmitoleic}{ fatty acids percent x 100 }
#'   \item{stearic}{fatty acids percent x 100}
#'   \item{oleic}{fatty acids percent x 100 }
#'   \item{linoleic}{fatty acids percent x 100}
#'   \item{linolenic}{fatty acids percent x 100  }
#'   \item{arachidic}{ fatty acids percent x 100}
#'   \item{eicosenoic}{ fatty acids percent x 100}
#' }
#'
#' @docType data
#' @keywords datasets
#' @name olive
#' @usage data(olive)
#' @format A data frame contains 573 observations and 10 variables
NULL

#' Parkinson Dataset
#'
#' A dataset containing 195 observations from 2 parkinson types.
#' \describe{
#'  \item{Type}{ Group variable has 2 groups, there are 48 cases of healthy people and 147 cases with Parkinson. The feature variables are biomedical voice measures}.
#'   \item{X1}{ Average vocal fundamental frequency}
#'   \item{X2}{ Maximum vocal fundamental frequency}
#'   \item{X3}{ Minimum vocal fundamental frequency}
#'   \item{X4}{ MDVP:Jitter(\%) measures of variation in fundamental frequency}
#'   \item{X5}{ MDVP:Jitter(Abs) measures of variation in fundamental frequency}
#'   \item{X6}{ MDVP:RAP measures of variation in fundamental frequency}
#'   \item{X7}{ MDVP:PPQ measures of variation in fundamental frequency}
#'   \item{X8}{ Jitter:DDP measures of variation in fundamental frequency}
#'   \item{X9}{ MDVP:Shimmer measures of variation in amplitude}
#'   \item{X10}{ MDVP:Shimmer(dB) measures of variation in amplitude}
#'   \item{X11}{ Shimmer:APQ3 measures of variation in amplitude}
#'   \item{X12}{ Shimmer:APQ5 measures of variation in amplitude}
#'   \item{X13}{ MDVP:APQ measures of variation in amplitude}
#'   \item{X14}{ Shimmer:DDA measures of variation in amplitude}
#'   \item{X15}{ NHR measures of ratio of noise to tonal components in the voice}
#'   \item{X16}{ HNR measures of ratio of noise to tonal components in the voice}
#'   \item{X17}{ RPDE nonlinear dynamical complexity measures}
#'   \item{X18}{ D2 nonlinear dynamical complexity measures}
#'   \item{X19}{ DFA - Signal fractal scaling exponent}
#'   \item{X20}{ spread1 Nonlinear measures of fundamental frequency variation}
#'   \item{X21}{ spread2 Nonlinear measures of fundamental frequency variation}
#'   \item{X22}{ PPE Nonlinear measures of fundamental frequency variation}
#' }
#'
#' @docType data
#' @keywords datasets
#' @name parkinson
#' @usage data(parkinson)
#' @format A data frame with 195 rows and 23 variables
#' @source \url{https://archive.ics.uci.edu/ml/datasets/Parkinsons}
NULL

#' California Housing Dataset
#'
#' Median house values and demographic/geographic predictors for 20,433 California
#' block groups, derived from the 1990 US Census. Classic regression benchmark.
#' Originally released by Pace & Barry (1997); this version drops the categorical
#' `ocean_proximity` column (to keep the bundled schema all-numeric, matching the
#' rest of the package's datasets) and removes rows with missing `total_bedrooms`.
#' The regression target, `median_house_value`, is the last column.
#'
#' \describe{
#'   \item{longitude}{Longitude of the block group (degrees, negative = west).}
#'   \item{latitude}{Latitude of the block group (degrees).}
#'   \item{housing_median_age}{Median age of houses in the block group (years).}
#'   \item{total_rooms}{Total number of rooms.}
#'   \item{total_bedrooms}{Total number of bedrooms.}
#'   \item{population}{Block group population.}
#'   \item{households}{Number of households.}
#'   \item{median_income}{Median household income (tens of thousands of USD).}
#'   \item{median_house_value}{Regression target: median house value (USD).}
#' }
#'
#' @docType data
#' @keywords datasets
#' @name california_housing
#' @usage data(california_housing)
#' @format A data frame with 20,433 rows and 9 variables.
#' @source Pace, R.K. and Barry, R. (1997) Sparse spatial autoregressions. Statistics & Probability Letters, 33, 291-297.
NULL

#' Wine Dataset
#'
#' A dataset containing 178 observations from 3 wine grown cultivares in Italy.
#'
#' \describe{
#'   \item{Type}{ Group variable has 3 groups that are 3 different wine grown cultivars in Italy. }
#'   \item{X1 to X13}{Check vbles}
#' }
#'
#' @docType data
#' @keywords datasets
#' @name wine
#' @usage data(wine)
#' @format A data frame with 178 rows and 14 variables
NULL
