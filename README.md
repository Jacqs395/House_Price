# House_Price - Advanced Regression Techniques

The purpose of this project is to build a machine learning model that will predict the sale price of a house, using the Ames, Iowa dataset provided by Kaggle. My approach to this project is to think like a buyer. In real estate, it's well known that buyers determine the true price of a property. By examining comparable sales in the area that they are considering purchasing in, and also assigning value to amenities and features that they view as important/not important -- buyers have their own set of criteria that they use, to determine how much a home should be worth. 

In this spirit, I will train my machine learning model from the viewpoint of a buyer. 

## Dataset Description

This dataset consists of 79 variables that describe almost every aspect of residential homes in Ames, Iowa.

In this dataset, you will find: 
* SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
* MSSubClass: The building class
* MSZoning: The general zoning classification
* LotFrontage: Linear feet of street connected to property
* LotArea: Lot size in square feet
* Street: Type of road access
* Alley: Type of alley access
* LotShape: General shape of property
* LandContour: Flatness of the property
* Utilities: Type of utilities available
* LotConfig: Lot configuration
* LandSlope: Slope of property
* Neighborhood: Physical locations within Ames city limits
* Condition1: Proximity to main road or railroad
* Condition2: Proximity to main road or railroad (if a second is present)
* BldgType: Type of dwelling
* HouseStyle: Style of dwelling
* OverallQual: Overall material and finish quality
* OverallCond: Overall condition rating
* YearBuilt: Original construction date
* YearRemodAdd: Remodel date
* RoofStyle: Type of roof
* RoofMatl: Roof material
* Exterior1st: Exterior covering on house
* Exterior2nd: Exterior covering on house (if more than one material)
* MasVnrType: Masonry veneer type
* MasVnrArea: Masonry veneer area in square feet
* ExterQual: Exterior material quality
* ExterCond: Present condition of the material on the exterior
* Foundation: Type of foundation
* BsmtQual: Height of the basement
* BsmtCond: General condition of the basement
* BsmtExposure: Walkout or garden level basement walls
* BsmtFinType1: Quality of basement finished area
* BsmtFinSF1: Type 1 finished square feet
* BsmtFinType2: Quality of second finished area (if present)
* BsmtFinSF2: Type 2 finished square feet
* BsmtUnfSF: Unfinished square feet of basement area
* TotalBsmtSF: Total square feet of basement area
* Heating: Type of heating
* HeatingQC: Heating quality and condition
* CentralAir: Central air conditioning
* Electrical: Electrical system
* 1stFlrSF: First Floor square feet
* 2ndFlrSF: Second floor square feet
* LowQualFinSF: Low quality finished square feet (all floors)
* GrLivArea: Above grade (ground) living area square feet
* BsmtFullBath: Basement full bathrooms
* BsmtHalfBath: Basement half bathrooms
* FullBath: Full bathrooms above grade
* HalfBath: Half baths above grade
* Bedroom: Number of bedrooms above basement level
* Kitchen: Number of kitchens
* KitchenQual: Kitchen quality
* TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
* Functional: Home functionality rating
* Fireplaces: Number of fireplaces
* FireplaceQu: Fireplace quality
* GarageType: Garage location
* GarageYrBlt: Year garage was built
* GarageFinish: Interior finish of the garage
* GarageCars: Size of garage in car capacity
* GarageArea: Size of garage in square feet
* GarageQual: Garage quality
* GarageCond: Garage condition
* PavedDrive: Paved driveway
* WoodDeckSF: Wood deck area in square feet
* OpenPorchSF: Open porch area in square feet
* EnclosedPorch: Enclosed porch area in square feet
* 3SsnPorch: Three season porch area in square feet
* ScreenPorch: Screen porch area in square feet
* PoolArea: Pool area in square feet
* PoolQC: Pool quality
* Fence: Fence quality
* MiscFeature: Miscellaneous feature not covered in other categories
* MiscVal: $Value of miscellaneous feature
* MoSold: Month Sold
* YrSold: Year Sold
* SaleType: Type of sale
* SaleCondition: Condition of sale

# About Ames, Iowa

Ames is a city in Story County, Iowa. It is about 30 miles north of Des Moines, the state capital. The cross country line of the Union Pacific Railroad passes thru Ames, as well as the South Skunk River and the Ioway Creek. 

Ames has a total of 24.27 square miles. Land accounts for 24.21 square miles and 0.06 square miles is water.

The population was 65,686 as of July 2023, according to the US Census Bureau. This is a 1% reduction from April 2020. Over 11.5% of residents are under 18 years old; and people 65 years or older account for 10.7%. Almost 80% of the population identifies as "White alone, not Hispanic or Latino." 

Ames is the 9th most populous city in Iowa.

From 2019-2023, 42.9% of the population lived in "owner-occupied" housing and the median value of those units was $263,800.

The mean travel time to work, in minutes, was 16 minutes (for 2019-2023).

The median household income in 2023 dollars was $60,102 and approximately 25.9% of the population was living in poverty. 

Ames is home to Iowa State University. Iowa States' student population is approximately 30,177 students, which is about half of Ames' population. If we were to deduct the college-student population, the true population count for Ames would be closer to 35k. 

On average, the warmest month in Ames is July and the coldest is January. 

# Summary
In the EDA phase, I created histograms for the numerical features to see how the values were spread out. Some columns, like `SalePrice` and `GrLivArea` were right-skewed.

I compared `SalePrice`, `OverallQual` and `YearBuilt` and found that homes with higher overall quality had higher sale prices. This was expected. 

I checked for correlations between numerical features and `SalePrice'.

`SalePrice` was right-skewed, so I made it a point to log transform it to a normal distribution, ahead of applying the machine learning model.

In the interest of time, I decided to considerably reduce the number of features that I would use in my final product. I dropped everything except for `OverallQual`, `YearBuilt`, `YearRemodAdd`, `TotalBsmtSF`, `GrLivArea`, `FullBath`, `GarageCars`, `SalePrice`,`ExterQual`, `BsmtQual`, and `KitchenQual`. In hindsight, there are a few other features that I wished I had kept. 

`OrdinalEncoder` was used to convert quality ratings into numbers, based on an ordered scale from Poor to Excellent.

* ## Models Used:
    * Random Forest Regressor
    * Gradient Booster Regressor
    * XGBoost Regressor

* ## Evaluation Metrics
I used Root Mean Squared Error (RMSE) and R-squared scored to measure how well the models performed. 

* ## Results
All 3 models performed similarly. XGBoost was slightly ahead after hyperparameter tuning, but only by a sliver.

* ## Best Model
XG Boost ended up being the best model, after tuning. 

* ## Reflections
I wish I had more time to dive into this dataset. In truth, out of all my projects thus far, this particular one was rushed. I was pressed for time and did not get to give this project the attention that it deserved. I plan to return to it and make some changes, at a slower pace.
