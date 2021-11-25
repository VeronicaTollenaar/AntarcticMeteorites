## Folder structure:
- ../Data_raw: contains the raw datasets as downloaded from their respective repositories
- ../Data_Locations: contains the locations of positive and unlabeled observations
- ../Data_Features: folder to save the values of the respective features, as xxx_at_mets.csv for positive observations and xxx_at_toclass.csv for unlabelled observations. Also contains a folder Missing_Values, where a xxxnans.csv can be saved to inspect the spatial distribution of missing values
- ../Results: folder to save the results of anlayses
- ../Figures: folder to save Figures of plotted data


## Additional files:
- bias_above200m1kmbuff_expanded_dissolved: shapefile of polygons of unlabelled observations
- meteorite_locations_raw.csv: contains locations of meteorite finds as defined in the meteoritical bulletin consulted on 05/07/2019
- meteorite_types.csv: contains meteorite names and types as defined in the meteoritical bulletin consulted on 05/07/2019
- validation_neg.csv: contains locations of negative observations used for validation
- TEST_neg.csv: contains locations of negative test observations
- TEST_pos.csv: contains locations of positive test obesrvations
- MSZs_ranked: shapefile of ranked meteorite stranding zones
- Test_neg4326: shapefile of locations used as negative test data
- Cal_neg4326: shapefile of locations used as negative calibration/validation data
- TestMSZs_pos4326: shapefile of locations used as positive test data in MSZ-level assesment
- 613MSZs: shapefile of outlines of meteorite stranding zones
- positive_classified.nc: netcdf of positive classified observations with their estimated a posteriori probabilities


## Scipts to extract locations of observations
- Grid_over_polygons.py: creates the files "locations_toclass.csv" and "locations_toclass_EE.csv"
- positive_observations_grid.py: creates the files "locations_mets.csv", "locations_mets_abbrevs.csv", "locations_mets_EE.csv", "locations_mets_exact_EE.csv", and "locations_mets_heavierthanXXXgrams.csv"


## Scripts to prepare data for analysis (Feature_xxx.py)
For each feature: extract values at observation locations and save as .csv
- Feature_DistanceOutcrops.py: creates a.o. the files "distance_to_outcrops.nc", "distanceoutcrops_at_mets.csv", and "distanceoutcrops_at_toclass.csv"
- Feature_IceThickness.py: creates a.o. the files "icethickness_at_mets.csv" and "icethickness_at_toclass.csv"
- Feature_RadarBackscatter.py: creates a.o. the files "radarbackscatter_at_mets.csv" and "radarbackscatter_at_toclass.csv"
- Feature_SurfaceSlope.py: creates a.o. the files "slope2km_at_mets.csv" and "slope2km_at_toclass.csv"
- Feature_SurfaceTemperature_percentile.py: creates a.o. the files "stempPERC99_at_mets.csv" and "stempPERC99_at_toclass.csv", refers to earth engine script to extract surface temperature observations from MODIS
- Feature_SurfaceVelocity.py: creates a.o. the files "velocities_at_mets.csv" and "velocities_at_toclass.csv"
- EE script snowfreedays: https://code.earthengine.google.com/188faacab7fe42ab3da459b26a390183: extracts the number of snow free days per meteorite stranding zone ("613MSZs" shapefile needed as asset)


## Scripts to analyse data
- Correlation_Features.py: calculates correlations between features
- Percentiles_Features_PerSite: calculates 99th percentiles of individual features, and plots distribution of values subdivided per field site


## Scripts to generate Figures/Tables
- Plot_ExhaustiveFeatureSelection_FIG3.py: plots Figure 3, needs results obtained in the exhaustive feature selection (ExhaustiveFeatureSelection_RUN.py).
- Plot_histogram_selected_features_FIG4.py: plots Figure 4, needs values of selected features ("xxx_at_mets.csv" and "xxx_at_toclass.csv")
- Plot_histogram_nonselected_features_FIGS4.py: plots Figure S4, needs values of nonselected features ("xxx_at_mets.csv" and "xxx_at_toclass.csv")
- Plot_DefinitionFeatures_FIGS2S3.py: plots Figure S2 and Figure S3, needs results obtained in the exhaustive feature selection (ExhaustiveFeatureSelection_RUN.py), where all combinations of slope and temperature definitions need to be obtained for a predefined set of features
- Plot_CrossValidation_forMap_FIG6.py: plots individual ROCs as part of Figure 6, needs results obtained during cross validation (as part of "ExhaustiveFeatureSelection.py", "ROC_values_CrossValidation_XXX_4pcs4fs0134.csv")
- Plot_CrossValidation.py: plots individual ROCs to investigate the performance of certain field sites in a single figure (vs. "Plot_CrossValidation_forMap_FIG6.py"), needs results obtained during cross validation (as part of "ExhaustiveFeatureSelection.py", "ROC_values_CrossValidation_XXX_4pcs4fs0134.csv")
- Plot_TypeMeteorites_FIGS7.py: plots Figure S7, needs raw data from meteoritical bulletin (as .csv)
- Plot_MSZs_TAB2.py: plots map of meteorite stranding zones as part of Table 2, needs values of "MSZs_ranked.shx"
- Plot_Sensitivity_FIGS5.py: plots Figures S5, needs values of all sensitivity analyses ("SensitivityTrainingdataSpecNeg.csv", "SensitivityDensitySpecNeg.csv", etc.)
- MSZlevel_assesment_TABS3S5.py: prepares Table S3, and Table S5, used for the assesment of the classification on MSZ-level. Needs "Test_neg4326.shx", "Cal_neg4326.shx", and  "TestMSZs_pos4326.shx" to generate three reorganized tables


## Scripts to calibrate classifier
- ExhaustiveFeatureSelection_RUN.py: loads functions from ExhaustiveFeatureSelection_CombinationsWithTemperatureVelocity.py and ExhaustiveFeatureSelection_AllOtherCombinations.py (see below), and runs these functions for different combinations of the definitions of the surface slope and temperature features
- ExhaustiveFeatureSelection_CombinationsWithTemperatureVelocity.py: contains function that calculates ROC curves for subsets in cross validation and a weighted average ROC curve, as well as the areas under the ROC curves for all different combinations of features that inclued the features Temperature and Velocity. Output is saved in generated subfolders in the ../Results folder.
- ExhaustiveFeatureSelection_AllOtherCombinations.py: complements ExhaustiveFeatureSelection_CombinationsWithTemperatureVelocity.py with a function that calculates ROC curves for subsets in cross validation and a weighted average ROC curve, as well as the areas under the ROC curves for all different combinations that do not contain both the feature Temperature and Velocity. Output is saved in generated subfolders in the ../Results folder.
- VariousPCs.py: allows defining the number of principal components used in the classification (that is theoretically limited by the number of positive observations due to the curse of dimensionality) and calculates ROC cruves in cross validation and a weighted average ROC curve, as well as the areas under the ROC curves
- VariousPCs_RUN.py: loads function from VariousPCs.py, and runs function for different numbers of principal components


## Scripts to classify observations
- Classify_observations.py: selects operating point by optimizing F1 score of the average ROC of a specified "version" (output of ExhaustiveFeatureSelection_CombinationsWithTemperatureVelocity.py). Classifies unlabelled observations, saves "positive_classified_xxx.csv" and "negative_classified_xxx.csv", that contains the locations of positive/negative classified observations and the corresponding estimated p(y=1|x) ("classification_value").
- Classifier_SampleCode.html: Sample code for classifying Positive and Unlabelled observations with explanation


## Scripts to investigate performance/sensitivity classifier
- Performance_metrics.py: calculates TP, FN, FP, and TN and related performance metrics of the performed classification. Needs a.o. "positive_classified_xxx.csv", "negative_classified_xxx.csv" (both output of Classify_observations.py), "TEST_neg.csv", and "TEST_pos.csv".
- ROC_classified_FIGS6.py: calculates ROC curve using the test observations (Figure S6) and compares AUCs. To plot comparison values obtained in exhaustive feature selection are needed ("slope2kmstempPERC99RandNeg/ROC_average_4pcs4fs0134.csv" and "slope2kmstempPERC99SpecNeg/ROC_average_4pcs4fs0134.csv")
- Sensitivity_Density.py: calculates ROC curve when using exact finding locations and saves results in the ../Results folder, needed for "Plot_Sensitivity_FIG5.py"
- Sensitivity_Isolated.py: calculates ROC curve when excluding isolated meteorite finds and saves results in the ../Results folder, needed for "Plot_Sensitivity_FIG5.py".
- Sensitivity_Trainingdata.py: calculates average ROC curve when excluding one field site consecutively and saves results in the ../Results folder, needed for "Plot_Sensitivity_FIGS5.py".
- Sensitivity_Wind.py: calculates average ROC curve when excluding light meteorite finds and saves results in the ../Results folder, needed for "Plot_Sensitivity_FIGS5.py".
- Compare_ROCs_ExhaustiveFeatureSelection.py: plots ROCs of Exhaustive Feature Selection to qualitatively inspect the performance (complementary to the AUC), needs results obtained in the exhaustive feature selection (ExhaustiveFeatureSelection_RUN.py)
- percentage_meteorites_found.py: calculates the percentage of meteorites that has been found based on the results of the classification, needs as input the number of positive classified observations, number of negative classified observations, the number of meteorites found today, a (range of) estimated sensitivity of the classification, and a (range of) estimated precision of the classificaiton
- Performance_metrics_calibration_TABS1.py: calculates performance metrics of the calibration data presented in Table S1, needs output of "ExhaustiveFeatureSelection_CombinationsWithTemperatureVelocity.py"





