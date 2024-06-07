# import packages --------------------------------------------------------------
library(data.table)
library(dplyr)
library(sf)
library(tidyr)
library(tibble)
library(pbmcapply)
library(purrr)
library(lubridate)
library(arrow)


# script functions -------------------------------------------------------------
landscapeGeneration <- function(fileParameter, aggregateArea){
  # split vector into relevant areas
  # extract landscape path
  landscapePath <- fileParameter[1]
  # extract startdate
  startDate <- as.Date(fileParameter[2])
  # extract enddate
  endDate <- as.Date(fileParameter[3])
  # read in landscape data
  landscapeData <- readRDS(landscapePath)
  # extract date of landscape file based on file name
  landscapeDate <- as.numeric(gsub(".*?([0-9]+).*", "\\1", landscapePath))
  # create landscape sequence
  landscapeDateSeq <- seq(from=startDate, to=endDate, by='months')
  # replicate landscape dataframe by date
  landscapeDateSeqRepl <- rep(landscapeDateSeq, nrow(landscapeData))
  # aggregate data
  if (aggregateArea) {
    landscapeData <- landscapeData %>%
      pivot_longer(!ID, names_to='LANDCOVER', values_to = 'area') %>%
      group_by(ID) %>%
      filter(area == max(area)) %>%
      slice(rep(1:n(), length(landscapeDateSeq))) %>%
      dplyr::select(-area)
  } else{
    landscapeData <- landscapeData %>%
      slice(rep(1:n(), length(landscapeDateSeq)))
  }
  landscapeData$DATE <- landscapeDateSeqRepl
  return(landscapeData)
}

# script parameters ------------------------------------------------------------
# list all Kriging interpolation files
krigingInterpolationList <- list.files('data/wildfire/interpolation', full.names = TRUE, include.dirs = FALSE,
                                       pattern = '.rds')
# list all landcover files in a list
landcoverList <- list.files('data/wildfire/landCover/polygon', full.names = TRUE)
# create monthly date sequence for date generation
monthlyDateSequence <- seq(from=as.Date('2010-01-01'), to=as.Date('2021-12-31'), by='months')
# determine cores to be used for multiprocessing
if (.Platform$OS.type == "windows") {
  warning('Due to Windows as OS no multiprocessing possible')
  cores <- 1
} else {
  warning('Set core variable carefully to prevent memory leakage for fork operations')
  cores <- detectCores() / 2
}
# general landscape settings
landscapeSettings <- list(c(landcoverList[1], '2010-01-01', '2012-12-01'),
                          c(landcoverList[2], '2013-01-01', '2015-12-01'),
                          c(landcoverList[3], '2016-01-01', '2018-12-01'),
                          c(landcoverList[4], '2019-01-01', '2021-12-01'))


# Base Case --------------------------------------------------------------------
# join data for weather variables
weatherDf <- data.frame()

for (krigingIteration in 1:length(krigingInterpolationList)){
  print(paste(krigingIteration, 'of', length(krigingInterpolationList), 'iteration'))
  krigingData <- krigingInterpolationList[krigingIteration]
  interpolateData <- readRDS(krigingData)
  interpolateData <- interpolateData %>%
    select(-geometry)
  if (nrow(weatherDf)==0){
    weatherDf <- interpolateData
  }else{
    weatherDf <- weatherDf %>%
      left_join(interpolateData, by=c('ID', 'DATE'))
  }
  rm(interpolateData)
  gc()
}

landscapeDataList <- pbmclapply(landscapeSettings, function(x) landscapeGeneration(x, TRUE), 
                                mc.cores = cores)
landscapeDf <- rbindlist(landscapeDataList, use.names = TRUE)
# wildfire
wildfire <- readRDS('data/wildfire/wildfire_data/wildfire.rds')

usecase3Df <- weatherDf %>%
  left_join(wildfire, by=c('ID', 'DATE')) %>%
  mutate(WILDFIRE = ifelse(is.na(WILDFIRE), 0, WILDFIRE)) %>%
  left_join(landscapeDf, by=c('ID', 'DATE')) %>%
  dplyr::select(-contains('geometry')) %>%
  dplyr::select(-c('AREA'))

arrow::write_parquet(usecase3Df, 'data/wildfire/ml_data/wildfire_base.parquet')

