# Vanni Bucci, Ph.D.
# Assistant Professor
# Department of Biology
# Room: 335A
# University of Massachusetts Dartmouth
# 285 Old Westport Road
# N. Dartmouth, MA 02747-2300
# Phone: (508)999-9219
# Email: vbucci@umassd.edu
# Web: www.vannibucci.org
#-------------------------------------------------------------------------
# This utility script reads in the ggplot2 compatible model parameter file
# and plots growth rates, interactions and response to perturbations as
# heatmaps (tile plots)
#------------------------------------------------------------------------
rm(list=ls())
# function to check for installed packages
pkgTest <- function(x)
{
  if (!require(x,character.only = TRUE))
  {
    install.packages(x,dep=TRUE,repos="http://cran.rstudio.com/")
    if(!require(x,character.only = TRUE)) stop("Package not found")
  }
}

#pseudoLog10 <- function(x) { asinh(x/2)/log(10) }
pseudoLog10 <- function(x){
  idx=which(x!=0)
  y=min(abs(x[idx]))
  r=log10(abs(x)/y+0.1)*sign(x)
  r}

# test for packages. If present load them
pkgTest("ggplot2")
pkgTest("RColorBrewer")
pkgTest("ggthemes")
pkgTest("gridExtra")
pkgTest("scales")

# This script needs three command line arguments. (1) file where parameter output has
# been stored from MATLAB; (2) level of significance for the edge detection; (3) output file name
# From terminal: Rscript print_model_parameters.R <infile> <significance_threshold> <outfile>
args <- commandArgs(trailingOnly=TRUE)
print(args)
my_parameters_file <- args[1]
#my_parameters_file="../../../k_honda_data/processed_data_20150714/results.parameters.txt"
#my_parameters_file="../../../gerber_c_diff_data/results_select.mean.weak.parameters.txt"
my_output_file <-args[2]
significance_threshold <- as.numeric(args[3])
if(is.na(significance_threshold)){
  significance_threshold <- 10
}
#significance_threshold=0
my_parameters <- read.csv(my_parameters_file,sep="\t")
#print(my_parameters)

# set to 0 the not significant parameters
my_parameters$value[which(my_parameters$significance<significance_threshold)]=NaN
my_parameters_growth <- subset(my_parameters,parameter_type=="growth_rate")
target<-sort(as.numeric(my_parameters_growth$value))

# growth rates
my_parameters_growth_2 <- my_parameters_growth[order(match(my_parameters_growth$value,target)),]
my_parameters_growth_2$target_taxon<-factor(my_parameters_growth_2$target_taxon,
                                            levels=
                                              my_parameters_growth$target_taxon
                                            [order(match(my_parameters_growth$value,target))])

jBuPuFun <- colorRampPalette(brewer.pal(n = 9, "RdBu"))
paletteSize <- 256
jBuPuPalette <- jBuPuFun(paletteSize)

g_growth<-ggplot(my_parameters_growth_2, aes(x = 1, y = target_taxon,
                                             #fill = sign(value)*log10(abs(value)+1)))  +
                                             fill = pseudoLog10(value)))+
  geom_tile(color="black") +
  scale_fill_gradient(low = "white",
                       high = muted(jBuPuPalette[paletteSize]),
                       name = "",na.value = "transparent",space="Lab")
g_growth<- g_growth + theme_gdocs()+
  ylab("")+ggtitle("Growth") + xlab("")+
  theme(axis.text.x = element_blank())+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# perturbations
if (length(which(my_parameters$parameter_type=="perturbation"))){
  my_parameters_perturbation <- subset(my_parameters,parameter_type=="perturbation")
  my_parameters_perturbation_2<-my_parameters_perturbation[order(match(my_parameters_perturbation$target_taxon,my_parameters_growth_2$target_taxon)),]
  my_parameters_perturbation_2$target_taxon<-factor(my_parameters_perturbation_2$target_taxon,
                                                    levels=
                                                      my_parameters_growth$target_taxon
                                                    [order(match(my_parameters_growth$value,target))])

  g_perturbation<-ggplot(my_parameters_perturbation_2, aes(x = source_taxon, y = target_taxon,
                                                           #fill = sign(value)*log10(abs(value)+1)))  +
                                                           fill = pseudoLog10(value)))+
    geom_tile(color="black") +
    scale_fill_gradient2(low = muted(jBuPuPalette[1]),
                         mid = "white",
                         high = muted(jBuPuPalette[paletteSize]),
                         midpoint = 0,
                         na.value = "transparent",
                         name = "",space="Lab")
  g_perturbation<- g_perturbation +theme_gdocs()+
    ylab("")+ggtitle("Perturbations")  + xlab("")+
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))+
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
}

# interactions
my_parameters_interaction <- subset(my_parameters,parameter_type=="interaction")
my_parameters_interaction_2<-my_parameters_interaction#[order(match(my_parameters_interaction$target_taxon,my_parameters_growth_2$target_taxon)),]

my_parameters_interaction_2$target_taxon<-factor(my_parameters_interaction_2$target_taxon,
                                                 levels=
                                                   my_parameters_growth$target_taxon
                                                 [order(match(my_parameters_growth$value,target))])

#print(my_parameters_interaction_2)

my_parameters_interaction_2$source_taxon<-factor(my_parameters_interaction_2$source_taxon,
                                                 levels=
                                                   my_parameters_growth$target_taxon
                                                 [order(-match(my_parameters_growth$value,target))])

g_interaction<-ggplot(my_parameters_interaction_2, aes(x = source_taxon,
                                                       y = target_taxon,
                                                       #fill = sign(value)*log10(abs(value)+1)))  +
                                                       fill = pseudoLog10(value)))+
                        geom_tile(color="black") +
                        scale_fill_gradient2(low = muted(jBuPuPalette[1]),
                                             mid = "white",
                                             high = muted(jBuPuPalette[paletteSize]),
                                             midpoint = 0,
                                             na.value = "transparent",
                                             name = "",space="Lab")
g_interaction<- g_interaction +theme_gdocs()+
  ylab("")+ggtitle("Interactions")+xlab("")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# put all together
gA <- ggplotGrob(g_growth)
gB <- ggplotGrob(g_interaction)
if (length(which(my_parameters$parameter_type=="perturbation"))){
  gC <- ggplotGrob(g_perturbation)
  maxh = grid::unit.pmax(gA$heights[2:5], gB$heights[2:5],gC$heights[2:5] )
  gA$heights[2:5] <- as.list(maxh)
  gB$heights[2:5] <- as.list(maxh)
  gC$heights[2:5] <- as.list(maxh)
  pdf(my_output_file,width=20, height=10)
  grid.arrange(gA, gB, gC, widths=c(1.5,2.5,1.5), ncol=3)
}else{
  maxh = grid::unit.pmax(gA$heights[2:5], gB$heights[2:5])
  gA$heights[2:5] <- as.list(maxh)
  gB$heights[2:5] <- as.list(maxh)
  pdf(my_output_file,width=20, height=10)
  grid.arrange(gA, gB,  widths=c(1.5,2.5),ncol=2)
}
