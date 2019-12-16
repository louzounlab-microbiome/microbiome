# Vanni Bucci, Ph.D.
# Assistant Professor
# Department of Biology
# Room: 335A
# University of Massachusetts Dartmouth
# 285 Old Westport Road
# N. Dartmouth, MA 02747-2300
# Phone: (508)999-9219S
# Email: vbucci@umassd.edu
# Web: www.vannibucci.org
#-------------------------------------------------------------------------
# This utility script reads in the ggplot2 compatible model file with the
# results from matlab numerical integration of the simulated trajectories
# along with data (as estimated concentrations if available)
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
# test for packages. If present load them
pkgTest("ggplot2")
pkgTest("RColorBrewer")
pkgTest("ggthemes")
pkgTest("gridExtra")

# This script needs four command line arguments:
# (1) filename where output has been stored (full path);
# (2) flag for grouping the plot subpanels ("taxon" or "trajectory", default is trajectory)
# (3) outputfile name (full path)
# From terminal: Rscript print_simulated_trajectories.R <infile> <flag> <outfile>
args <- commandArgs(trailingOnly=TRUE)
print(args)
my_simulations_file <- args[1]
my_output_file <-args[2]
grouping_by <- args[3]
if(is.na(grouping_by)){
  grouping_by <- 'taxon'
}

my_simulations <- read.csv(my_simulations_file,sep="\t")
my_palette = colorRampPalette(rev(brewer.pal(9, "Set1")))
#my_palette = rev(brewer.pal(7, "Set1"))
scale_colour_discrete = function(...) scale_colour_manual(..., values = palette())
my_simulations$trajectory_ID=factor(my_simulations$trajectory_ID)

idata<-(which(my_simulations$type=="data"))
if (length(idata)>0){
  if (grouping_by!="trajectory"){
    g_simulations<-ggplot()+
      geom_point(data = my_simulations[my_simulations$type=="data",],
                 aes(x=time,y=abundance,color=trajectory_ID),shape=19,size=2,alpha=0.5)
      g_simulations<-g_simulations+
        geom_line(data = my_simulations[my_simulations$type=="simulation",],
                  aes(x=time,y=abundance,color=trajectory_ID),size=1,alpha=0.8)
    g_simulations<-g_simulations+
      facet_wrap(~taxon, scales = "free_y")+
      theme_few()
    palette(my_palette(length(unique(my_simulations$trajectory_ID))))
  }else{
    g_simulations<-ggplot()+
      geom_point(data = my_simulations[my_simulations$type=="data",],
                 aes(x=time,y=abundance,color=taxon),shape=19,size=2,alpha=0.5)
      g_simulations<-g_simulations+
        geom_line(data = my_simulations[my_simulations$type=="simulation",],
                  aes(x=time,y=abundance,color=taxon),size=1,alpha=1)
    g_simulations<-g_simulations+
      facet_wrap(~trajectory_ID, scales = "free_y")+
      theme_few()
    palette(my_palette(length(unique(my_simulations$taxon))))
  }
}else{
  if (grouping_by!="trajectory"){
      g_simulations<-ggplot()+
        geom_line(data = my_simulations[my_simulations$type=="simulation",],
                  aes(x=time,y=abun,color=trajectory_ID),size=1,alpha=1)
    g_simulations<-g_simulations+
      facet_wrap(~taxon, scales = "free_y")+
      theme_few()
    palette(my_palette(length(unique(my_simulations$trajectory_ID))))
  }else{
      g_simulations<-ggplot()+
        geom_line(data = my_simulations[my_simulations$type=="simulation",],
                  aes(x=time,y=abundance,color=taxon),size=1,alpha=1)
    g_simulations<-g_simulations+
      facet_wrap(~trajectory_ID, scales = "free_y")+
      theme_few()
    palette(my_palette(length(unique(my_simulations$taxon))))
  }
}

pdf(my_output_file,width=12, height=6,useDingbats=FALSE)
grid.arrange(arrangeGrob(g_simulations,widths=c(2), ncol=1))
