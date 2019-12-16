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
#-----------------------------------------------------------------------------------------------------
# This utility script reads in the ggplot2 compatible linear stability analysis predictions
# (e.g. steady states) and returns an heatmap with columns and rows clusterd based on similarity
# The ggheatmap visualization adapt some of the code from Chris Wallace's ggheatmap script see
# https://github.com/chr1swallace/random-functions/blob/master/R/ggplot-heatmap.R
# This script needs two command line arguments. (1) file with stability analysis output (from MATLAB)
# (2) prefix for output file name
# From terminal: Rscript plot_stability_analysis.R <infile> <outfile.prefix>
#----------------------------------------------------------------------------------------------------
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
# function to check for installed packages using source
pkgTest_source <- function(x)
{
  if (!require(x,character.only = TRUE))
  {
    install.packages(x,dep=TRUE,type="source")
    if(!require(x,character.only = TRUE)) stop("Package not found")
  }
}

# test for packages. If installed load them, otherwise install then load.
pkgTest("ggplot2")
pkgTest("RColorBrewer")
pkgTest("ggthemes")
pkgTest("gridExtra")
pkgTest("ggdendro")
pkgTest("reshape2")
pkgTest("grDevices")
pkgTest("plyr")
pkgTest("pracma")
pkgTest("amap")
pkgTest_source("viridis")
pkgTest("grid")

# run auxiliary functions for plotting
mydplot <- function(ddata, row=!col, col=!row, labels=col) {
  ## plot a dendrogram
  yrange <- range(ddata$segments$y)
  yd <- yrange[2] - yrange[1]
  nc <- max(nchar(as.character(ddata$labels$label)))
  tangle <- if(row) { 0 } else { 90 }
  tshow <- col
  p <- ggplot() +
    geom_segment(data=segment(ddata), aes(x=x, y=y, xend=xend, yend=yend)) +
    labs(x = " ", y =" ") + theme_dendro()
  if(row) {
    p <- p +
      scale_x_continuous(expand=c(0.5/length(ddata$labels$x),0)) +
      coord_flip()
  } else {
    p <- p +
      theme(axis.text.x = element_text(angle = 90, hjust = 1))
  }
  return(p)
}
g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}
ggheatmap.show <- function(L, col.width=0.2, row.width=0.2) {
  grid.newpage()
  top.layout <- grid.layout(nrow = 2, ncol = 4,
                            widths = unit(c(0.5*row.width,
                                            0.5*row.width,
                                            2*row.width,
                                            row.width), "null"),
                              heights = unit(c(col.width,
                                               1-row.width), "null"))
  pushViewport(viewport(layout=top.layout))
  if(col.width>0)
    print(L$col, vp=viewport(layout.pos.col=3, layout.pos.row=1))
  if(row.width>0)
    print(L$row, vp=viewport(layout.pos.col=4, layout.pos.row=2))
  print(L$centre +
          theme(axis.line=element_blank(),
                axis.text.x=element_blank(),
                axis.text.y=element_blank(),
                axis.ticks=element_blank(),
                legend.position="none",
                panel.border=element_blank(),
                panel.grid.major=element_blank(),
                panel.grid.minor=element_blank(),
                plot.background=element_blank()),
        vp=viewport(layout.pos.col=3, layout.pos.row=2))
  print(L$left +
          theme(axis.line=element_blank(),
                    axis.text.x=element_blank(),
                     axis.text.y=element_blank(),
                     axis.ticks=element_blank(),
                     legend.position="none",
                     panel.border=element_blank(),
                     panel.grid.major=element_blank(),
                     panel.grid.minor=element_blank(),
                plot.background=element_blank()),
        vp=viewport(layout.pos.col=2, layout.pos.row=2))
  pushViewport(viewport(layout.pos.col=4, layout.pos.row=1))
  upViewport(0)
  l1 <- g_legend(L$centre)
  l1$vp$x <- unit(.055, 'npc')
  l1$vp$y <- unit(.6, 'npc')
  grid.draw(l1)
  upViewport(0)
  l2 <- g_legend(L$left)
  l2$vp$x <- unit(.06, 'npc')
  l2$vp$y <- unit(.3, 'npc')
  grid.draw(l2)
}

# run the main function
args <- commandArgs(trailingOnly=TRUE)
my_sa_file <- args[1]
my_output_file_prefix <-args[2]

cores <- as.numeric(args[3])
if( is.na(cores)){
  cores <- 2
}

my_sa <- read.csv(my_sa_file,sep="\t") # read in the data
my_sa<-na.omit(my_sa)
uPerturbationID<-unique(my_sa$PerturbationID)# unique PerturbationID array

# ---------- reference state ----------------------------------------------
my_sa.reference<-my_sa[which(my_sa$PerturbationID==min(uPerturbationID)),]#lowest PerturbationID is our reference
my_sa.reference.strains<-my_sa.reference[,5:ncol(my_sa.reference)] #keep only the cols to be clustered
xtmp1<-log10(my_sa.reference.strains+1)
rownames(xtmp1)<-my_sa.reference[,1]
if(is.null(colnames(xtmp1)))
  colnames(xtmp1) <- sprintf("col%s",1:ncol(xtmp1))
if(is.null(rownames(xtmp1)))
  rownames(xtmp1) <- sprintf("row%s",1:nrow(xtmp1))
dist.row<-Dist(xtmp1, method = "euclidean", nbproc=cores)
dist.col<-Dist(t(xtmp1), method = "euclidean", nbproc=cores)
row.hc <- hclust(dist.row,method="ward.D2")
col.hc <- hclust(dist.col,method="ward.D2")
row.dendro <- dendro_data(as.dendrogram(row.hc),type="rectangle")
col.dendro <- dendro_data(as.dendrogram(col.hc),type="rectangle")
col.plot <- mydplot(col.dendro, col=TRUE, labels=TRUE) +
  scale_x_continuous(breaks = 1:ncol(xtmp1),labels=col.hc$labels[col.hc$order]) +
  theme(plot.margin = unit(c(0,0,0,0), "lines"))
row.plot <- mydplot(row.dendro, row=TRUE, labels=FALSE) +
  theme(plot.margin = unit(rep(0, 4), "lines"))+xlab(" ")
row.ord <- match(row.dendro$labels$label, rownames(xtmp1))
col.ord <- match(col.dendro$labels$label, colnames(xtmp1))
xxtmp1 <- xtmp1[row.ord,col.ord]
xxtmp1$SID<-dimnames(xxtmp1)[[1]]
xxtmp1.m <- melt(xxtmp1)
my.colours<-viridis
xxtmp1.m.w.na<-xxtmp1.m
xxtmp1.m.w.na[xxtmp1.m.w.na==0]<-NA
names(xxtmp1.m.w.na)[3]="density"
xxtmp1.m.w.na$SID<-factor(xxtmp1.m.w.na$SID,levels=xxtmp1$SID)
xxtmp1.m.w.na$variable<-factor(xxtmp1.m.w.na$variable,levels=names(xxtmp1))

mean.hm.plot <- ggplot(xxtmp1.m.w.na, aes(variable,SID)) + geom_tile(aes(fill=log(density+1))) +
  scale_fill_gradientn(name ="",colours = my.colours(11), na.value = "transparent") +
  theme_few()+
  labs(x = "Density (log)", y = NULL) +
  theme(plot.margin = unit(rep(0, 4), "lines")) 

my.colours.f<-colorRampPalette(brewer.pal(7, "Reds"), space="Lab")
my_sa.reference.frequency<-my_sa.reference[,c(1,4)]
my_sa.reference.frequency.m<-melt(my_sa.reference.frequency,id.vars="ProfileID")
my_sa.reference.frequency.m$ProfileID<-factor(my_sa.reference.frequency.m$ProfileID,levels=xxtmp1$SID)

prob.plot<-ggplot(my_sa.reference.frequency.m,aes(variable,ProfileID,fill=value))+
  geom_tile(aes(fill=value))+
  scale_fill_gradientn(colours = my.colours.f(11), na.value = "transparent")+
  theme_few()+
  labs(x = "Frequency" , y = NULL) +
  theme(plot.margin = unit(rep(0, 4), "lines")) +
  theme(legend.title=element_blank())

combined.plot <- list(left=prob.plot,col=col.plot,row=row.plot,centre=mean.hm.plot)
pdf(sprintf("%s.%s.pdf",my_output_file_prefix,min(uPerturbationID)),width=8, height=8)
ggheatmap.show(combined.plot)
dev.off()

# reference state 
print(sprintf('number of steady states for reference %s: %s ',
              min(uPerturbationID),nrow(xtmp1)))
reference.non.zero<-repmat(0,nrow(xtmp1),ncol(xtmp1))
reference.non.zero[which(xtmp1>0)]<-1
reference.non.zero.dist<-rowSums(reference.non.zero)

#------- loop over all the other non-reference state ------------------------------
# to be consistent with the reference one the column clustering ordering remains the
# same.

for (i in seq(2,length(uPerturbationID))){
  my_sa.nrf<-my_sa[which(my_sa$PerturbationID==uPerturbationID[i]),]#lowest PerturbationID is our reference
  my_sa.nrf.strains<-my_sa.nrf[,5:ncol(my_sa.nrf)] #keep only the cols to be clustered
  xtmp2<-log10(my_sa.nrf+1)
  rownames(xtmp2)<-my_sa.nrf[,1]
  if(is.null(colnames(xtmp2)))
    colnames(xtmp2) <- sprintf("col%s",1:ncol(xtmp2))
  if(is.null(rownames(xtmp2)))
    rownames(xtmp2) <- sprintf("row%s",1:nrow(xtmp2))
  dist.row<-Dist(xtmp2, method = "euclidean", nbproc=cores)
  row.hc <- hclust(dist.row,method="ward.D2")
  row.dendro <- dendro_data(as.dendrogram(row.hc),type="rectangle")
  col.plot <- mydplot(col.dendro, col=TRUE, labels=TRUE) +
    scale_x_continuous(breaks = 1:ncol(xtmp1),labels=col.hc$labels[col.hc$order]) +
    theme(plot.margin = unit(c(0,0,0,0), "lines"))
  row.plot <- mydplot(row.dendro, row=TRUE, labels=FALSE) +
    theme(plot.margin = unit(rep(0, 4), "lines"))+xlab(" ")
  row.ord <- match(row.dendro$labels$label, rownames(xtmp2))
  col.ord <- match(col.dendro$labels$label, colnames(xtmp2))
  xxtmp2 <- xtmp2[row.ord,col.ord]
  xxtmp2$SID<-dimnames(xxtmp2)[[1]]
  xxtmp2.m <- melt(xxtmp2)
  xxtmp2.m.w.na<-xxtmp2.m
  xxtmp2.m.w.na[xxtmp2.m.w.na==0]<-NA
  names(xxtmp2.m.w.na)[3]="density"
  xxtmp2.m.w.na$SID<-factor(xxtmp2.m.w.na$SID,levels=xxtmp2$SID)
  xxtmp2.m.w.na$variable<-factor(xxtmp2.m.w.na$variable,levels=names(xxtmp1))
  
  mean.hm.plot <- ggplot(xxtmp2.m.w.na, aes(variable,SID)) + geom_tile(aes(fill=log(density+1))) +
    scale_fill_gradientn(name ="",colours = my.colours(11), na.value = "transparent") +
    theme_few()+
    labs(x = "Density (log)", y = NULL) +
    theme(plot.margin = unit(rep(0, 4), "lines")) 
  
  my_sa.nrf.frequency<-my_sa.nrf[,c(1,4)]
  my_sa.nrf.frequency.m<-melt(my_sa.nrf.frequency,id.vars="ProfileID")
  my_sa.nrf.frequency.m$ProfileID<-factor(my_sa.nrf.frequency.m$ProfileID,levels=xxtmp2$SID)
  
  prob.plot<-ggplot(my_sa.nrf.frequency.m,aes(variable,ProfileID,fill=value))+
    geom_tile(aes(fill=value))+
    scale_fill_gradientn(colours = my.colours.f(11), na.value = "transparent")+
    theme_few()+
    labs(x = "Frequency" , y = NULL) +
    theme(plot.margin = unit(rep(0, 4), "lines")) +
    theme(legend.title=element_blank())
  
  combined.plot <- list(left=prob.plot,col=col.plot,row=row.plot,centre=mean.hm.plot)
  pdf(sprintf("%s.%s.pdf",my_output_file_prefix,uPerturbationID[i]),width=8, height=8)
  ggheatmap.show(combined.plot)
  dev.off()
  
  print(sprintf('number of steady states for perturbation %s: %s ',
                uPerturbationID[i],nrow(xtmp2)))
  print(nrow(xtmp2))
  pert.non.zero<-repmat(0,nrow(xtmp2),ncol(xtmp2))
  pert.non.zero[which(xtmp2>0)]<-1
  pert.non.zero.dist<-rowSums(pert.non.zero)
  
  # perform the wilcoxon test agains the reference state
  median.reference.non.zero.dist<-median(reference.non.zero.dist)
  print(sprintf('median number of present taxa for reference: %s',median.reference.non.zero.dist))
  median.pert.non.zero.dist<-median(pert.non.zero.dist)
  print(sprintf('median number of present taxa for perturbation %s: %s',uPerturbationID[i],
                median.pert.non.zero.dist))
  wilc.test.results<-wilcox.test(reference.non.zero.dist,median.pert.non.zero.dist)
  print(wilc.test.results)
}




