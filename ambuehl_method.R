library(data.table)
library(ggplot2)
library(deldir)
library(igraph)
library(doParallel)
library(foreach)
library(doSNOW)
library(plyr)
library(pracma)
library(scales)
library(svglite)
library(dplyr)

rm(list = ls())
options(stringsAsFactors = FALSE)

JINT=1

# parameters to play with
# set the minimum detector number of one cluster
# cluster number related
min.nr.det <- 10
min.clust <- 2
max.clust <- 25

# repeated times
used.clust <- 50
resamp.times <- 500


# load the detector data
dets <- read.csv('./data/Zurich/input/detectors_loc.csv')
setDT(dets)
detlist <- copy(dets)


# load the measurement data
meas <- read.csv('./data/Zurich/input/detectors_input.csv')
setDT(meas)


##### k-measns ####

# create the coordinate dataframe
coord <- data.frame(dets$detid, dets$lat, dets$long)
colnames(coord) <- c('detid', 'lat', 'long')
setDT(coord)

coord_clust <- coord
coord <- data.frame(dets$lat, dets$long)
colnames(coord) <- c('lat', 'long')

# set random seeds here
set.seed(96)

system.time({
  for(i in 1:100){
    print(i)

    centers <- coord[sample(1:nrow(coord),size = sample(min.clust:max.clust,1),replace = F),]

    clust <- data.table(clust=kmeans(coord,centers = centers)$cluster)
    clust[,N:=.N,by=clust]
    if(min(clust$N)>min.nr.det){
      coord_clust <- cbind(coord_clust,clust=clust$clust)
    }

  }
})


# combine the measurement dataframe with detector information
meas_orig <- meas
meas <- merge(detlist, meas, by="detid")
setDT(meas)

# estimate necessary columns: RELPOS, RELINT
intervallby <- 1/JINT
meas$RELPOS <- meas$pos/meas$length

meas[,RELINT:=findInterval(RELPOS,seq(0,1,by=intervallby))]

# set of meas
meas_sub <- meas[,.(detid, flow,length,interval,RELINT)]
names(meas_sub)[names(meas_sub) == 'flow'] <- 'arima.flow'


# only keep detector id
detlist <- detlist[,.(detid)]

# set everything into datatable again
setDT(meas)
setDT(meas_sub)
setDT(detlist)
setDT(clust)

# set the number of cores
cores <- 2
cl <- makeCluster(cores)
registerDoSNOW(cl)

# set the progress bar
pb <- txtProgressBar(min=1, max=20, style=3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress=progress)

# initiate parallel running
qmax <- list()
Qmax <- list()

system.time({

tt <-   foreach(i = 1:used.clust,.combine = "rbind.fill",
                .packages = c("data.table","plyr"),
                .inorder=FALSE, .options.snow=opts) %dopar% {

                  set.seed(100+i)

                  detlist_seed <- list()
                  detlist_seed_l <- list()

                  # merge two dataframes by adding cluster number into "meas" and "detlist"
                  meas_sub[coord_clust[,c(1,i+3),with=F],clust:=clust,on=c(detid="detid")]
                  detlist[coord_clust[,c(1,i+3),with=F],clust:=clust,on=c(detid="detid")]

                  setorder(detlist,clust)
                  measurements_agg <- list()
                  setkey(meas_sub,detid)

                  # random seed sample
                  for(k in 1:resamp.times){

                    # set sampling size from 0.2 to 0.8
                    for(j in c(0.2,0.4,0.6,0.8)){

                      # .N is an integer, length 1, containing the number of rows in the group.
                      detlist_seed[[j*5]] <- detlist[,.SD[sample(1:.N,round(.N*j,digits = 0),replace = FALSE),],by=clust][,sa_size:=j]
                    }

                    # combine 4 rows with random seed k and different sampling sizes
                    detlist_seed_l[[k]] <- rbindlist(detlist_seed)[,rs_sample:=k]

                  }
                  detlist_seed <- rbindlist(detlist_seed_l)

                  setkey(meas_sub,detid)
                  setkey(detlist_seed,detid)

                  meas_sub_sasize <- meas_sub[detlist_seed,allow.cartesian=T]

                  # calculate the MFD indicators?
                  system.time({
                    MFD_sample <- meas_sub_sasize[,.(flow_INT=weighted.mean(arima.flow,length,na.rm=TRUE)),
                                                  by=.(interval,clust,sa_size,RELINT,rs_sample)][,.(flow=mean(flow_INT, na.rm=T)),by=.(clust,interval,sa_size,rs_sample)]
                  })


                  qmax_sa <- MFD_sample[,.(flowmax=quantile(flow,0.975,na.rm = T)),by=.(clust,sa_size)][,clust_rs:=i]

  }

})

close(pb)
stopCluster(cl)

tt <- data.table(tt)

# compute the area of a function with values flowmax at the points sa_size.
acc <- tt[,trapz(sa_size,flowmax),by=.(clust,clust_rs)]

# compute the area among random seeds
acc <- acc[,mean(V1,na.rm = T),by=clust_rs]

# with the least area?
rightclust <- acc[V1==min(V1)]$clust_rs

# Set the color palette
setwd('./results/Zurich/ambuehl_method')

colors <- c('#57d3db', '#5784db', '#7957db', '#c957db', '#db579e', '#cb6843',
            '#db5f57', '#dbae57', '#b9db57', '#69db57', '#57db94', '#770001',
            '#ebced5', '#dbdbdb', '#dbdb57', '#db5768', '#db5784', '#db57b9',
            '#57a5db', '#a557db')


# Plot: clusters of LD on the map
right_coord <- coord_clust[,c(1,2,3,rightclust+3),with=F]
cluster.nr <- length(unique(right_coord$clust))
ggplot(data=right_coord) +
  geom_point(aes(lat, long, colour = as.factor(clust))) +
  scale_color_manual(values = colors)

ggsave('detectors_cluster.svg', dpi = 320, width = 8, height = 6)


# combine the cluster number with the measurement data
names(right_coord)[names(right_coord) == 'long'] <- 'coords.x1'
names(right_coord)[names(right_coord) == 'lat'] <- 'coords.x2'

right_coord[,N:=.N,by=clust]

meas[right_coord,clust:=clust,on=c(detid="detid")]
labels <- data.frame(meas$detid, meas$clust)
labels <- distinct(labels)
colnames(labels) <- c('detector_id', 'label')

# convert clust to factor with ordered levels based on color palette
measurements_agg <- meas[,.(flow=weighted.mean(flow,length,na.rm=TRUE),
                                         occ=weighted.mean(occ,length,na.rm=TRUE)),
                          by=.(interval,clust)]


# Plot: MFD
ggplot(data = measurements_agg, aes(occ, flow, color = factor(clust))) +
  geom_point() +
  scale_color_manual(values = colors) +
  facet_wrap(~clust) +
  guides(color = FALSE) +
  xlab("Occupancy (%)") +
  ylab("Flow (vehicles per hour)")

ggsave('MFD_cluster.svg', dpi = 320, width = 8, height = 6)

# Save the results
write.csv(meas, 'ambuehl_results.csv', row.names = FALSE)
write.csv(labels, 'ambuehl_labels.csv', row.names = FALSE)

