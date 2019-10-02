clear all, close all, clc
T = readtable('final_countries_data.csv');
coords = [T.x_coord, T.y_coord];
figure(1); subplot(211);
for idx = 1:size(T,1)
    xx = coords(idx, 1); 
    yy = coords(idx, 2);
    scatter(xx, yy); hold on;
    text(xx, yy, T.CountryName(idx),'FontSize',6); hold on;
end
title('Countries'); grid on; 
set(gca,'yticklabel',[]); set(gca,'xticklabel',[])

colors = {'r', 'g', 'b', 'm'};
D = pdist(coords, 'euclidean');
Z = linkage(D, 'single');
clust = cluster(Z, 'maxclust', 4);

subplot(212);
for idx = 1:max(unique(clust))
    where = idx == clust;
    scatter(coords(where, 1), coords(where, 2), [], colors{idx}); 
    hold on;
end
title('Clusters'); grid on; 
set(gca,'yticklabel',[]); set(gca,'xticklabel',[])


