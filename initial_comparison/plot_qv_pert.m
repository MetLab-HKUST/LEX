%%% This script plots the warm bubble case in CM1 and LEX
%% first row: time = 0.0 min
qvpcm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000001.nc", "qv") ;
qvplex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0000.nc", "qv_now");
qvplex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0000.nc", "qv_now") ;

x_cm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000001.nc", "xh");
x_lex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0000.nc", "x");
x_lex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0000.nc", "x");
z_cm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000001.nc", "zh");
z_lex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0000.nc", "z");
z_lex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0000.nc", "z");

tiledlayout(4, 3, "TileSpacing", "compact")
nexttile
[X, Y] = meshgrid(x_cm1, z_cm1);
contourf(X, Y, squeeze(1000.0*qvpcm1(:,120,:,1))', 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
title("CM1 (\Deltax = 100m)")
text(-5.7, 6, "t = 0.0 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex1/1000-12.0, z_lex1/1000);
contourf(X, Y, squeeze(1000.0*qvplex1(:,120,:,1)), 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
title("LEX (\Deltax = 100m)")
text(-5.7, 6, "t = 0.0 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex2/1000-12.0, z_lex2/1000);
contourf(X, Y, squeeze(1000.0*qvplex2(:,20,:,1)), 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
title("LEX (\Deltax = 600m)")
text(-5.7, 6, "t = 0.0 min")
grid on
pbaspect([1,1,1])

%% second row: time = 10.0 min
qvpcm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000041.nc", "qv") ;
qvplex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0040.nc", "qv_now") ;
qvplex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0040.nc", "qv_now") ;

nexttile
[X, Y] = meshgrid(x_cm1, z_cm1);
contourf(X, Y, squeeze(1000.0*qvpcm1(:,120,:,1))', 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 6, "t = 10 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex1/1000-12.0, z_lex1/1000);
contourf(X, Y, squeeze(1000.0*qvplex1(:,120,:,1)), 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 6, "t = 10 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex2/1000-12.0, z_lex2/1000);
contourf(X, Y, squeeze(1000.0*qvplex2(:,20,:,1)), 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 6, "t = 10 min")
grid on
pbaspect([1,1,1])

%% second row: time = 20.0 min
qvpcm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000081.nc", "qv") ;
qvplex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0080.nc", "qv_now") ;
qvplex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0080.nc", "qv_now") ;

nexttile
[X, Y] = meshgrid(x_cm1, z_cm1);
contourf(X, Y, squeeze(1000.0*qvpcm1(:,120,:,1))', 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 6, "t = 20 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex1/1000-12.0, z_lex1/1000);
contourf(X, Y, squeeze(1000.0*qvplex1(:,120,:,1)), 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 6, "t = 20 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex2/1000-12.0, z_lex2/1000);
contourf(X, Y, squeeze(1000.0*qvplex2(:,20,:,1)), 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 6, "t = 20 min")
grid on
pbaspect([1,1,1])

%% second row: time = 30.0 min
qvpcm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000121.nc", "qv") ;
qvplex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0120.nc", "qv_now");
qvplex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0120.nc", "qv_now") ;

nexttile
[X, Y] = meshgrid(x_cm1, z_cm1);
contourf(X, Y, squeeze(1000.0*qvpcm1(:,120,:,1))', 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 6, "t = 30 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex1/1000-12.0, z_lex1/1000);
contourf(X, Y, squeeze(1000.0*qvplex1(:,120,:,1)), 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 6, "t = 30 min")
grid on
pbaspect([1,1,1])

h = nexttile;
[X, Y] = meshgrid(x_lex2/1000-12.0, z_lex2/1000);
contourf(X, Y, squeeze(1000.0*qvplex2(:,20,:,1)), 'LineColor', 'none')
colormap(flip(centered2("RdBu", 40)))
clim([0, 2.3])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 6, "t = 30 min")
grid on
pbaspect([1,1,1])

cbh = colorbar(h); 
cbh.Layout.Tile = 'south';
cbh.Label.String = "q_v (g/kg)";


