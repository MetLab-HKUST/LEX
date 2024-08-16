%%% This script plots the warm bubble case in CM1 and LEX
%% first row: time = 0.0 min
thp_cm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000001.nc", "thpert");
thp_lex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0000.nc", "theta_now") - ...
    ncread("LEX_dx100dz100_no_noise\lex_reference_state.nc", "theta0");
thp_lex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0000.nc", "theta_now") - ...
    ncread("LEX_dx600dz300_no_noise\lex_reference_state.nc", "theta0");

x_cm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000001.nc", "xh");
x_lex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0000.nc", "x");
x_lex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0000.nc", "x");
z_cm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000001.nc", "zh");
z_lex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0000.nc", "z");
z_lex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0000.nc", "z");

tiledlayout(4, 3, "TileSpacing", "compact")
nexttile
[X, Y] = meshgrid(x_cm1, z_cm1);
contourf(X, Y, squeeze(thp_cm1(:,120,:,1))', 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
title("CM1 (\Deltax = 100m)")
text(-5.7, 0.75, "t = 0.0 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex1/1000-12.0, z_lex1/1000);
contourf(X, Y, squeeze(thp_lex1(:,120,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
title("LEX (\Deltax = 100m)")
text(-5.7, 0.75, "t = 0.0 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex2/1000-12.0, z_lex2/1000);
contourf(X, Y, squeeze(thp_lex2(:,20,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
title("LEX (\Deltax = 600m)")
text(-5.7, 0.75, "t = 0.0 min")
grid on
pbaspect([1,1,1])

%% second row: time = 10.0 min
thp_cm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000041.nc", "thpert");
thp_lex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0040.nc", "theta_now") - ...
    ncread("LEX_dx100dz100_no_noise\lex_reference_state.nc", "theta0");
thp_lex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0040.nc", "theta_now") - ...
    ncread("LEX_dx600dz300_no_noise\lex_reference_state.nc", "theta0");

nexttile
[X, Y] = meshgrid(x_cm1, z_cm1);
contourf(X, Y, squeeze(thp_cm1(:,120,:,1))', 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 0.75, "t = 10 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex1/1000-12.0, z_lex1/1000);
contourf(X, Y, squeeze(thp_lex1(:,120,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 0.75, "t = 10 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex2/1000-12.0, z_lex2/1000);
contourf(X, Y, squeeze(thp_lex2(:,20,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 0.75, "t = 10 min")
grid on
pbaspect([1,1,1])

%% second row: time = 20.0 min
thp_cm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000081.nc", "thpert");
thp_lex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0080.nc", "theta_now") - ...
    ncread("LEX_dx100dz100_no_noise\lex_reference_state.nc", "theta0");
thp_lex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0080.nc", "theta_now") - ...
    ncread("LEX_dx600dz300_no_noise\lex_reference_state.nc", "theta0");

nexttile
[X, Y] = meshgrid(x_cm1, z_cm1);
contourf(X, Y, squeeze(thp_cm1(:,120,:,1))', 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 0.75, "t = 20 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex1/1000-12.0, z_lex1/1000);
contourf(X, Y, squeeze(thp_lex1(:,120,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 0.75, "t = 20 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex2/1000-12.0, z_lex2/1000);
contourf(X, Y, squeeze(thp_lex2(:,20,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 0.75, "t = 20 min")
grid on
pbaspect([1,1,1])

%% second row: time = 30.0 min
thp_cm1 = ncread("CM1_dx100dz100_no_noise\cm1out_000121.nc", "thpert");
thp_lex1 = ncread("LEX_dx100dz100_no_noise\lex_out_0120.nc", "theta_now") - ...
    ncread("LEX_dx100dz100_no_noise\lex_reference_state.nc", "theta0");
thp_lex2 = ncread("LEX_dx600dz300_no_noise\lex_out_0120.nc", "theta_now") - ...
    ncread("LEX_dx600dz300_no_noise\lex_reference_state.nc", "theta0");

nexttile
[X, Y] = meshgrid(x_cm1, z_cm1);
contourf(X, Y, squeeze(thp_cm1(:,120,:,1))', 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 0.75, "t = 30 min")
grid on
pbaspect([1,1,1])

nexttile
[X, Y] = meshgrid(x_lex1/1000-12.0, z_lex1/1000);
contourf(X, Y, squeeze(thp_lex1(:,120,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 0.75, "t = 30 min")
grid on
pbaspect([1,1,1])

h = nexttile;
[X, Y] = meshgrid(x_lex2/1000-12.0, z_lex2/1000);
contourf(X, Y, squeeze(thp_lex2(:,20,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)")
ylabel("z (km)")
text(-5.7, 0.75, "t = 30 min")
grid on
pbaspect([1,1,1])

cbh = colorbar(h); 
cbh.Layout.Tile = 'south';
cbh.Label.String = "\theta' (K)";