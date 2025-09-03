load("theta_pert.mat")

%% first row: time = 0.0 min
t = tiledlayout(2, 2, "TileSpacing", "Compact", "Units", "inches", ...
    "Position", [0.5 1.1 6 6]);

nexttile
[X, Y] = meshgrid(x_lex1/1000-12.0, z_lex1/1000);
contourf(X, Y, squeeze(thp_lex1(:,120,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)", "FontSize", 9, "FontName", "Aptos")
ylabel("z (km)", "FontSize", 9, "FontName", "Aptos")
ax = gca;
ax.FontSize = 9;
ax.FontName = "Aptos";
% title("LEX (\Deltax = 100m)")
text(-5.7, 0.75, "a) t = 0.0 min", "FontSize", 9, "FontName", "Aptos", "FontWeight", "bold")
grid on
pbaspect([1,1,1])

%% second row: time = 10.0 min
nexttile
contourf(X, Y, squeeze(thp_lex2(:,120,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)", "FontSize", 9, "FontName", "Aptos")
ylabel("z (km)", "FontSize", 9, "FontName", "Aptos")
ax = gca;
ax.FontSize = 9;
ax.FontName = "Aptos";
text(-5.7, 0.75, "b) t = 10 min", "FontSize", 9, "FontName", "Aptos", "FontWeight", "bold")
grid on
pbaspect([1,1,1])

%% third row: time = 20.0 min
nexttile
contourf(X, Y, squeeze(thp_lex3(:,120,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)", "FontSize", 9, "FontName", "Aptos")
ylabel("z (km)", "FontSize", 9, "FontName", "Aptos")
ax = gca;
ax.FontSize = 9;
ax.FontName = "Aptos";
text(-5.7, 0.75, "c) t = 20 min", "FontSize", 9, "FontName", "Aptos", "FontWeight", "bold")
grid on
pbaspect([1,1,1])

%% fourth row: time = 30.0 min
h = nexttile;
contourf(X, Y, squeeze(thp_lex4(:,120,:,1)), 'LineColor', 'none')
colormap(centered("RdBu", 40))
clim([0, 1])
xlim([-6, 6])
xticks([-6, -3, 0, 3, 6])
ylim([0.15, 12])
xlabel("x (km)", "FontSize", 9, "FontName", "Aptos")
ylabel("z (km)", "FontSize", 9, "FontName", "Aptos")
ax = gca;
ax.FontSize = 9;
ax.FontName = "Aptos";
text(-5.7, 0.75, "d) t = 30 min", "FontSize", 9, "FontName", "Aptos", "FontWeight", "bold")
grid on
pbaspect([1,1,1])

cbh = colorbar(h); 
cbh.Layout.Tile = 'south';
cbh.Label.String = "\theta' (K)";
cbh.FontSize = 9;
cbh.FontName = "Aptos";

%% save figure as PDF
exportgraphics(t, "theta_perturbation.jpg", "ContentType", "image", ... 
    "Resolution", 600, "Width", 7, "Height", "auto", "Units", "inches", ...
    "PreserveAspectRatio", "on")
