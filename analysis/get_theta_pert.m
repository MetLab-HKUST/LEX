
thp_lex1 = ncread("./lex_out_0000.nc", "th") - ...
    ncread("./lex_reference_state.nc", "theta0");
x_lex1 = ncread("./lex_out_0000.nc", "x");
z_lex1 = ncread("./lex_out_0000.nc", "z");

thp_lex2 = ncread("./lex_out_0010.nc", "th") - ...
    ncread("./lex_reference_state.nc", "theta0");

thp_lex3 = ncread("./lex_out_0020.nc", "th") - ...
    ncread("./lex_reference_state.nc", "theta0");

thp_lex4 = ncread("./lex_out_0030.nc", "th") - ...
    ncread("./lex_reference_state.nc", "theta0");

save("theta_pert.mat")

