%%(CUDA Parallelized)
clear;
clc;
tic
%% Define Simulation Based oFF Source and Wavelength
f0         = 1e6;             % Frequency of Source        [Hertz]
Lf         = 10;              % Divisions per Wavelength   [unitless]
[Lx,Ly]    = deal(32,32);       % Wavelengths x,y            [unitless]
nt         = 3000;            % Number of time steps       [unitless]

%% Spatial and Temporal System
e0 = 8.854*10^-12;  % Permittivity of vacuum [farad/meter]
u0 = 4*pi*10^-7;    % Permeability of vacuum [henry/meter]
c0 = 1/(e0*u0)^.5;  % Speed of light         [meter/second]
L0  = c0/f0;        % Freespace Wavelength   [meter]
t0  = 1/f0;         % Source Period          [second]

[Nx,Ny] = deal(Lx*Lf,Ly*Lf);    % Points in x,y           [unitless]
x  = linspace(0,Lx,Nx+1)*L0;    % x vector                [meter]
y  = linspace(0,Ly,Ny+1)*L0;    % y vector                [meter]
[dx,dy] = deal(x(2),y(2));      % x,y,z increment         [meter]
dt = (dx^-2+dy^-2)^-.5/c0*.99;  % Time step CFL condition [second]

%% Initialize Vectors on GPU
Hx = gpuArray.zeros(Nx, Ny);     % Magnetic Field (x-component)
Hy = gpuArray.zeros(Nx, Ny);     % Magnetic Field (y-component)
Ez = gpuArray.zeros(Nx, Ny);     % Electric Field (z-component)

[udy, udx] = deal(dt/(u0*dy), dt/(u0*dx));  % H Field Coefficients
[edx, edy] = deal(dt/(e0*dx), dt/(e0*dy));  % E Field Coefficients
%% Precompute Source Terms
timeVector = gpuArray((1:nt) * dt);  % Time vector on GPU
source = sin(2 * pi * f0 * timeVector) .* exp(-0.5 * ((timeVector - 20 * dt) / (8 * dt)).^2);

% Source injection point
srcX = round(Nx / 2);
srcY = round(Ny / 2);

%% Time Evolution Without Loop
for t = 1:nt
    % Magnetic Field Update
    Hx(1:Nx-1,1:Ny-1) = Hx(1:Nx-1,1:Ny-1) - udy * diff(Ez(1:Nx-1,:), 1, 2);
    Hy(1:Nx-1,1:Ny-1) = Hy(1:Nx-1,1:Ny-1) + udx * diff(Ez(:,1:Ny-1), 1, 1);

    % Electric Field Update
    Ez(2:Nx-1,2:Ny-1) = Ez(2:Nx-1,2:Ny-1) + ...
        edx * diff(Hy(1:Nx-1,2:Ny-1), 1, 1) - edy * diff(Hx(2:Nx-1,1:Ny-1), 1, 2);

    % Inject Source
    Ez(srcX, srcY) = Ez(srcX, srcY) + source(t);

    % Visualization
    if mod(t, 10) == 0  % Plot every 10 time steps
        imagesc(x, y, gather(Ez)); drawnow;
    end
end
toc
