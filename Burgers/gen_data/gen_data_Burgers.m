%% Burgers equation
% % u_t + u*u_t —（0.01/pi)*u_xx =0
% % u(0,x) = -sin(pi*x), u(t,-1) = u(t,1) = 0
nn = 256;
steps = 200;

dom = [-1 1]; x = chebfun('x',dom); tspan = linspace(0,1,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) + 0.01/pi*diff(u,2);
S.nonlin = @(u) - 0.5*diff(u.^2); % spin cannot parse "u.*diff(u)"
S.init = -sin(pi*x);
u = spin(S,nn,1e-4);

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(-1,1,nn+1);
x = x(1:end-1);
t = tspan;
pcolor(t,x,real(usol)); shading interp, axis tight, colormap(jet);
save('Burgers_solution.mat','t','x','usol')
