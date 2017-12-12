function [spikes, stimuli] = runBinNoiseExpt(kernel, duration)

%Simplified simulation of a white noise (reverse correlation)
%experiment, but with binary random noise (to illustrate biases).
%The KERNEL is a spatial weight vector that will be applied at each
%time step, and DURATION is an integer specifying the total number of
%random stimuli that will be shown.  The function returns SPIKES, a
%binary vector indicating which stimuli produced spikes, and STIMULI,
%a matrix whose rows contain the Gaussian white noise simuli.

%% If kernel is not 1-dimensional, make it one-dimensional
kernel = kernel(:);

% unit variance binary noise
stimuli = 2*double(rand(duration, length(kernel)) < 0.5) - 1;

projection = stimuli * kernel;

%nl = @(proj) 0.5 * (1 + erf(2*(proj-1)));
nl = @(proj) 0.5 * (1 + erf(1.5*(proj-2)));

% generate spikes by drawing from a uniform distribution and
% comparing to the probability
spikes = rand(size(projection)) < nl(projection) ;

return;
