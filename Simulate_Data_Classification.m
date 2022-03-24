function [X, y] = Simulate_Data_Classification(varargin)
%% Generate a random n-class classification problem.

% This initially creates clusters of points normally distributed (std=1)
% about vertices of an ``n_informative``-dimensional hypercube with sides of
% length ``2*class_sep`` and assigns an equal number of clusters to each
% class. It introduces interdependence between these features and adds
% various types of further noise to the data.
% 
% Without shuffling, `X` horizontally stacks features in the following
% order: the primary ``n_informative`` features, followed by ``n_redundant``
% linear combinations of the informative features, followed by ``n_repeated``
% duplicates, drawn randomly with replacement from the informative and
% redundant features. The remaining features are filled with random noise.
% Thus, without shuffling, all useful features are contained in the columns
% `X[:, :n_informative + n_redundant + n_repeated]`.

% Parameters
%     ----------
%     n_samples : int, optional (default=100)
%         The number of samples.
% 
%     n_features : int, optional (default=20)
%         The total number of features. These comprise ``n_informative``
%         informative features, ``n_redundant`` redundant features,
%         ``n_repeated`` duplicated features and
%         ``n_features-n_informative-n_redundant-n_repeated`` useless features
%         drawn at random.
% 
%     n_informative : int, optional (default=2)
%         The number of informative features. Each class is composed of a number
%         of gaussian clusters each located around the vertices of a hypercube
%         in a subspace of dimension ``n_informative``. For each cluster,
%         informative features are drawn independently from  N(0, 1) and then
%         randomly linearly combined within each cluster in order to add
%         covariance. The clusters are then placed on the vertices of the
%         hypercube.
% 
%     n_redundant : int, optional (default=2)
%         The number of redundant features. These features are generated as
%         random linear combinations of the informative features.
% 
%     n_repeated : int, optional (default=0)
%         The number of duplicated features, drawn randomly from the informative
%         and the redundant features.
% 
%     n_classes : int, optional (default=2)
%         The number of classes (or labels) of the classification problem.
% 
%     n_clusters_per_class : int, optional (default=2)
%         The number of clusters per class.
% 
%     weights : list of floats or None (default=None)
%         The proportions of samples assigned to each class. If None, then
%         classes are balanced. Note that if ``len(weights) == n_classes - 1``,
%         then the last class weight is automatically inferred.
%         More than ``n_samples`` samples may be returned if the sum of
%         ``weights`` exceeds 1.
% 
%     flip_y : float, optional (default=0.01)
%         The fraction of samples whose class are randomly exchanged. Larger
%         values introduce noise in the labels and make the classification
%         task harder.
% 
%     class_sep : float, optional (default=1.0)
%         The factor multiplying the hypercube size.  Larger values spread
%         out the clusters/classes and make the classification task easier.
% 
%     hypercube : boolean, optional (default=True)
%         If True, the clusters are put on the vertices of a hypercube. If
%         False, the clusters are put on the vertices of a random polytope.
% 
%     shift : float, array of shape [1, n_features] or NAN, optional (default=0.0)
%         Shift features by the specified value. If Nan, then features
%         are shifted by a random value drawn in [-class_sep, class_sep].
% 
%     scale : float, array of shape [1, n_features] or NAN, optional (default=1.0)
%         Multiply features by the specified value. If Nan, then features
%         are scaled by a random value drawn in [1, 100]. Note that scaling
%         happens after shifting.
% 
%     shuffle : boolean, optional (default=True)
%         Shuffle the samples and the features.
% 
%     random_state : int, RandomState instance or Nan (default)
%         Determines random number generation for dataset creation. Pass an int
%         for reproducible output across multiple function calls.
%         See :term:`Glossary <random_state>`.
% 
%     Returns
%     -------
%     X : array of shape [n_samples, n_features]
%         The generated samples.
% 
%     y : array of shape [n_samples]
%         The integer labels for class membership of each sample.

% Example:
%      [features, response] = Simulate_Data_Classification(...
%                                 'nsamples', 200, ...
%                                 'nfeatures', 5, ...
%                                 'ninformative', 5, ...
%                                 'nredundant', 0, ...
%                                 'nclasses', 3, ...
%                                 'weights', [0.2, 0.3, 0.8]);
%      [X, y] = Simulate_Data_Classification(
%                           'nfeatures', 2, ...
%                           'ninformative', 2, ...
%                           'nredundant', 0, ...
%                           'nclusters_per_class', 1, ...
%                           'nclasses', 3);

% Translated from Python package 'scikit-learn' by Jian Wang
% jian.k.wang@foxmail.com
% Mar-24-2022

%% Default parameters
p = inputParser;
default_nsamples = 100;
default_nfeatures = 20;
default_ninformative = 2;
default_nredundant = 2;
default_nrepeated = 0;
default_nclasses = 2;
default_nclusters_per_class = 2;
default_weights = nan;
default_flipy = 0.01;
default_class_sep = 1.0;
default_hypercube = 1;
default_shift = 0.0;
default_scale = 1.0;
default_shuffle = 1;
default_random_state = nan;

addParameter(p, 'nsamples', default_nsamples, @isnumeric);
addParameter(p, 'nfeatures', default_nfeatures, @isnumeric);
addParameter(p, 'ninformative', default_ninformative, @isnumeric);
addParameter(p, 'nredundant', default_nredundant, @isnumeric);
addParameter(p, 'nrepeated', default_nrepeated, @isnumeric);
addParameter(p, 'nclasses', default_nclasses, @isnumeric);
addParameter(p, 'nclusters_per_class', default_nclusters_per_class, @isnumeric);
addParameter(p, 'weights', default_weights, @isnumeric);
addParameter(p, 'flipy', default_flipy, @isnumeric);
addParameter(p, 'class_sep', default_class_sep, @isnumeric);
addParameter(p, 'hypercube', default_hypercube, @isnumeric);
addParameter(p, 'shift', default_shift, @isnumeric);
addParameter(p, 'scale', default_scale, @isnumeric);
addParameter(p, 'shuffle', default_shuffle, @isnumeric);
addParameter(p, 'random_state', default_random_state, @isnumeric);

parse(p, varargin{:});
nsamples = p.Results.nsamples;
nfeatures = p.Results.nfeatures;
ninformative = p.Results.ninformative;
nredundant = p.Results.nredundant;
nrepeated = p.Results.nrepeated;
nclasses = p.Results.nclasses;
nclusters_per_class = p.Results.nclusters_per_class;
weights = p.Results.weights;
flipy = p.Results.flipy;
class_sep = p.Results.class_sep;
hypercube = p.Results.hypercube;
shift = p.Results.shift;
scale = p.Results.scale;
shuffle = p.Results.shuffle;
random_state = p.Results.random_state;
%% 

if ~isnan(random_state)
    rng(random_state);
end

% Count features, clusters and samples
if ninformative + nredundant + nrepeated > nfeatures
        error("Number of informative, redundant and repeated features must ..." + ...
            "sum to less than the number of total features");
end

% Use log2 to avoid overflow errors
if ninformative < log2(nclasses * nclusters_per_class)
    error("nclasses * nclusters_per_class must be smaller or equal 2 ** ninformative");
end

if ~isnan(weights) && (numel(weights) ~= nclasses && numel(weights) ~= nclasses - 1) 
    error("Weights specified but incompatible with number of classes.");
end

nuseless = nfeatures - ninformative - nredundant - nrepeated;
nclusters = nclasses * nclusters_per_class;

if ~isnan(weights) && (numel(weights) == nclasses - 1)
    weights = [weights, 1.0 - sum(weights)];
end

if isnan(weights)
    weights = ones(1, nclasses)*1.0/nclasses;
    weights(end) = 1 - sum(weights(1:end - 1));
end

n_samples_per_cluster = arrayfun(@(k) floor(nsamples*weights(mod(k, nclasses)+1)/nclusters_per_class), 0:nclusters-1);

for i = 0:nsamples - sum(n_samples_per_cluster) - 1
    n_samples_per_cluster(mod(i, nclusters)+1) = n_samples_per_cluster(mod(i, nclusters)+1) + 1;
end
        
%% Initialize X and y
X = zeros(nsamples, nfeatures);
y = zeros(nsamples, 1);

centroids = Generate_hypercube(nclusters, ninformative, random_state);
centroids = centroids*(2*class_sep);        
centroids = centroids - class_sep;    

if hypercube ~= 1
    centroids = centroids.*repmat(rand(nclusters, 1), 1, size(centroids, 2));
    centroids = centroids.*repmat(rand(1, ninformative), size(centroids, 1), 1);
end

% Initially draw informative features from the standard normal
X(:, 1:ninformative) = randn(nsamples, ninformative);

% Create each cluster
stop = 0;
for k = 1:size(centroids, 1)
    centroid = centroids(k, :);
    start = stop + 1;
    stop = stop + n_samples_per_cluster(k);
    y(start:stop) = mod(k - 1, nclasses); % assign labels
    
    X_k = X(start:stop, 1:ninformative); % slice a view of the cluster
    A = 2*rand(ninformative, ninformative) - 1;
    X_k = X_k*A; % introduce random covariance
    X_k = X_k + repmat(centroid, size(X_k, 1), 1); % shift the cluster to a vertex
    X(start:stop, 1:ninformative) = X_k;
end

% Create redundant features
if nredundant > 0
    B = 2*rand(ninformative, nredundant) - 1;
    X(:, ninformative + 1:ninformative + nredundant) = X(:, 1:ninformative)*B;
end

% Repeat some features
if nrepeated > 0
    n = ninformative + nredundant;
    indices = ceil((n - 1)*rand(nrepeated, 1) + 0.5);
    X(:, n + 1:n + nrepeated) = X(:, indices);
end

% Fill useless features
if nuseless > 0
    X(:, end - nuseless + 1:end) = randn(nsamples, nuseless);
end

% Randomly replace labels
if flipy >= 0
    flip_mask = rand(nsamples, 1) < flipy;
    y(flip_mask) = randi(nclasses, sum(flip_mask), 1) - 1;
end

% Randomly shift and scale
if isnan(shift)
    shift = (2*rand(1, nfeatures) - 1)*class_sep;
elseif numel(shift) == 1
    shift = repmat(shift, 1, nfeatures);
elseif numel(shift) ~= nfeatures
    error('The size of shift vector should equal number of features');
end
shift = reshape(shift, 1, []);
X = X + repmat(shift, nsamples, 1);

if isnan(scale)
    scale = 1 + 100*rand(1, nfeatures);
elseif numel(scale) == 1
    scale = repmat(scale, 1, nfeatures);
elseif numel(scale) ~= nfeatures
    error('The size of scale vector should equal number of features');
end
scale = reshape(scale, 1, []);
X = X.*repmat(scale, nsamples, 1);

if shuffle
    % Randomly permute samples
    idxsample = randperm(nsamples);
    X = X(idxsample, :);
    y = y(idxsample);    
    % Randomly permute features
    idxfeatures = randperm(nfeatures);
    X = X(:, idxfeatures);    
end

end

%% 
function out = Generate_hypercube(samples, dimensions, seed)
% Returns distinct binary samples of length dimensions
if ~isnan(seed)
    rng(seed);
end
if dimensions > 30    
    temp = randi([0, 1], samples, dimensions - 30);
    res = Generate_hypercube(samples, 30, seed);
    out = [temp, res];
else
    out = randsample(0:2^dimensions - 1, samples, false);
    out = dec2bin(out, dimensions);
    out = (out == '1')/1;
end

end