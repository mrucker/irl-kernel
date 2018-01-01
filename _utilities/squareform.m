function s = squareform(m)
    [r, c] = size(m);
    distanceMatrix = zeros(r);
    distanceMatrix(tril(true(r),-1)) = distances;
    distanceMatrix = distanceMatrix + distanceMatrix';
end