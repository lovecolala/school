function ant = resort(ant, opts)
tt = struct2table(ant);

[~, sorting] = sortrows(tt(:, 2));
ant = ant(sorting);
ant = ant(1:opts.nAnt); 
end